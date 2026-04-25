
"""
Build deterministic AIFS matched-data CSV from matched_data_ECMWF.csv.

Rules fixed by user:
1) Read matched_data_ECMWF.csv
2) Select sample units at forecast_hour=0 with:
   - years in [1985, 2024]
   - obs_wind_speed >= 34 kt
   - obs_pressure >= 1000 hPa   (change PRESSURE_FILTER_MODE if you want the opposite)
   - ens_member_count >= 50
3) Download / prepare ERA5 initial field for each selected sample
4) Pickle the input_state immediately
5) Run AIFS to the same lead time as the sample
6) Save MSLP + 10 m wind fields for every forecast state
7) Detect TC center by MSLP minimum within a moving search window
8) Write/update a new deterministic matched-data CSV, using ctrl_* columns for AIFS output

All downloaded/generated files are written OUTSIDE the AI-weather-models repository.
"""

import os
import json
import glob
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from get_data_aifs import AIFSDataDownloader
from run_aifs import AIFSRunner


# =====================================================================
# PATHS
# =====================================================================
def detect_repo_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current = script_dir
    for _ in range(8):
        if os.path.basename(current) == "AI-weather-models":
            return current
        current = os.path.dirname(current)
    # fallback: keep old behavior if repo root not found
    return os.path.dirname(os.path.abspath(__file__))


REPO_ROOT = detect_repo_root()
REPO_PARENT = os.path.dirname(REPO_ROOT)
OUTPUT_ROOT = os.path.join(REPO_PARENT, "AIFS_MATCHED_WORKFLOW")

MATCHED_DATA_FILE = "matched_data_ECMWF.csv"
RAW_INIT_DIR = os.path.join(OUTPUT_ROOT, "Input", "AIFS_raw")
PROC_INIT_DIR = os.path.join(OUTPUT_ROOT, "Input", "AIFS")
FORECAST_DIR = os.path.join(OUTPUT_ROOT, "Output", "AIFS")
FIELD_DIR = os.path.join(OUTPUT_ROOT, "Fields")
STATUS_DIR = os.path.join(OUTPUT_ROOT, "Status")
PICKLE_DIR = os.path.join(OUTPUT_ROOT, "Pickle")
CSV_OUT = os.path.join(OUTPUT_ROOT, "matched_data_AIFS.csv")
FAILED_JSONL = os.path.join(OUTPUT_ROOT, "failed_samples.jsonl")

# =====================================================================
# CONFIG
# =====================================================================
DEVICE = "cuda"
USE_HUGGINGFACE = False
MODEL_PATH = None

YEAR_START = 1985
YEAR_END = 2024

MIN_INIT_WIND_KT = 34.0
PRESSURE_THRESHOLD_HPA = 1000.0
PRESSURE_FILTER_MODE = "ge"  # "ge" for >=1000 hPa, "le" for <=1000 hPa
MIN_ENSEMBLE_MEMBERS = 50

SEARCH_RADIUS_DEG = 5.0
CENTER_WIND_RADIUS_DEG = 3.0

OUTPUT_COLUMNS = [
    "model", "sid", "base_time", "forecast_hour", "cyclone_name",
    "obs_lat", "obs_lon", "obs_pressure", "obs_wind_speed", "obs_rmw",
    "ctrl_lat", "ctrl_lon", "ctrl_pressure", "ctrl_wind_speed",
]


# =====================================================================
# HELPERS
# =====================================================================
def ensure_dirs():
    for d in [OUTPUT_ROOT, RAW_INIT_DIR, PROC_INIT_DIR, FORECAST_DIR, FIELD_DIR, STATUS_DIR, PICKLE_DIR]:
        os.makedirs(d, exist_ok=True)


def norm_lon(lon: float) -> float:
    x = (float(lon) + 180.0) % 360.0 - 180.0
    if x == -180.0:
        x = 180.0
    return x


def lon_diff(lon2: np.ndarray, lon1: float) -> np.ndarray:
    return (lon2 - lon1 + 180.0) % 360.0 - 180.0


def pressure_filter_ok(p: float) -> bool:
    if pd.isna(p):
        return False
    if PRESSURE_FILTER_MODE == "ge":
        return float(p) >= PRESSURE_THRESHOLD_HPA
    if PRESSURE_FILTER_MODE == "le":
        return float(p) <= PRESSURE_THRESHOLD_HPA
    raise ValueError(f"Unknown PRESSURE_FILTER_MODE: {PRESSURE_FILTER_MODE}")


def sample_key_str(sid: str, base_time: datetime) -> str:
    return f"{sid}_{base_time.strftime('%Y%m%d%H')}"


def save_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def append_failed(sample_key: str, err: str):
    with open(FAILED_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_key": sample_key, "error": err}, ensure_ascii=False) + "\n")


def extract_1d_field(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError("Scalar field is invalid.")
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] == 2:
            return np.asarray(arr[-1])
        if arr.shape[1] == 2:
            return np.asarray(arr[:, -1])
    return arr.reshape(-1)


def get_state_core_fields(state: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = extract_1d_field(state["latitudes"])
    lon = extract_1d_field(state["longitudes"])
    fields = state["fields"]
    u10 = extract_1d_field(fields["10u"])
    v10 = extract_1d_field(fields["10v"])
    msl = extract_1d_field(fields["msl"])
    return lat, lon, u10, v10, msl


def extract_state_valid_datetime(state: Dict, init_dt: datetime, state_index: int) -> datetime:
    if "date" in state and state["date"] is not None:
        if isinstance(state["date"], datetime):
            return state["date"]
        return datetime.fromisoformat(str(state["date"]))
    return init_dt + timedelta(hours=6 * state_index)


@dataclass
class CenterResult:
    lat: float
    lon: float
    pressure_hpa: float
    max10mwind: float
    method: str


def detect_center_from_state(state: Dict, prev_guess: Tuple[float, float]) -> CenterResult:
    lat, lon, u10, v10, msl = get_state_core_fields(state)

    lon = np.array([norm_lon(x) for x in lon], dtype=float)
    lat = lat.astype(float)
    msl = msl.astype(float)
    u10 = u10.astype(float)
    v10 = v10.astype(float)
    wspd = np.sqrt(u10 ** 2 + v10 ** 2)

    if np.nanmax(msl) > 2000:
        msl_hpa = msl / 100.0
    else:
        msl_hpa = msl.copy()

    guess_lat, guess_lon = prev_guess
    mask = (np.abs(lat - guess_lat) <= SEARCH_RADIUS_DEG) & (np.abs(lon_diff(lon, guess_lon)) <= SEARCH_RADIUS_DEG)
    if not np.any(mask):
        mask = np.isfinite(msl_hpa)

    masked_msl = np.where(mask, msl_hpa, np.nan)
    if not np.any(np.isfinite(masked_msl)):
        raise RuntimeError("No finite MSLP values found in search window.")

    idx = int(np.nanargmin(masked_msl))
    c_lat = float(lat[idx])
    c_lon = norm_lon(float(lon[idx]))
    c_prs = float(msl_hpa[idx])

    wind_mask = (np.abs(lat - c_lat) <= CENTER_WIND_RADIUS_DEG) & (np.abs(lon_diff(lon, c_lon)) <= CENTER_WIND_RADIUS_DEG)
    if not np.any(wind_mask):
        wind_mask = mask
    c_wind = float(np.nanmax(np.where(wind_mask, wspd, np.nan)))

    return CenterResult(
        lat=c_lat,
        lon=c_lon,
        pressure_hpa=c_prs,
        max10mwind=c_wind,
        method="mslp_min",
    )


def save_mslp_wind_npz(state: Dict, out_path: str):
    lat, lon, u10, v10, msl = get_state_core_fields(state)
    wspd = np.sqrt(u10 ** 2 + v10 ** 2)
    np.savez_compressed(
        out_path,
        lat=lat,
        lon=lon,
        u10=u10,
        v10=v10,
        wspd=wspd,
        msl=msl,
    )


def update_csv(csv_path: str, new_df: pd.DataFrame):
    new_df = new_df[OUTPUT_COLUMNS].copy()
    if os.path.exists(csv_path):
        old = pd.read_csv(csv_path)
        merged = pd.concat([old, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["model", "sid", "base_time", "forecast_hour"], keep="last")
    else:
        merged = new_df.copy()
    merged = merged.sort_values(["sid", "base_time", "forecast_hour"]).reset_index(drop=True)
    merged.to_csv(csv_path, index=False)


def load_pickled_input_state(pkl_path: str) -> Optional[Dict]:
    if not os.path.exists(pkl_path):
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def save_pickled_input_state(input_state: Dict, pkl_path: str):
    with open(pkl_path, "wb") as f:
        pickle.dump(input_state, f)


def build_state_list_from_saved_outputs(runner: AIFSRunner, init_dt: datetime, input_state: Dict) -> List[Dict]:
    pattern = os.path.join(FORECAST_DIR, f"output_aifs_{init_dt.strftime('%Y%m%d%H')}+*h.npz")
    files = sorted(glob.glob(pattern))
    states = []

    has_f000 = False
    for fp in files:
        if fp.endswith("+000h.npz"):
            has_f000 = True
        st = runner.load_forecast_state(fp)
        if st is not None:
            states.append(st)

    if not has_f000:
        states.insert(0, {
            "date": input_state["date"],
            "latitudes": input_state.get("latitudes", np.array([])),
            "longitudes": input_state.get("longitudes", np.array([])),
            "fields": input_state["fields"],
        })

    states = sorted(states, key=lambda s: extract_state_valid_datetime(s, init_dt, 0))
    return states


def build_aifs_records_for_sample(sample0: pd.Series, sample_group: pd.DataFrame, forecast_states: List[Dict]) -> pd.DataFrame:
    init_dt = pd.to_datetime(sample0["base_time"]).to_pydatetime()
    sid = str(sample0["sid"])
    cyclone_name = sample0.get("cyclone_name", "")
    prev_guess = (float(sample0["obs_lat"]), norm_lon(float(sample0["obs_lon"])))

    rows = []
    out_dir = os.path.join(FIELD_DIR, sample_key_str(sid, init_dt))
    os.makedirs(out_dir, exist_ok=True)

    for i, state in enumerate(forecast_states):
        valid_dt = extract_state_valid_datetime(state, init_dt, i)
        fh = int(round((valid_dt - init_dt).total_seconds() / 3600.0))

        field_file = os.path.join(out_dir, f"fields_f{fh:03d}.npz")
        if not os.path.exists(field_file):
            save_mslp_wind_npz(state, field_file)

        center = detect_center_from_state(state, prev_guess)
        prev_guess = (center.lat, center.lon)

        sub = sample_group[sample_group["forecast_hour"] == fh]
        if len(sub) > 0:
            r = sub.iloc[0]
            obs_lat = r.get("obs_lat", np.nan)
            obs_lon = r.get("obs_lon", np.nan)
            obs_pressure = r.get("obs_pressure", np.nan)
            obs_wind_speed = r.get("obs_wind_speed", np.nan)
            obs_rmw = r.get("obs_rmw", np.nan)
        else:
            obs_lat = obs_lon = obs_pressure = obs_wind_speed = obs_rmw = np.nan

        rows.append({
            "model": "AIFS",
            "sid": sid,
            "base_time": init_dt.strftime("%Y%m%d%H"),
            "forecast_hour": fh,
            "cyclone_name": cyclone_name,
            "obs_lat": obs_lat,
            "obs_lon": obs_lon,
            "obs_pressure": obs_pressure,
            "obs_wind_speed": obs_wind_speed,
            "obs_rmw": obs_rmw,
            "ctrl_lat": center.lat,
            "ctrl_lon": center.lon,
            "ctrl_pressure": center.pressure_hpa,
            "ctrl_wind_speed": center.max10mwind,
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("forecast_hour").reset_index(drop=True)
    return out_df


def process_one_sample(sample0: pd.Series, sample_group: pd.DataFrame, downloader: AIFSDataDownloader, runner: AIFSRunner) -> pd.DataFrame:
    sid = str(sample0["sid"])
    base_time = pd.to_datetime(sample0["base_time"]).to_pydatetime()
    skey = sample_key_str(sid, base_time)
    lead_time = int(sample_group["forecast_hour"].max())

    status_json = os.path.join(STATUS_DIR, f"{skey}.json")
    input_pkl = os.path.join(PICKLE_DIR, f"{skey}_input_state.pkl")

    if os.path.exists(input_pkl):
        input_state = load_pickled_input_state(input_pkl)
    else:
        input_state = downloader.process_era5_data(
            datetime_str=base_time.strftime("%Y%m%d%H"),
            skip_existing=True,
        )
        if input_state is None or "date" not in input_state:
            raise RuntimeError(f"Failed to prepare ERA5 input_state for {skey}.")
        save_pickled_input_state(input_state, input_pkl)

    runner.run_forecast(
        input_state=input_state,
        lead_time=lead_time,
        save_outputs=True,
        datetime_str=base_time.strftime("%Y%m%d%H"),
        skip_existing=True,
    )

    forecast_states = build_state_list_from_saved_outputs(runner, base_time, input_state)
    if len(forecast_states) == 0:
        raise RuntimeError(f"No forecast states available for {skey}.")

    out_df = build_aifs_records_for_sample(sample0, sample_group, forecast_states)

    save_json(status_json, {
        "sample_key": skey,
        "status": "done",
        "lead_time": lead_time,
        "n_rows": int(len(out_df)),
        "updated_at": datetime.now().isoformat(),
    })
    return out_df


def main():
    ensure_dirs()

    if not os.path.exists(MATCHED_DATA_FILE):
        raise FileNotFoundError(f"Cannot find {MATCHED_DATA_FILE}")

    print(f"Repository root : {REPO_ROOT}")
    print(f"Output root     : {OUTPUT_ROOT}")

    df = pd.read_csv(MATCHED_DATA_FILE)
    df["base_time"] = pd.to_datetime(df["base_time"].astype(str), format="%Y%m%d%H")

    df0 = df[df["forecast_hour"] == 0].copy()
    df0 = df0[(df0["base_time"].dt.year >= YEAR_START) & (df0["base_time"].dt.year <= YEAR_END)]
    df0 = df0[df0["obs_wind_speed"].notna() & (df0["obs_wind_speed"] >= MIN_INIT_WIND_KT)]
    df0 = df0[df0["obs_pressure"].notna() & df0["obs_pressure"].apply(pressure_filter_ok)]
    df0 = df0[df0["ens_member_count"].notna() & (df0["ens_member_count"] >= MIN_ENSEMBLE_MEMBERS)]
    df0 = df0.sort_values(["sid", "base_time"]).reset_index(drop=True)

    done_keys = set()
    if os.path.exists(CSV_OUT):
        old = pd.read_csv(CSV_OUT)
        if len(old) > 0:
            for sid, bt in old[["sid", "base_time"]].drop_duplicates().itertuples(index=False):
                done_keys.add(f"{sid}_{pd.to_datetime(str(bt), format='%Y%m%d%H').strftime('%Y%m%d%H')}")

    downloader = AIFSDataDownloader(input_dir=PROC_INIT_DIR, raw_input_dir=RAW_INIT_DIR)
    runner = AIFSRunner(
        device=DEVICE,
        model_path=MODEL_PATH,
        output_dir=FORECAST_DIR,
        use_huggingface=USE_HUGGINGFACE,
    )
    if getattr(runner, "runner", None) is None:
        raise RuntimeError("AIFS runner initialization failed.")

    total = len(df0)
    print(f"Selected initial samples: {total}")

    for i, sample0 in enumerate(df0.itertuples(index=False), start=1):
        s = pd.Series(sample0._asdict())
        sid = str(s["sid"])
        base_time = pd.to_datetime(s["base_time"]).to_pydatetime()
        skey = sample_key_str(sid, base_time)
        print(f"\n[{i}/{total}] Processing {skey}")

        if skey in done_keys:
            print("  Skip: already present in output CSV.")
            continue

        sample_group = df[(df["sid"] == sid) & (df["base_time"] == pd.Timestamp(base_time))].copy()

        try:
            out_df = process_one_sample(s, sample_group, downloader, runner)
            update_csv(CSV_OUT, out_df)
            print(f"  Done: wrote {len(out_df)} rows to {CSV_OUT}")
        except Exception as e:
            append_failed(skey, str(e))
            save_json(os.path.join(STATUS_DIR, f"{skey}.json"), {
                "sample_key": skey,
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.now().isoformat(),
            })
            print(f"  Failed: {e}")
            continue

    print("\nAll finished.")
    print(f"Output CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
