from PIL import Image

def combine_images(image_paths, output_path):
    # 确保输入的图像路径数量为4
    if len(image_paths) != 4:
        raise ValueError("Exactly 4 image paths are required")

    # 加载图像
    images = [Image.open(image_path) for image_path in image_paths]

    # 获取单张图像的宽度和高度（假设所有图像大小相同）
    width, height = images[0].size

    # 创建一个新的图像画布，大小为四张图像的组合大小
    combined_image = Image.new('RGB', (2 * width, 2 * height))

    # 将四张图像按顺序粘贴到新画布上
    combined_image.paste(images[0], (0, 0))
    combined_image.paste(images[1], (width, 0))
    combined_image.paste(images[2], (0, height))
    combined_image.paste(images[3], (width, height))

    # 保存组合后的图像
    combined_image.save(output_path)

# 示例用法
image_paths = [
    r'draw\mslp_and_wind_2013110712+06h.png',
    r'draw\mslp_and_wind_2013110712+12h.png',
    r'draw\mslp_and_wind_2013110712+18h.png',
    r'draw\mslp_and_wind_2013110712+24h.png'
]
output_path = r'draw\mslp_and_wind_2013110712_combined.png'

combine_images(image_paths, output_path)