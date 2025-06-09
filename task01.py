import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

def shuchu(tif_file, output_file='rgb_output_compressed.png', scale=0.1):
    # 打开 TIFF 文件
    with rasterio.open(tif_file) as src:
        bands = src.read()

    # 提取 RGB 波段（B04 红, B03 绿, B02 蓝）
    blue = bands[0].astype(float)   # B02
    green = bands[1].astype(float)  # B03
    red = bands[2].astype(float)    # B04

    # 组合成 RGB 图像
    rgb = np.dstack((red, green, blue))
    rgb = np.clip(rgb, 0, 10000)
    rgb = (rgb / 10000.0) * 255
    rgb = rgb.astype(np.uint8)

    # 压缩图像大小（例如 scale=0.1 表示压缩到原来的10%）
    height, width, _ = rgb.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    rgb_resized = resize(rgb, (new_height, new_width), preserve_range=True).astype(np.uint8)

    # 保存压缩后的图像
    plt.imsave(output_file, rgb_resized)
    print(f"压缩后的RGB图像已保存为: {output_file}（原始大小: {height}x{width} -> 新大小: {new_height}x{new_width}）")

if __name__ == '__main__':
    tif_path = r"D:\专业实习\cc1a50cc6138a053140d9660f7f624c8.par.temp"
    output_path = "rgb_output_compressed.png"
    if os.path.exists(tif_path):
        shuchu(tif_path, output_path, scale=0.1)  # 你可以调整 scale 比例，如 0.2 表示压缩到20%
    else:
        print(f"未找到指定文件: {tif_path}")
