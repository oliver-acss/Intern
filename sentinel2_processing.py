import numpy as np
import rasterio
import cv2
import os
import matplotlib.pyplot as plt

def normalize_band(band_data):
    """
    将0-10000范围的波段数据归一化到0-255
    """
    min_val = np.min(band_data)
    max_val = np.max(band_data)
    normalized = ((band_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized

def adjust_image(image, brightness_factor=1.0, contrast_factor=1.0):
    """
    调整图像亮度和对比度
    brightness_factor: 亮度因子，1.0为原始亮度
    contrast_factor: 对比度因子，1.0为原始对比度
    """
    # 调整对比度
    mean = np.mean(image)
    contrast_adjusted = mean + contrast_factor * (image - mean)
    
    # 调整亮度
    brightness_adjusted = contrast_adjusted * brightness_factor
    
    # 确保值在0-255范围内
    return np.clip(brightness_adjusted, 0, 255).astype(np.uint8)

def plot_images(original, processed, title1="原始图像", title2="处理后的图像"):
    """
    显示原始图像和处理后的图像对比
    """
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title(title1)
    plt.axis('off')
    
    # 显示处理后的图像
    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title(title2)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_sentinel2_image(input_path, output_path):
    """
    处理哨兵2号图像
    参数:
        input_path: 输入图像路径
        output_path: 输出RGB图像路径
    """
    # 读取图像
    with rasterio.open(input_path) as src:
        # 读取所有波段
        bands_data = []
        for i in range(1, src.count + 1):
            band = src.read(i)
            bands_data.append(band)

    # 归一化每个波段到0-255
    normalized_bands = [normalize_band(band) for band in bands_data]

    # 创建RGB图像
    # 假设波段顺序为：B2(蓝), B3(绿), B4(红), B8(近红外), B11(短波红外)
    rgb_image = np.dstack((
        normalized_bands[2],  # 红波段
        normalized_bands[1],  # 绿波段
        normalized_bands[0]   # 蓝波段
    ))

    # 保存原始RGB图像
    original_rgb = rgb_image.copy()
    
    # 调整亮度和对比度
    rgb_image = adjust_image(rgb_image, brightness_factor=0.9, contrast_factor=0.9)

    # 显示图像对比
    plot_images(original_rgb, rgb_image)

    # 保存RGB图像
    cv2.imwrite(output_path, rgb_image)
    print(f"处理完成，RGB图像已保存至: {output_path}")

if __name__ == "__main__":
    # 创建输出目录
    output_dir = "image"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 使用示例
    input_path = "2019_1101_nofire_B2348_B12_10m_roi.tif"
    output_path = os.path.join(output_dir, "output_rgb_image.png")
    process_sentinel2_image(input_path, output_path) 