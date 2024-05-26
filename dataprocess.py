from PIL import Image
import os

def resize_images(input_folder, output_folder, interpolation_method):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            # 打开图片文件
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            # 进行降采样
            resized_img = img.resize((512, 512), resample=interpolation_method)

            # 保存降采样后的图片到输出文件夹
            output_path = os.path.join(output_folder, filename)
            resized_img.save(output_path)

            print(f"Resized {filename} and saved to {output_path}")

if __name__ == "__main__":
    # 输入文件夹路径
    input_rgb_folder = "2_Ortho_RGB"
    input_label_folder = "5_Labels_all"

    # 输出文件夹路径
    output_rgb_folder = "trainPics\pics"
    output_label_folder = "trainPics\label"

    # 调整RGB图片
    resize_images(input_rgb_folder, output_rgb_folder, Image.BILINEAR)

    # 调整标签图片
    resize_images(input_label_folder, output_label_folder, Image.NEAREST)

