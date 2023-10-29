

#1、读取mask_path下面的所有mask，然后将他们统一resize到512*512，共计101张图片
#2、mask内部的值分别只有0，1，2，然后计算101张图片像素值为1和像素值为2共有的部分，将它保存为all_cover.png
mask_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后/Drishti-GS1_files/my_gts_cropped'
from PIL import Image
import os
import numpy as np

mask_path = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后/Drishti-GS1_files/my_gts_cropped'
output_path = 'all_cover.png'


def resize_masks(mask_path):
    resized_masks = []
    for filename in os.listdir(mask_path):
        if filename.endswith('.png'):
            img = Image.open(os.path.join(mask_path, filename))
            img_resized = img.resize((512, 512))
            resized_masks.append(np.array(img_resized))
    return resized_masks


def compute_common_area(resized_masks):
    # Initialize the common areas
    common_area_1 = np.ones_like(resized_masks[0], dtype=np.uint8)
    common_area_2 = np.ones_like(resized_masks[0], dtype=np.uint8)

    for mask in resized_masks:
        common_area_1 = np.logical_and(common_area_1, mask == 1)
        common_area_2 = np.logical_and(common_area_2, mask == 2)

    # Assign values according to your requirements
    result = np.zeros_like(resized_masks[0], dtype=np.uint8)
    result[common_area_1] = 150
    result[common_area_2] = 255

    return Image.fromarray(result)


if __name__ == '__main__':
    resized_masks = resize_masks(mask_path)
    common_area_img = compute_common_area(resized_masks)
    common_area_img.save(output_path)

