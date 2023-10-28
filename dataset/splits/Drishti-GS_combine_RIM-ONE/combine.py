import numpy as np
from PIL import Image
import os
import random

Drishti_img_path = '../../../data/Drishti-GS/imgs'
Drishti_gt_path = '../../../data/Drishti-GS/gts'

RIM_img_path = '../../../data/RIM-ONE/imgs'
RIM_gt_path = '../../../data/RIM-ONE/gts'


Dr_combine_img_path = '../../../data/Drishti-GS_Combine_RIM-ONE/imgs'
Dr_combine_gt_path = '../../../data/Drishti-GS_Combine_RIM-ONE/gts'
Dr_combine_gt_points_path = '../../../data/Drishti-GS_Combine_RIM-ONE/points'



def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 0, 0])
    cmap[2] = np.array([0, 255, 0])
    cmap[3] = np.array([255, 255, 255])

    return cmap

"""
我希望使用PIL完成以下要求
已知Drishti_img_path和Drishti_gt_path中都是101张照片；RIM_img_path和RIM_gt_path都是159张
(1)现在需要读取Drishti_img_path的一张图片假设名字为A.png,从RIM_img_path中随机读取一张出来假设为B.jpg，然后将两张图片拼接在一起,Drishti_img_path的在左边，
RIM_img_path的在右边，如果高度不一致，就按照左边的高度为准，然后命名为A_B.png，存储在Dr_combine_img_path下面

(2)在Drishti_gt_path中读取步骤(1)中Drishti_img_path同名(A.png)的照片mask1，在RIM_gt_path中读取步骤(1)中RIM_img_path同名(img是B.jpg,mask是B.png)的照片mask2
然后将两张图片拼接在一起,Drishti_gt_path的在左边，RIM_gt_path的在右边，如果高度不一致，就按照左边的高度为准，然后命名为A_B.png，存储在Dr_combine_gt_path下面，存储的时候使用提供的调色板

(3)然后生成一个像素值全是3的灰度图像mask3，他的大小和mask2图片大小一致，
接着也是拼接，mask1在左，mask3在右，高度以mask1为准，然后命名为A_B.PNG(这里的名字和步骤(1)的一样)，
存储在Dr_combine_gt_points_path下面,存储的时候使用提供的调色板
(4)步骤(2)和(3)的存储一定要用调色板
"""
# Start of the process
Drishti_images = sorted([f for f in os.listdir(Drishti_img_path) if f.endswith(".png")])
RIM_images = sorted([f for f in os.listdir(RIM_img_path) if f.endswith(".jpg")])

if not Drishti_images:
    print("No images found in Drishti_img_path.")
    exit()
if not RIM_images:
    print("No images found in RIM_img_path.")
    exit()

for Dr_image_name in Drishti_images:
    # Step 1
    RIM_image_name = random.choice(RIM_images)
    Dr_image = Image.open(os.path.join(Drishti_img_path, Dr_image_name))
    RIM_image = Image.open(os.path.join(RIM_img_path, RIM_image_name)).resize((Dr_image.width, Dr_image.height))

    combined_image = Image.new("RGB", (2*Dr_image.width, Dr_image.height))
    combined_image.paste(Dr_image, (0, 0))
    combined_image.paste(RIM_image, (Dr_image.width, 0))
    combined_image.save(os.path.join(Dr_combine_img_path, Dr_image_name.split('.')[0] + "_" + RIM_image_name.replace(".jpg", ".png")))

    # Step 2
    Dr_gt = Image.open(os.path.join(Drishti_gt_path, Dr_image_name))
    RIM_gt = Image.open(os.path.join(RIM_gt_path, RIM_image_name.replace(".jpg", ".png"))).resize((Dr_image.width, Dr_image.height))

    combined_gt = Image.new("P", (2*Dr_image.width, Dr_image.height))
    combined_gt.paste(Dr_gt, (0, 0))
    combined_gt.paste(RIM_gt, (Dr_image.width, 0))

    combined_gt = Image.fromarray(np.asarray(combined_gt).astype('uint8'))
    combined_gt.putpalette(color_map().tobytes())
    combined_gt.save(os.path.join(Dr_combine_gt_path, Dr_image_name.split('.')[0] + "_" + RIM_image_name.replace(".jpg", ".png")))

    # Step 3
    mask3 = Image.new("P", (RIM_gt.width, RIM_gt.height), 3)
    combined_gt_points = Image.new("P", (2*Dr_gt.width, Dr_gt.height))
    combined_gt_points.paste(Dr_gt, (0, 0))
    combined_gt_points.paste(mask3, (Dr_image.width, 0))

    combined_gt_points = Image.fromarray(np.asarray(combined_gt_points).astype('uint8'))
    combined_gt_points.putpalette(color_map().tobytes())
    combined_gt_points.save(os.path.join(Dr_combine_gt_points_path, Dr_image_name.split('.')[0] + "_" + RIM_image_name.replace(".jpg", ".png")))

print("Processing Completed!")





