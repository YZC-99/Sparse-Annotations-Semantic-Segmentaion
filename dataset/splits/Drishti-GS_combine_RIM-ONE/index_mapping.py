import os

# 指定图像和对应掩码的文件夹路径
img_folder = r'D:/Dev_projects/AGMM-SASS/data/Drishti-GS_Combine_RIM-ONE/imgs'
gt_folder = r'D:/Dev_projects/AGMM-SASS/data/Drishti-GS_Combine_RIM-ONE/gts'

# 获取图像文件列表和对应掩码文件列表
img_files = os.listdir(img_folder)
gt_files = os.listdir(gt_folder)

# 创建并打开txt文件以写入文件名
with open('file_mapping.txt', 'w') as file_mapping:
    # 遍历图像文件列表
    for img_file in img_files:
        # 确保图像文件和对应掩码文件存在
        img_path = os.path.join(img_folder, img_file)
        gt_file = img_file.replace('.PNG', '_mask.PNG')
        gt_path = os.path.join(gt_folder, gt_file)

        imgs_dir = 'imgs/'
        gts_dir = 'gts/'
        if os.path.exists(gt_path):
            # 写入图像文件名和对应掩码文件名到txt文件中
            file_mapping.write(f"{imgs_dir}{img_file} {gts_dir}{gt_file}\n")
        else:
            print(f"对应的掩码文件不存在：{gt_file}")

print("文件映射已生成并保存到file_mapping.txt")
