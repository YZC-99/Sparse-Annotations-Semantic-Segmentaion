import os
from PIL import Image
import numpy as np
import random
from collections import deque

def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')
    cmap[0] = np.array([0, 0, 0])
    cmap[1] = np.array([255, 0, 0])
    cmap[2] = np.array([0, 255, 0])
    cmap[3] = np.array([255, 255, 255])
    return cmap

def bfs_select_adjacent(img_arr, class_val, n_pixels=100):
    # Find initial pixel of the given class
    y_indices, x_indices = np.where(img_arr == class_val)
    if len(y_indices) == 0:
        return []

    start_idx = random.choice(range(len(y_indices)))
    start_pixel = (y_indices[start_idx], x_indices[start_idx])

    visited = set()
    to_explore = deque([start_pixel])
    selected_pixels = []

    while to_explore and len(selected_pixels) < n_pixels:
        current_pixel = to_explore.popleft()

        y, x = current_pixel
        if (y, x) not in visited and img_arr[y, x] == class_val:
            selected_pixels.append((y, x))
            visited.add((y, x))

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if 0 <= y+dy < img_arr.shape[0] and 0 <= x+dx < img_arr.shape[1]:
                        to_explore.append((y+dy, x+dx))

    return selected_pixels

def process_mask(file_path, target_path, n_pixels=50):
    img = Image.open(file_path)
    img_arr = np.array(img)

    for class_val in [0, 1, 2]:
        selected_pixels = bfs_select_adjacent(img_arr, class_val, n_pixels)

        mask = np.ones_like(img_arr, dtype=bool)
        for y, x in selected_pixels:
            mask[y, x] = False

        img_arr[(img_arr == class_val) & mask] = 3

    new_img = Image.fromarray(img_arr.astype('uint8'))
    new_img.putpalette(color_map().ravel())
    new_img.save(target_path)

root = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后/my_gts_cropped'
target_root = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后/my_gts_cropped_100points'

if not os.path.exists(target_root):
    os.makedirs(target_root)

for file_name in os.listdir(root):
    process_mask(os.path.join(root, file_name), os.path.join(target_root, file_name))
