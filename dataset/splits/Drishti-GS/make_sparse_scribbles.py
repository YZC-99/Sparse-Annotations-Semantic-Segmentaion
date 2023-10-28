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

def bfs_select_adjacent(img_arr, class_val, n_pixels=50):
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

def bfs_select_line(img_arr, class_val, n_pixels=100, width=2):
    # Find initial pixel of the given class
    y_indices, x_indices = np.where(img_arr == class_val)
    if len(y_indices) == 0:
        return []

    start_idx = random.choice(range(len(y_indices)))
    start_pixel = (y_indices[start_idx], x_indices[start_idx])

    visited = set([start_pixel])
    current_line = deque([start_pixel])
    potential_next = deque()

    selected_pixels = [start_pixel]

    while current_line and len(selected_pixels) < n_pixels:
        current_pixel = current_line.popleft()
        y, x = current_pixel

        adjacent_pixels = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if 0 <= y+dy < img_arr.shape[0] and 0 <= x+dx < img_arr.shape[1] and (y+dy, x+dx) not in visited:
                    adjacent_pixels.append((y+dy, x+dx))

        # Filter pixels based on the class_val
        adjacent_pixels = [p for p in adjacent_pixels if img_arr[p] == class_val]

        if len(adjacent_pixels) <= width:
            current_line.extend(adjacent_pixels)
            selected_pixels.extend(adjacent_pixels)
            visited.update(adjacent_pixels)
        else:
            current_line.append(random.choice(adjacent_pixels))
            potential_next.extend([p for p in adjacent_pixels if p not in current_line])

        # If no pixels left in current_line, choose from potential_next
        if not current_line:
            while potential_next and len(current_line) < width:
                next_pixel = potential_next.popleft()
                if next_pixel not in visited:
                    current_line.append(next_pixel)
                    visited.add(next_pixel)

    return selected_pixels


def process_mask(file_path, target_path, n_pixels=50):
    img = Image.open(file_path)
    img_arr = np.array(img)

    for class_val in [0, 1, 2]:
        selected_pixels = bfs_select_line(img_arr, class_val, n_pixels)

        mask = np.ones_like(img_arr, dtype=bool)
        for y, x in selected_pixels:
            mask[y, x] = False

        img_arr[(img_arr == class_val) & mask] = 3

    new_img = Image.fromarray(img_arr.astype('uint8'))
    new_img.putpalette(color_map().ravel())
    new_img.save(target_path)

root = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后/my_gts_cropped'
target_root = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后/my_gts_cropped_50scribbles'

if not os.path.exists(target_root):
    os.makedirs(target_root)

for file_name in os.listdir(root):
    process_mask(os.path.join(root, file_name), os.path.join(target_root, file_name))
