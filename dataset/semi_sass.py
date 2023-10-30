import numpy as np
import matplotlib.pyplot as plt
from dataset.transform import *
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
from util.utils import *
import torch
from torch.utils.data import DataLoader
from model.tools import *


class Semi_DrishtiDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 3

        self.img_path = root
        self.true_mask_path = root

        if mode == 'val':
            self.label_path = self.true_mask_path
            self.label_type = 'my_gts_cropped'
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
                self.ids = [self.labeled_ids]
        else:

            labeled_id_path = 'dataset/splits/%s/labeled.txt' % name
            unlabeled_id_path = 'dataset/splits/%s/unlabeled.txt' % name
            # labeled_id_path = 'D:/Dev_projects/AGMM-SASS/dataset/splits/%s/labeled.txt' % name
            # unlabeled_id_path = 'D:/Dev_projects/AGMM-SASS/dataset/splits/%s/unlabeled.txt' % name
            with open(labeled_id_path, 'r') as f:
                self.labeled_ids = f.read().splitlines()
            with open(unlabeled_id_path, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            # 如果labeled_ids数量不及unlabeled_ids，则通过复制的形式将labeled_ids数量补齐到和unlabeled_ids一致
            if len(self.labeled_ids) < len(self.unlabeled_ids):
                multiplier = len(self.unlabeled_ids) // len(self.labeled_ids) + 1
                self.labeled_ids = self.labeled_ids * multiplier
                self.labeled_ids = self.labeled_ids[:len(self.unlabeled_ids)]

            if len(self.unlabeled_ids) < len(self.labeled_ids):
                multiplier = len(self.labeled_ids) // len(self.unlabeled_ids) + 1
                self.unlabeled_ids = self.unlabeled_ids * multiplier
                self.unlabeled_ids = self.unlabeled_ids[:len(self.labeled_ids)]
            self.ids = [self.labeled_ids,self.unlabeled_ids]

    def preprocess(self,img,mask):
        # basic augmentation on all training images
        img, mask = resize([img, mask], (0.5, 2.0))
        img, mask = crop([img, mask], self.size)
        img, mask = hflip(img, mask, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        return img,mask

    def __getitem__(self, item):

        if self.mode == 'val':
            labeled_id = self.labeled_ids[item]
            labeled_img = Image.open(os.path.join(self.root, labeled_id.split(' ')[0]))
            labeled_mask = Image.open(os.path.join(self.root, labeled_id.split(' ')[1]))
            img, mask = val_resize([labeled_img, labeled_mask], 256,256)
            img, mask = normalize(img, mask)
            return img,mask,labeled_id,"","",""


        labeled_id = self.labeled_ids[item]
        unlabeled_id = self.unlabeled_ids[item]

        labeled_img = Image.open(os.path.join(self.root, labeled_id.split(' ')[0]))
        labeled_mask = Image.open(os.path.join(self.root, labeled_id.split(' ')[1]))

        unlabeled_img = Image.open(os.path.join(self.root, unlabeled_id.split(' ')[0]))
        unlabeled_mask = Image.open(os.path.join(self.root, unlabeled_id.split(' ')[1]))

        labeled_img,labeled_mask = self.preprocess(labeled_img,labeled_mask)
        unlabeled_img,unlabeled_mask = self.preprocess(unlabeled_img,unlabeled_mask)

        return labeled_img,labeled_mask,labeled_id,unlabeled_img,unlabeled_mask,unlabeled_id

    def __len__(self):
        if len(self.ids) > 1:
            if len(self.ids[0]) > len(self.ids[0]):
                return len(self.ids[0])
            else:
                return len(self.ids[1])
        else:
            return len(self.ids[0])

if __name__ == '__main__':
    name = 'Drishti-GS/semi-supervised'
    root = 'D:/1-Study/220803研究生阶段学习/221216论文写作专区/OD_OC/数据集/Drishti-GS-裁剪后'
    mode = 'semi'
    size = 256
    dataset = Semi_DrishtiDataset(name,root,mode,size)
    from torch.utils.data import DataLoader
    data_loader = DataLoader(dataset,batch_size=2)
    for i,(labeled_img,labeled_mask,labeled_id,unlabeled_img,unlabeled_mask,unlabeled_id) in enumerate(data_loader):
        print(labeled_img.shape)
        print("=========")
        print(unlabeled_img.shape)

