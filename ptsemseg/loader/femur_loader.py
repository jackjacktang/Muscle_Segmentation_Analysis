from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
import collections
import os
from os.path import join as pjoin
from PIL import Image
from ptsemseg.utils import *
from scipy.ndimage.interpolation import zoom
from random import randint
import cv2
import random

class FemurLoader(data.Dataset):
    def __init__(
        self,
        root,
        split,
        augmentations=None,
        n_classes=2
    ):
        self.root = os.path.expanduser(root)
        self.n_classes = n_classes
        self.split = split
        self.pathList = self.getInfoLists()

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, index):
        if self.split == 'train':
            img_path = self.root + '/train/img/' + self.pathList[index]
        elif self.split == 'test':
            img_path = self.root + '/test/img/' + self.pathList[index]
        
        tmp_idx = img_path.rindex('_')
        lbl_path = img_path.replace('img', 'lbl')[:tmp_idx] + "_femur" + img_path[tmp_idx:]

        img = torch.load(img_path)

        lbl = torch.load(lbl_path)
        # img = img.astype(np.float64)
        # img /= 255.0
        # lbl = np.array(lbl, dtype=np.uint8) // 255

        # if self.augmentations is not None:
        #     img, lbl = self.augmentations(img, lbl)

        # print('aug shape: ', img.shape, lbl.shape)

        # img = img.transpose(2, 0, 1)

         # print('load img/lbl shape: ', img.shape, lbl.shape)
        img, lbl = self.transform(img, lbl)

        return img, lbl


    def getInfoLists(self):
        if self.split == 'train':
            lst = os.listdir(self.root + '/train/img/')
            lst = self.check_annotated(lst)
            random.shuffle(lst)
            print("Number of inputs:", len(lst))
            return lst
        if self.split == 'test':
            return os.listdir(self.root + '/test/img/')


    # transform from numpy to tensor
    def transform(self, img, lbl):
        # img = torch.from_numpy(img).float()
        # lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def check_annotated(self, lst):
        new_lst = []
        for img_name in lst:
            img_path = self.root + '/train/img/' + img_name
            tmp_idx = img_path.rindex('_')
            lbl_path = img_path.replace('img', 'lbl')[:tmp_idx] + "_femur" + img_path[tmp_idx:]
            if os.path.isfile(lbl_path):
                new_lst.append(img_name)
        return new_lst