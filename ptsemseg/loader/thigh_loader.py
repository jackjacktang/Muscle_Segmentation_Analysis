# DEBUG=False
# def log(s):
#     if DEBUG:
#         print(s)

# from torch.utils import data
# from torchvision import transforms
# import numpy as np
# import torch
# import collections
# import os
# from os.path import join as pjoin
# from PIL import Image, ImageOps
# from ptsemseg.utils import *
# from scipy.ndimage.interpolation import zoom
# from random import randint
# import cv2
# import random

# class thighLoader(data.Dataset):
#     def __init__(
#         self,
#         root,
#         split,
#         augmentations=None,
#         n_classes=1
#     ):
#         self.root = os.path.expanduser(root)
#         self.n_classes = n_classes # for regression
#         self.augmentations = augmentations
#         self.split = split
#         self.pathList = self.getInfoLists()
#         log('Init Data Loader: self.split is {}, length of self.pathList is {}'.format(self.split, len(self.pathList)))

#     def __len__(self):
#         return len(self.pathList)

#     def __getitem__(self, index):
#         if self.split == 'train':
#             img_path = self.root + '/train/img/' + self.pathList[index]
#         elif self.split == 'val':
#             img_path = self.root + '/val/img/' + self.pathList[index]
#         elif self.split == 'test':
#             img_path = self.root + '/test/img/' + self.pathList[index]

#         lbl_path = img_path.replace('img', 'lbl').rstrip('.png') + '_lbl.png'


#         log('Loader: {}: img: {} label: {}'.format(index, img_path, lbl_path))
#         # img = cv2.imread(img_path)
#         img = np.asarray(Image.open(img_path).convert('RGB'))
#         # print('load size: ', img.shape)
#         # print('path: ', img_path)
#         # lbl = cv2.imread(lbl_path, 0)
#         lbl = np.asarray(Image.open(lbl_path))
#         img = img.astype(np.float64)
#         img /= 255.0
#         lbl = np.array(lbl, dtype=np.uint8) // 255

#         # img = np.expand_dims(img, axis=2)
#         # lbl = np.expand_dims(lbl, axis=1)

#         # print('np.unique(img)', np.unique(img))
#         # print('np.unique(lbl)', np.unique(lbl))

#         # print('shape: ', img.shape, lbl.shape)

#         # print(self.augmentations)
#         if self.augmentations is not None:
#             img, lbl = self.augmentations(img, lbl)

#         # print('aug shape: ', img.shape, lbl.shape)

#         img = img.transpose(2, 0, 1)

#         img, lbl = self.transform(img, lbl)
#         # print('img/lbl shape: ', img.shape, lbl.shape)

#         # if self.split == 'test':
#         #     return img, lbl, self.pathList[index]
#         return img, lbl


#     def getInfoLists(self):
#         if self.split == 'train':
#             lst = os.listdir(self.root + '/train/img/')
#             print(len(lst))
#             random.shuffle(lst)
#             print(len(lst))
#             return lst
#         if self.split == 'val':
#             return os.listdir(self.root + '/val/img/')
#         if self.split == 'test':
#             return os.listdir(self.root + '/test/img/')


#     # transform from numpy to tensor
#     def transform(self, img, lbl):

#         # img = np.stack([img], axis=0)
#         # if self.n_classes == 1:
#         #     lbl = np.stack([lbl], axis=0)
#         # else:
#         #     lbl = (lbl > 0).astype('int')

#         img = torch.from_numpy(img).float()
#         # img_tensor = torch.unsqueeze(img, 0).type(torch.FloatTensor)
#         # lbl = torch.from_numpy(lbl).float()
#         # lbl_tensor = torch.unsqueeze(lbl, 0).type(torch.FloatTensor)

#         # if self.n_classes == 1:
#         #     lbl = torch.from_numpy(lbl).float()
#         # else:
#         #     lbl = torch.from_numpy(lbl).long()

#         lbl = torch.from_numpy(lbl).float()

#         return img, lbl

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

class thighLoader(data.Dataset):
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
        self.augmentations = augmentations

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, index):
        if self.split == 'train':
            img_path = self.root + '/train/img/' + self.pathList[index]
        elif self.split == 'test':
            img_path = self.root + '/test/img/' + self.pathList[index]
        
        tmp_idx = img_path.rindex('_')
        lbl_path = img_path.replace('img', 'lbl')[:tmp_idx] + "_mask" + img_path[tmp_idx:]

        img = torch.load(img_path)

        lbl = torch.load(lbl_path)

        if self.augmentations is not None:
            img, lbl = img.repeat(3, 1, 1), lbl.repeat(3, 1, 1)
            img, lbl = self.augmentations(img, lbl)
            # print('before shape: ', img.shape, lbl.shape)
            # assert torch.all(img[0, :, :].eq(img[1, :, :]))
            img, lbl = img[0, :, :], lbl[0, :, :]
            img = torch.unsqueeze(img, 0)
            lbl = torch.unsqueeze(lbl, 0)
            # print('after shape: ', img.shape, lbl.shape)

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

    def transform(self, img, lbl):
        # anchor_imgs, pos_imgs, neg_imgs = random_crop_triplet(images, labels)
        return img, lbl

    def check_annotated(self, lst):
        new_lst = []
        for img_name in lst:
            img_path = self.root + '/train/img/' + img_name
            tmp_idx = img_path.rindex('_')
            lbl_path = img_path.replace('img', 'lbl')[:tmp_idx] + "_mask" + img_path[tmp_idx:]
            if os.path.isfile(lbl_path):
                new_lst.append(img_name)
        return new_lst