from torch.utils import data
from torchvision import transforms
import numpy as np
import torch
import collections
import os
from os.path import join as pjoin
from PIL import Image
from ptsemseg.utils import *
from ptsemseg.processing.preprocessing import norm_slice
from scipy.ndimage.interpolation import zoom
from random import randint
import cv2
import random

import warnings
# warnings.filterwarnings("error")
# np.seterr(all='raise')

class ThighAmerLoader(data.Dataset):
    def __init__(
        self,
        root,
        root_triplet,
        split,
        augmentations=None,
        n_classes=2,
        patch_size=4,
        triplet_mode=True
    ):
        self.root = os.path.expanduser(root)
        self.root_triplet = os.path.expanduser(root_triplet)
        self.n_classes = n_classes
        self.split = split
        self.pathList = self.getInfoLists()
        # print(self.pathLists)
        self.augmentations = augmentations
        self.patch_size = patch_size
        self.triplet_mode = triplet_mode

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, index):
        if self.split == 'train':
            img_path = self.root + '/train/img/' + self.pathList[index]
            img_path_triplet = self.root + '_triplet/train/img/' + self.pathList[index]
        elif self.split == 'test':
            img_path = self.root + '/test/img/' + self.pathList[index]
            img_path_triplet = self.root + '_triplet/test/img/' + self.pathList[index]
        
        tmp_idx = img_path.rindex('_')
        lbl_path = img_path.replace('img', 'lbl')[:tmp_idx] + "_mask" + img_path[tmp_idx:]
        tmp_idx = img_path_triplet.rindex('_')
        lbl_path_triplet = img_path_triplet.replace('img', 'lbl')[:tmp_idx] + "_mask" + img_path_triplet[tmp_idx:]

        img = torch.load(img_path)
        lbl = torch.load(lbl_path)
        # print(self.triplet_mode)
        if not self.triplet_mode:
            torch_zeros = torch.zeros(1, self.patch_size, self.patch_size)
            return img, lbl, torch_zeros, torch_zeros, torch_zeros, torch_zeros, torch_zeros, torch_zeros

        img_triplet = torch.load(img_path_triplet).numpy()[0]
        lbl_triplet = torch.load(lbl_path_triplet).numpy()[0]

        triplet = self.find_triplet(img_triplet, lbl_triplet)

        if len(triplet) != 3:
            print(f"Skipped {img_path_triplet} due to no triplets found.")
            torch_zeros = torch.zeros(1, self.patch_size, self.patch_size)
            return img, lbl, torch_zeros, torch_zeros, torch_zeros, torch_zeros, torch_zeros, torch_zeros
            
        (A_img, A_lbl), (P_img, P_lbl), (N_img, N_lbl)  = triplet

        A_img = np.expand_dims(norm_slice(A_img, mode=(0,1)), 0).astype(np.float64)
        A_img = torch.from_numpy(A_img).float()
        P_img = np.expand_dims(norm_slice(P_img, mode=(0,1)), 0).astype(np.float64)
        P_img = torch.from_numpy(P_img).float()
        # print(N_img)
        N_img = np.expand_dims(norm_slice(N_img, mode=(0,1)), 0).astype(np.float64)
        N_img = torch.from_numpy(N_img).float()

        A_lbl = torch.from_numpy(np.expand_dims(A_lbl, 0).astype(np.float64)).float()
        P_lbl = torch.from_numpy(np.expand_dims(P_lbl, 0).astype(np.float64)).float()
        N_lbl = torch.from_numpy(np.expand_dims(N_lbl, 0).astype(np.float64)).float()
        
        # if self.augmentations is not None:
        #     img, lbl = img.repeat(3, 1, 1), lbl.repeat(3, 1, 1)
        #     img, lbl = self.augmentations(img, lbl)
        #     # print('before shape: ', img.shape, lbl.shape)
        #     # assert torch.all(img[0, :, :].eq(img[1, :, :]))
        #     img, lbl = img[0, :, :], lbl[0, :, :]
        #     img = torch.unsqueeze(img, 0)
        #     lbl = torch.unsqueeze(lbl, 0)
        #     # print('after shape: ', img.shape, lbl.shape)

        # print('aug shape: ', img.shape, lbl.shape)

        # img = img.transpose(2, 0, 1)

        # print('load img/lbl shape: ', img.shape, lbl.shape, A_img.shape, A_lbl.shape)
        # img, lbl = self.transform(img, lbl)

        return img, lbl, A_img, P_img, N_img, A_lbl, P_lbl, N_lbl

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

    def find_triplet(self, img, lbl):
        # print(np.unique(lbl), np.unique(img))
        TARGET_SIZE = self.patch_size
        APs, Ns = [], []
        # APs_coors = []
        i = 0
        while i < len(lbl)-TARGET_SIZE:
            j = 0
            while j < len(lbl[i])-TARGET_SIZE:
                patch_img = img[i:i+TARGET_SIZE, j:j+TARGET_SIZE]
                patch_lbl = lbl[i:i+TARGET_SIZE, j:j+TARGET_SIZE]
                if np.all(patch_img == 0) or np.isnan(patch_img[patch_img > 0].std()):
                    j+=1
                    continue
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        (patch_img-patch_img[patch_img > 0].mean())/patch_img[patch_img > 0].std()
                    except Warning:
                        j+=1
                        continue

                if np.count_nonzero(patch_lbl) > (TARGET_SIZE**2)*0.8:
                    APs.append((patch_img, patch_lbl))
                    # APs_coors.append((i, j))
                    j+=TARGET_SIZE
                elif 1 not in patch_lbl:
                    if np.count_nonzero(patch_img) > (TARGET_SIZE**2)/2:
                        Ns.append((patch_img, patch_lbl))
                        j+=TARGET_SIZE
                    else:
                        j+=1 
                else:
                    j+=1
            i+=1
        # print(APs, Ns)
        triplet = []
        A_index = random.randint(0, len(APs)-1)
        # A_range = APs_coors[A_index]
        A = APs[A_index]
        triplet.append(A)
        APs.pop(A_index)
        # APs_coors.pop(A_index)

        # while True:
        #     if (len(APs) == 0):
        #         break
        #     P_index = random.randint(0, len(APs)-1)
        #     i, j = APs_coors[P_index]
        #     cover_range = (range(i, i+TARGET_SIZE), range(j, j+TARGET_SIZE))
        #     if (A_range[0] in cover_range[0] and A_range[1] in cover_range[1]) \
        #         or (A_range[0] in cover_range[0] and A_range[1]+TARGET_SIZE in cover_range[1]) \
        #         or (A_range[0]+TARGET_SIZE in cover_range[0] and A_range[1] in cover_range[1]) \
        #         or (A_range[0]+TARGET_SIZE in cover_range[0] and A_range[1]+TARGET_SIZE in cover_range[1]): 
        #         # overlap
        #         APs.pop(P_index)
        #         APs_coors.pop(P_index)
        #         print("APs", len(APs))
        #         continue
        #     triplet.append(APs[P_index])
        #     break

        P_index = random.randint(0, len(APs)-1)
        # P_range = APs_coors[P_index]
        P = APs[P_index]
        triplet.append(P)

        if len(Ns) >= 1:
            triplet.append(random.choice(Ns))
        # print(len(triplet))
        return triplet