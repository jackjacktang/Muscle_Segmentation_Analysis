import os
import numpy as np
import nibabel as nib
import glob
import torch
from ptsemseg.functions import create_folder


'''
Split dataset into train and validation randomly by ratio
'''
def init_data_split(root, split_ratio, compound_dataset = False):
    from random import shuffle
    from os.path import join as pjoin
    from ptsemseg.utils import dataset_meta
    ratio = split_ratio

    meta_path = pjoin(root, 'meta_images.txt')
    dataset_meta(root)
    dataset_meta(root, target='test')
    img_paths = []
    with open(meta_path) as f:
        lines = f.read().splitlines()
        for item in lines:
            if item.__contains__('.tif'):
                #print(item[:-2])
                #item = item.replace('\n','')
                img_paths.append(item)
    shuffle(img_paths)
    val_paths = img_paths[:int(ratio*(len(img_paths)))]
    # log('length of val_paths is: {}'.format(len(val_paths)))
    # log('length of img_paths is: {}'.format(len(img_paths)))
    shuffle(val_paths)
    return {'img_paths': img_paths, 'val_paths': val_paths}


def prep_class_val_weights(ratio):
    weight_foreback = torch.ones(2)
    weight_foreback[0] = 1 / (1 - ratio) # ?????????
    weight_foreback[1] = 1 / ratio
    if torch.cuda.is_available():
        weight_foreback = weight_foreback.cuda()
    # display("Cross Entropy's Weight:{}".format(weight_foreback))
    return weight_foreback

'''
Split femur dataset to train and test dataset by ratio
'''
def femur_data_split(root_path, subject_names, ratio="8:2"):
    # Assume train/test splits only
    train_ratio, test_ratio = map(lambda x: int(x)/10, ratio.split(':'))
    slice_counter = {subject_name:0 for subject_name in subject_names}

    train_save_path_img = root_path+"/train/img/"
    train_save_path_lbl = root_path+"/train/lbl/"
    test_save_path_img = root_path+"/test/img/"
    test_save_path_lbl = root_path+"/test/lbl/"

    create_folder(train_save_path_img)
    create_folder(train_save_path_lbl)
    create_folder(test_save_path_img)
    create_folder(test_save_path_lbl)

    skip_files = [('MSTHIGH_06', 'BL_left.nii.gz'), 
                ('MSTHIGH_06', 'BL_left_femur.nii.gz'),
                ('MSTHIGH_06', 'BL_right.nii.gz'),
                ('MSTHIGH_06', 'BL_right_femur.nii.gz'),
                ('MSTHIGH_13', 'BL_right_bone.nii.gz'),
                ('MSTHIGH_13', 'BL_right_femur.nii.gz'),
                ('MSTHIGH_13', 'BL_right.nii.gz'),
                ('MSTHIGH_13', 'FU2_right_femur.nii.gz'),
                ('MSTHIGH_13', 'FU2_right.nii.gz'),
                ('MSTHIGH_13', 'FU2_left_femur.nii.gz'),
                ('MSTHIGH_13', 'FU2_left.nii.gz'),
                ('MSTHIGH_13', 'FU_right_femur.nii.gz'),
                ('MSTHIGH_13', 'FU_right.nii.gz'),
                ('MSTHIGH_13', 'FU_left_femur.nii.gz'),
                ('MSTHIGH_13', 'FU_left.nii.gz'),
                ('MSTHIGH_13', 'BL_left_femur.nii.gz'),
                ('MSTHIGH_13', 'BL_left.nii.gz'),
                ('MSTHIGH_14', 'BL_right_femur.nii.gz'),
                ('MSTHIGH_14', 'BL_right.nii.gz'),
                ('MSTHIGH_14', 'BL_left_femur.nii.gz'),
                ('MSTHIGH_14', 'BL_left.nii.gz'),
                ]
    
    for subject_name in subject_names:
        nitfi_file_names = os.listdir(f'{root_path}/{subject_name}/')
        # print("nitfi_file_names", nitfi_file_names)
        for file_name in nitfi_file_names:
            if (subject_name, file_name) in skip_files:
                continue
            if ".nii.gz" not in file_name:
                continue
            if "femur" not in file_name:
                slice_counter[subject_name] += load_nifti_to_dat(f'{root_path}/{subject_name}/{file_name}', train_save_path_img, subject_name)
            else:
                load_nifti_to_dat(f'{root_path}/{subject_name}/{file_name}', train_save_path_lbl, subject_name, is_lbl=True)

    total_slices = sum(slice_counter.values())
    test_subjects = find_subjects_to_test(slice_counter, total_slices, test_ratio)
    print(f"Test subjects are: {test_subjects}. Please use them during testing.") # by default, they're 06 and 07

    # Move all relevant subject slices to another folder
    files = []
    for s in test_subjects:
        files.extend(glob.glob(f'{train_save_path_img}/{s}*'))
    print(f"{len(files)} number of files to remove from train dataset.")
    for f in files:
        img_path_remove = f
        os.remove(f) # remove files in train dataset
        print("Removing", img_path_remove)
        tmp_idx = img_path_remove.rindex('_')
        lbl_path_remove = img_path_remove.replace('img', 'lbl')[:tmp_idx] + "_femur" + img_path_remove[tmp_idx:]
        if os.path.isfile(lbl_path_remove):
            os.remove(lbl_path_remove)
        print("Removing", lbl_path_remove)

    print("Slice Counter:", slice_counter)


# Find which subjects should move to test dataset
def find_subjects_to_test(slice_counter, total_slices, test_ratio):
    approx_slices_to_test = int(total_slices*test_ratio)
    sorted_counter = sorted(slice_counter.items(), key=lambda x: x[1], reverse=True)
    result = []
    for (subject, count) in sorted_counter:
        if approx_slices_to_test <= 0:
            break
        approx_slices_to_test -= count
        result.append(subject)
    return result
    
def load_nifti_to_dat(file_path, save_path, subject_name, is_test=False, is_lbl=False):
    head, tail = os.path.split(file_path)
    
    roi_img = nib.load(file_path)
    roi_dat = roi_img.get_fdata()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    result_roi = np.zeros(roi_dat.shape)
    slice_no = roi_dat.shape[2]
    count = 0
    for s in range(slice_no):
        roi_slice = roi_dat[:, :, s]
        if len(np.unique(roi_slice)) <= 1:
            continue
        # print("before_padded_slice shape", roi_slice.shape)
        padded_slice, pad_info = pad_slice(roi_slice)

        if not is_lbl:
            # print('before', np.unique(padded_slice))
            padded_slice = norm_slice(padded_slice) # normalization
            # print('after', np.unique(padded_slice))

        padded_slice = padded_slice.astype(np.float64)
        padded_slice = np.expand_dims(padded_slice, 0)

        save_file_path= f'{save_path}{subject_name}_{tail.rstrip(".nii.gz")}_{s:03d}.dat'
        if np.isnan(padded_slice).any():
            print(f"NaN in {save_file_path}")
            raise RuntimeError()

        padded_slice = torch.from_numpy(padded_slice).float()

        torch.save(padded_slice, save_file_path)
        count+=1

        if s == 0 or s == slice_no - 1 or s % 100 == 0:
            print(f"Saved {save_file_path}")
    return count


def norm_slice(img, mode=(-1,1)):
    img = img.astype(np.float)
    # print("unnormalized unique:", np.unique(img))

    # Normalised [0,1]
    if mode == (0, 1):
        img = (img - np.min(img[img > 0]))/(np.max(img[img > 0])-np.min(img[img > 0]))
        img = img.clip(min=0)
    # Normalised [-1,1]
    elif mode == (-1, 1):
        img = (img-img[img > 0].mean())/img[img > 0].std()
    else:
        raise ValueError('Normalisation undefined mode')
    
    # print("normalized unique:", np.unique(img))
    return img


def pad_bg_x(img, size=256):
    x_left = 0
    x_right = 0
    x, y = img.shape
    if x < size:
        padded_x = size - x
        x_left = int(padded_x / 2)
        x_right = padded_x - x_left
        return [x_left, x_right]

    if x > size:
        cut_x = x - size
        x_left = int(cut_x / 2)
        x_right = cut_x - x_left
        return [-x_left, -x_right]

    return [x_left, x_right]


def pad_bg_y(img, size=256):
    y_up = 0
    y_down = 0
    x, y = img.shape

    if y < size:
        padded_y = size - y
        y_up = int(padded_y / 2)
        y_down = padded_y - y_up
        return [y_up, y_down]

    if y > size:
        cut_y = y - size
        y_up = int(cut_y / 2)
        y_down = cut_y - y_up
        return [-y_up, -y_down]

    return [y_up, y_down]

def pad_slice(img_slice):
    # print('original: ', img_slice.shape)
    img_slice = np.rot90(img_slice)


    # ---- pad to 256*256 ----
    padded_img = img_slice.copy()
    # print('before pad', padded_img.shape)
    pad_left, pad_right = pad_bg_x(img_slice)
    # print('pad left/right: ', pad_left, pad_right)
    if pad_left >= 0 and pad_right >= 0:
        padded_img = np.pad(padded_img, ((pad_left, pad_right), (0, 0)), mode='constant')
    elif pad_left <= 0 and pad_right <= 0:
        cut_left = -pad_left
        cut_right = -pad_right
        x_len = padded_img.shape[0]
        padded_img = padded_img[cut_left:x_len - cut_right, ...]
        cut_img_x_left = padded_img[:cut_left, ...]
        cut_img_x_right = padded_img[x_len - cut_right:, ...]
        # padded_img = padded_img[x-cut_x:,...]
    # print('after pad x: ', padded_img.shape)

    pad_up, pad_down = pad_bg_y(img_slice)
    # print('pad up/down: ', pad_up, pad_down)
    if pad_up >= 0 and pad_down >= 0:
        padded_img = np.pad(padded_img, ((0, 0), (pad_up, pad_down)), mode='constant')
    elif pad_up <= 0 and pad_down <= 0:
        cut_up = -pad_up
        cut_down = -pad_down
        y_len = padded_img.shape[1]
        padded_img = padded_img[..., cut_up:y_len - cut_down]
        cut_img_y_up = padded_img[..., :cut_up]
        cut_img_y_down = padded_img[..., y_len - cut_down:]
    # print('after pad y: ', padded_img.shape)

    return padded_img, [pad_left, pad_right, pad_up, pad_down]

def unpad_slice(pred_slice, pad_info):

    pad_left, pad_right, pad_up, pad_down = pad_info

    unpadded_slice = pred_slice.copy()
    size_x, size_y = pred_slice.shape
    if pad_up >= 0 and pad_down >= 0:
        unpadded_slice = unpadded_slice[..., pad_up:size_y - pad_down]
    elif pad_up <= 0 and pad_down <= 0:
        cut_up = -pad_up
        cut_down = -pad_down
        unpadded_slice = np.pad(unpadded_slice, ((0, 0), (cut_up, cut_down)), mode='constant')

    # print('after unpadded y: ', unpadded_slice.shape)

    if pad_left >= 0 and pad_right >= 0:
        unpadded_slice = unpadded_slice[pad_left:size_x - pad_right, ...]
    elif pad_left <= 0 and pad_right <= 0:
        cut_left = -pad_left
        cut_right = -pad_right
        unpadded_slice = np.pad(unpadded_slice, ((cut_left, cut_right), (0, 0)), mode='constant')

    # print('after unpadded y: ', unpadded_slice.shape)
    unpadded_slice = np.rot90(unpadded_slice, 3)
    # print('original: ', unpadded_slice.shape)

    return unpadded_slice
    
def femur_img_split(img, bound_1=70, bound_2=350):
    imgs_split = [img[:, :, :bound_1], 
                img[:, : bound_1:bound_2], 
                img[:, : bound_2:]] 
    return imgs_split