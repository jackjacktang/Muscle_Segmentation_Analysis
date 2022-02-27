from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import csv
import argparse
import nibabel as nib
from ptsemseg.functions import create_folder

def get_ssim_nifti(nii_1, nii_2):
    roi_img_1 = nib.load(nii_1)
    roi_dat_1 = roi_img_1.get_fdata()
    # print('roi_dat_1.shape', roi_dat_1.shape)

    roi_img_2 = nib.load(nii_2)
    roi_dat_2 = roi_img_2.get_fdata()
    # print('roi_dat_2.shape', roi_dat_2.shape)

    return ssim(roi_dat_1, roi_dat_2)

def get_volume(nii_1):
    roi_img_1 = nib.load(nii_1)
    roi_dat_1 = roi_img_1.get_fdata()

    # print(np.unique(roi_dat_1))

    sum_1 = np.sum(roi_dat_1)
    
    return sum_1


def scores_to_csv(score_dict, subject_name, save_path, args):
    fields = ['Subject & Case'] + list(score_dict.keys())
    dw = { subject_name: score_dict}

    def mergedict(a,b):
        a.update(b)
        return a

    path = f"{args.save_path}/{args.mode}_stats.csv"
    create_folder(args.save_path)
    file_exists = os.path.isfile(path) 
    
    f = open(path, "a")
    w = csv.DictWriter(f, fields)
    if not file_exists:
        w.writeheader()
    for k,d in sorted(dw.items()):
        w.writerow(mergedict({'Subject & Case': k},d))
    f.close()
    print(f'CSV file {path} has been edited.')

def main(args):
    mode = args.mode
    img_1 = args.nifti_1
    img_lst = args.niftis
    
    if mode == 'volume':
        # volume changes comparing two masks
        vol_1 = get_volume(img_1)
        print(f'Volume of {img_1} is {vol_1}.')
    elif mode == 'ssim':
        # SSIM between two images
        score_dict = {}
        for img_2 in img_lst:
            data = get_ssim_nifti(img_1, img_2)
            case = os.path.split(img_2)[1].rstrip('.nii.gz')
            print(f'Subject {args.subject_name} Case {case} SSIM: {data}')
            score_dict['SSIM'] = data
        scores_to_csv(score_dict, args.subject_name, args.save_path, args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/pipeline_stats',
        help='path to save stats as csv'
    )

    parser.add_argument(
        '--mode',
        nargs='?',
        type=str,
        default='ssim',
        help='path to save stats as csv'
    )

    parser.add_argument(
        '--nifti_1',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/MSTHIGH_07/BL_left.nii.gz',
        help='path to FU (after registration with ground truth) nifti / path to thigh mask nifiti to test volume'
    )

    parser.add_argument(
        '--subject_name',
        nargs='?',
        type=str,
        default='MSTHIGH_07_left',
        help='subject name'
    )

    parser.add_argument(
        '--niftis',
        nargs='+',
        type=str,
        default='/raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/FU_left_ref2bl.nii.gz',
        help='path to FU (after registration with predicted femur) nifti files to calculate ssim'
    )

    args = parser.parse_args()

    main(args)