import nibabel as nib
import argparse
import numpy as np
import os
from ptsemseg.functions import create_folder

def main(args):
    nii_pred = args.pred
    nii_gt = args.gt

    roi_img_pred = nib.load(nii_pred)
    roi_dat_pred = np.asarray(roi_img_pred.get_fdata())
    roi_aff = roi_img_pred.affine
    roi_hdr = roi_img_pred.header

    roi_img_gt = nib.load(nii_gt)
    roi_dat_gt = np.asarray(roi_img_gt.get_fdata())

    true_pred = np.where(roi_dat_pred == 1)
    false_pred = np.where(roi_dat_pred == 0)

    # print(roi_dat_pred.shape, roi_dat_gt.shape)
    TP = np.zeros(roi_dat_pred.shape)
    for x in range(len(true_pred[0])):
        i, j, s = true_pred[0][x], true_pred[1][x], true_pred[2][x]
        if roi_dat_gt[i, j, s] == 1:
            TP[i, j, s] = 1
    FP = np.zeros(roi_dat_pred.shape)
    for x in range(len(true_pred[0])):
        i, j, s = true_pred[0][x], true_pred[1][x], true_pred[2][x]
        if roi_dat_gt[i, j, s] == 0:
            FP[i, j, s] = 1
    FN = np.zeros(roi_dat_pred.shape)
    for x in range(len(false_pred[0])):
        i, j, s = false_pred[0][x], false_pred[1][x], false_pred[2][x]
        if roi_dat_gt[i, j, s] == 1:
            FN[i, j, s] = 1

    out_path = args.save_path
    create_folder(out_path)
    subject = args.subject_name

    nii = nib.Nifti1Image(TP, roi_aff, header=roi_hdr)
    save_as = os.path.join(out_path, subject + '_TP' + '.nii.gz')
    nib.save(nii, save_as)
    print("Saved", save_as)
    nii = nib.Nifti1Image(FP, roi_aff, header=roi_hdr)
    save_as = os.path.join(out_path, subject + '_FP' + '.nii.gz')
    nib.save(nii, save_as)
    print("Saved", save_as)
    nii = nib.Nifti1Image(FN, roi_aff, header=roi_hdr)
    save_as = os.path.join(out_path, subject + '_FN' + '.nii.gz')
    nib.save(nii, save_as)
    print("Saved", save_as)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_path',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/compare',
        help='path to save stats as csv'
    )

    parser.add_argument(
        '--subject_name',
        nargs='?',
        type=str,
        default='MSTHIGH_07_left',
        help='subject name'
    )

    parser.add_argument(
        '--pred',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/MSTHIGH_07/BL_left.nii.gz',
        help=''
    )

    parser.add_argument(
        '--gt',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/MSTHIGH_07/BL_left.nii.gz',
        help=''
    )

    args = parser.parse_args()

    main(args)