import subprocess, os, sys
from subprocess import Popen, PIPE
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse

from scipy import ndimage

DTYPE = np.int

def main():
    parser = argparse.ArgumentParser(description='Arguments for LegPro 2.0.')
    parser.add_argument('-s', type=str, default=None, required=True, help='The subject folder name')
    # parser.add_argument('-t', nargs='+', default=['bl'], required=False, help='How many timepoints to process',)
    parser.add_argument('-t', type=str, default=None, required=True, help='The timepoint to process',)

    # SNAC path
    parser.add_argument('-p', type=str, default='/Volumes/Studies/LEGmuscle/Analysis/MSTHIGH_', required=False, help='The subject folder path')
    # Home path
    # parser.add_argument('-p', type=str, default='/media/jacktang/Work/SNAC/LEGmuscle/Analysis/', required=False, help='The subject folder path')
    parser.add_argument('-bottom', type=int, default=0, help='z-axis slice to start')
    parser.add_argument('-top', type=int, default=45, help='z-axis slice to end')
    parser.add_argument('-upthr', type=float, default=2.5, help='Upper threshold for segmenting the muscle')
    parser.add_argument('-lowthr', type=float, default=0.0, help='Lower threshold for segmenting the muscle')
    parser.add_argument('-eron', type=int, default=2, help='Time of erosions')
    parser.add_argument('-diln', type=int, default=10, help='Time of dilations')
    parser.add_argument('-clthr', type=float, default=0.2, help='Cleaning threshold')
    parser.add_argument('-calb', type=int, default=0, help='z-axis slice to start calculation')
    parser.add_argument('-calt', type=int, default=45, help='z-axis slice to end calculation')

    parser.add_argument('-l', dest='left', action='store_true' ,help='Process left leg')
    parser.set_defaults(left=False)
    parser.add_argument('-r', dest='right', action='store_true' ,help='Process right leg')
    parser.set_defaults(right=False)

    parser.add_argument('--crop', dest='crop', action='store_true', help='Perform cropping')
    parser.add_argument('--no-crop', dest='crop', action='store_false', help='Skip cropping')
    parser.set_defaults(crop=True)

    parser.add_argument('--snip', dest='snip', action='store_true', help='Perform snipping')
    parser.add_argument('--no-snip', dest='snip', action='store_false', help='Skip snipping')
    parser.set_defaults(snip=True)

    parser.add_argument('--n3', dest='n3', action='store_true', help='Perform N3 correction')
    parser.add_argument('--no-n3', dest='n3', action='store_false', help='Skip N3 correction')
    parser.set_defaults(n3=True)

    parser.add_argument('--norm', dest='norm', action='store_true', help='Perform normalization')
    parser.add_argument('--no-norm', dest='norm', action='store_false', help='Skip normalization')
    parser.set_defaults(norm=True)

    parser.add_argument('--thr', dest='thr', action='store_true', help='Perform thresholding')
    parser.add_argument('--no-thr', dest='thr', action='store_false', help='Skip thresholding')
    parser.set_defaults(thr=True)

    parser.add_argument('--mor', dest='mor', action='store_true', help='Perform morphlogical process')
    parser.add_argument('--no-mor', dest='mor', action='store_false', help='Skip morphological process')
    parser.set_defaults(mor=True)

    parser.add_argument('--inter', dest='inter', action='store_true', help='Perform intersection')
    parser.add_argument('--no-inter', dest='inter', action='store_false', help='Skip intersection')
    parser.set_defaults(inter=True)

    parser.add_argument('--dsam', dest='resam', action='store_true', help='Perform downsampling')
    parser.add_argument('--no-downsam', dest='resam', action='store_false', help='Skip downsampling')
    parser.set_defaults(resam=True)

    parser.add_argument('--usam', dest='upsam', action='store_true', help='Perform upsampling')
    parser.add_argument('--no-upsam', dest='upsam', action='store_false', help='Skip upsampling')
    parser.set_defaults(upsam=True)

    parser.add_argument('--clean', dest='clean', action='store_true', help='Perform resampling')
    parser.add_argument('--no-clean', dest='clean', action='store_false', help='Skip resampling')
    parser.set_defaults(clean=True)

    parser.add_argument('--cal', dest='cal', action='store_true', help='Perform calculating')
    parser.add_argument('--no-cal', dest='cal', action='store_false', help='Skip calculating')
    parser.set_defaults(cal=True)

    args = parser.parse_args()

    target_path = args.p + args.s + '/' + args.t.upper() + '/Analysis'
    timepoint = args.t.upper()

    print('########                Reading arguments...             ########')
    print('########     Staring processing ' + args.s + args.t + '...      ########')

    if (args.crop == True and (os.path.isfile(target_path + '/' + timepoint + '_left.nii.gz') == False or os.path.isfile(target_path + '/' + timepoint + '_right.nii.gz') == False)):
        print('########       Cropping left leg and right leg...        ########')
        crop_middle(target_path + '/' + timepoint, args.bottom, args.top)
    else:
        print('########                Skip cropping...                 ########')

    spec = retrieve_spec(target_path,timepoint,args.left,args.right)

    left_current_path = target_path + '/' + timepoint + '_left'
    right_current_path = target_path + '/' + timepoint + '_right'

    if (args.left == True and os.path.isfile(left_current_path + '_bone_dil.nii.gz') == False):
        subprocess.call(['fslmaths', left_current_path + '_bone.nii.gz', '-dilM', '-kernel', 'boxv', '5', left_current_path + '_bone_dil.nii.gz'])
        subprocess.call(['fslmaths', left_current_path + '.nii.gz', '-mas', left_current_path + '_bone_dil.nii.gz', left_current_path + '_bone_dil_mask.nii.gz'])
    if (args.right == True and os.path.isfile(right_current_path + '_bone_dil.nii.gz') == False):
        subprocess.call(['fslmaths', right_current_path + '_bone.nii.gz', '-dilM', '-kernel', 'boxv', '5', right_current_path + '_bone_dil.nii.gz'])
        subprocess.call(['fslmaths', right_current_path + '.nii.gz', '-mas', right_current_path + '_bone_dil.nii.gz', right_current_path + '_bone_dil_mask.nii.gz'])

    if ('fu' in args.t):
        print('########             Register to baseline...             ########')
        if (args.left == True and os.path.isfile(left_current_path + '_ref2bl.nii.gz') == False):
            subprocess.call(['flirt', '-in', left_current_path + '_bone_dil_mask.nii.gz', '-ref', args.p + args.s + '/BL/Analysis/BL_left_bone_dil_mask.nii.gz', '-out', left_current_path + '_ref2bl_bone.nii.gz', '-omat',left_current_path + '_ref2bl.mat','-dof', '6'])
            subprocess.call(['flirt', '-in', left_current_path + '.nii.gz', '-applyxfm', '-init', left_current_path + '_ref2bl.mat','-ref',args.p + args.s + '/BL/Analysis/BL_left_bone_dil_mask.nii.gz','-out',left_current_path + '_ref2bl.nii.gz'])
        if (args.right == True and os.path.isfile(right_current_path + '_ref2bl.nii.gz') == False):
            subprocess.call(['flirt', '-in', right_current_path + '_bone_dil_mask.nii.gz', '-ref', args.p + args.s + '/BL/Analysis/BL_right_bone_dil_mask.nii.gz', '-out', right_current_path + '_ref2bl_bone.nii.gz', '-omat',right_current_path + '_ref2bl.mat','-dof', '6'])
            subprocess.call(['flirt', '-in', right_current_path + '.nii.gz', '-applyxfm', '-init', right_current_path + '_ref2bl.mat','-ref',args.p + args.s + '/BL/Analysis/BL_right_bone_dil_mask.nii.gz','-out',right_current_path + '_ref2bl.nii.gz'])

        left_current_path = left_current_path + '_ref2bl'
        right_current_path = right_current_path + '_ref2bl'

    if (args.snip == True):
        print('########                    Snipping...                   ########')
        if (args.left == True and os.path.isfile(left_current_path + '_snip.nii.gz') == False):
            snip(left_current_path, args.bottom, args.top)
        if (args.right == True and os.path.isfile(right_current_path + '_snip.nii.gz') == False):
            snip(right_current_path, args.bottom, args.top)
    else:
        print('########                 Skip snipping...                 ########')
    left_current_path = left_current_path + '_snip'
    right_current_path = right_current_path + '_snip'

    if (args.n3 == True):
        print('########                 N3 correction...                ########')
        if (args.left == True):
            # subprocess.call(['mri_nu_correct.mni', '--i', left_current_path + '.nii.gz', '--o', left_current_path + '_n3.nii.gz'])
            subprocess.call(['N4BiasFieldCorrection', '-i', left_current_path + '.nii.gz', '-o', left_current_path + '_n3.nii.gz'])
        if (args.right == True):
            # subprocess.call(['mri_nu_correct.mni', '--i', right_current_path + '.nii.gz', '--o', right_current_path + '_n3.nii.gz'])
            subprocess.call(['N4BiasFieldCorrection', '-i', right_current_path + '.nii.gz', '-o', right_current_path + '_n3.nii.gz'])
    else:
        print('########              Skip N3 correction...              ########')
    left_current_path = left_current_path + '_n3'
    right_current_path = right_current_path + '_n3'

    if (args.norm == True):
        print('########                 Normalization...                ########')
        if (args.left == True):
            roi_norm(left_current_path)
        if (args.right == True):
            roi_norm(right_current_path)
    else:
        print('########              Skip normalization...              ########')
    left_current_path = left_current_path + '_norm'
    right_current_path = right_current_path + '_norm'

    if (args.thr == True):
        print('########                 Thresholding...                 ########')
        if (args.left == True):
            roi_thr(left_current_path, str(args.lowthr), str(args.upthr), left_current_path + '_' + str(args.lowthr) + '_' + str(args.upthr))
        if (args.right == True):
            roi_thr(right_current_path, str(args.lowthr), str(args.upthr), right_current_path + '_' + str(args.lowthr) + '_' + str(args.upthr))
    else:
        print('########               Skip thresholding...              ########')
    left_current_path = left_thr = left_current_path + '_' + str(args.lowthr) + '_' + str(args.upthr)
    right_current_path = right_thr = right_current_path + '_' + str(args.lowthr) + '_' + str(args.upthr)

    left_temp = left_current_path
    right_temp = right_current_path
    if (args.mor == True):
        if args.left == True:
            roi_img = nib.load(left_current_path + '.nii.gz')
        elif args.right == True:
            roi_img = nib.load(right_current_path + '.nii.gz')
        roi_dat = roi_img.get_data()
        roi_aff = roi_img.affine
        roi_hdr = roi_img.header
        print('########             Morphological processing...         ########')
        opt_dat = np.copy(roi_dat)
        if (int(args.eron) >= 1):
            for i in range(args.eron):
                print('########               Erosion iteration:' + str(i+1) + '             ########')
                opt_dat = new_roi_ero(opt_dat)
            # print('########               Erosion iteration: 1             ########')
            # left_temp = left_current_path + '_ero1'
            # if (args.left == True):
            #     # subprocess.call(['fslmaths', left_current_path + '.nii.gz', '-ero', left_temp + '.nii.gz'])
            #     roi_erode(left_current_path, left_temp, 2)

            # right_temp = right_current_path + '_ero1'
            # if (args.right == True):
            #     roi_erode(right_current_path, right_temp, 2)

            # if (int(args.eron) >= 2):
            #     for i in range(int(args.eron-1)):
            #         print('########               Erosion iteration: ' + str(i+2) + '              ########')
            #         left_temp = left_current_path + '_ero' + str(i+2)
            #         if (args.left == True):
            #             # subprocess.call(['fslmaths', left_current_path + '_ero' + str(i+1) + '.nii.gz', '-ero', left_temp + '.nii.gz'])
            #             roi_erode(left_current_path + '_ero' + str(i+1), left_temp, 2)

            #         right_temp = right_current_path + '_ero' + str(i+2)
            #         if (args.right == True):
            #             roi_erode(right_current_path + '_ero' + str(i+1), right_temp, 2)


        if (int(args.diln) >= 1):
            for i in range(args.diln):
                print('########               Dilation iteration:' + str(i+1) + '             ########')
                opt_dat = new_roi_dil(opt_dat)
        if (args.left == True):
            left_temp = left_temp + '_mor'
            left_current_path = left_temp
            opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
            nib.save(opt_img, left_current_path + '.nii.gz')

        if (args.right == True):
            right_temp = right_temp + '_mor'
            right_current_path = right_temp
            opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
            nib.save(opt_img, right_current_path + '.nii.gz')


        # if (int(args.diln) >= 1):
        #     print('########              Dilation iteration: 1              ########')
        #     left_temp = left_current_path + '_dil1'
        #     if (args.left == True):
        #         # subprocess.call(['fslmaths', left_current_path + '.nii.gz', '-dilM', left_temp + '.nii.gz'])
        #         roi_dilate(left_current_path, left_temp, 2)

        #     right_temp = right_current_path + '_dil1'
        #     if (args.right == True):
        #         roi_dilate(right_current_path, right_temp, 2)

        #     if (int(args.diln) >= 2):
        #         for i in range(int(args.diln-1)):
        #             print('########              Dilation iteration: ' + str(i+2) + '             ########')
        #             left_temp = left_current_path + '_dil' + str(i+2)
        #             if (args.left == True):
        #                 # subprocess.call(['fslmaths', left_current_path + '_dil' + str(i+1) + '.nii.gz', '-dilM', left_temp + '.nii.gz'])
        #                 roi_dilate(left_current_path + '_dil' + str(i+1), left_temp, 2)

        #             right_temp = right_current_path + '_dil' + str(i+2)
        #             if (args.right == True):
        #                 roi_dilate(right_current_path + '_dil' + str(i+1), right_temp, 2)
    else:
        print('########         Skip morphological processing...        ########')

    if (args.inter == True):
        print('########                   Intersection...               ########')
        if (args.left == True):
            roi_intersect(left_thr, left_temp, target_path + '/' + timepoint + '_left_mask')
        if (args.right == True):
            roi_intersect(right_thr, right_temp, target_path + '/' + timepoint + '_right_mask')

    else: 
        print('########                Skip intersection...             ########')

    if (args.resam == True):
        print('########                   Downsampling...               ########')

        if (args.left == True):
            subprocess.call(['mri_convert', '-vs', str(spec[0][0]), str(spec[0][1]), str(10), target_path + '/' + timepoint + '_left_mask.nii.gz', target_path + '/' + timepoint + '_left_mask_downsampled_before_manual.nii.gz'])
            if ('bl' in args.t):
                subprocess.call(['mri_convert', '-vs', str(spec[0][0]), str(spec[0][1]), str(10), target_path + '/' + timepoint + '_left_snip.nii.gz', target_path + '/' + timepoint + '_left_downsampled.nii.gz'])
            elif ('fu' in args.t):
                subprocess.call(['mri_convert', '-vs', str(spec[0][0]), str(spec[0][1]), str(10), target_path + '/' + timepoint + '_left_ref2bl_snip.nii.gz', target_path + '/' + timepoint + '_left_ref2bl_downsampled.nii.gz'])
        if (args.right == True):
            subprocess.call(['mri_convert', '-vs', str(spec[1][0]), str(spec[1][1]), str(10), target_path + '/' + timepoint + '_right_mask.nii.gz', target_path + '/' + timepoint + '_right_mask_downsampled_before_manual.nii.gz'])
            if ('bl' in args.t):
                subprocess.call(['mri_convert', '-vs', str(spec[1][0]), str(spec[1][1]), str(10), target_path + '/' + timepoint + '_right_snip.nii.gz', target_path + '/' + timepoint + '_right_downsampled.nii.gz'])
            elif ('fu' in args.t):
                subprocess.call(['mri_convert', '-vs', str(spec[1][0]), str(spec[1][1]), str(10), target_path + '/' + timepoint + '_right_ref2bl_snip.nii.gz', target_path + '/' + timepoint + '_right_ref2bl_downsampled.nii.gz'])
    else:
        print('########               Skip downsampling...              ########')


    if (args.upsam == True):
        print('########                   Upsampling...                 ########')
        if (args.left == True):
            subprocess.call(['flirt', '-in', target_path + '/' + timepoint + '_left_mask_downsampled.nii.gz', '-ref', target_path + '/' + timepoint + '_left_mask.nii.gz', '-applyxfm', '-init', 'resample.mat', '-out', target_path + '/' + timepoint + '_left_mask_upsampled.nii.gz'])        
        if (args.right == True):
            subprocess.call(['flirt', '-in', target_path + '/' + timepoint + '_right_mask_downsampled.nii.gz', '-ref', target_path + '/' + timepoint + '_right_mask.nii.gz', '-applyxfm', '-init', 'resample.mat', '-out', target_path + '/' + timepoint + '_right_mask_upsampled.nii.gz']) 
    else:
        print('########               Skip upsampling...                ########')

    if (args.clean == True):
        print('########                    Cleaning...                  ########')
        if (args.left == True):
            subprocess.call(['fslmaths', target_path +  '/' + timepoint + '_left_mask_upsampled.nii.gz', '-thr', str(args.clthr), '-bin', target_path + '/' + timepoint + '_left_clean_temp.nii.gz'])      
            subprocess.call(['fslmaths', target_path + '/' + timepoint + '_left_mask.nii.gz', '-mas', target_path + '/' + timepoint + '_left_clean_temp.nii.gz', target_path + '/' + timepoint + '_left_final.nii.gz'])  
        if (args.right == True):
            subprocess.call(['fslmaths', target_path + '/' + timepoint + '_right_mask_upsampled.nii.gz', '-thr', str(args.clthr), '-bin', target_path + '/' + timepoint + '_right_clean_temp.nii.gz'])      
            subprocess.call(['fslmaths', target_path + '/' + timepoint + '_right_mask.nii.gz', '-mas', target_path + '/' + timepoint + '_right_clean_temp.nii.gz', target_path + '/' + timepoint + '_right_final.nii.gz']) 
    else:
        print('########                Skip cleaning...                 ########')

    if (args.cal == True):
        print('########          Calculating muscle volume...           ########')
        if (args.left == True):
            vol = cal_muscle_vol(target_path + '/' + timepoint + '_left_mask_downsampled_cal', spec[0], args.calb, args.calt)
            # vol = cal_vol(target_path + '/' + timepoint + '_left_mask_downsampled_cal', spec[0])
            print('The muscle volume for', args.s, timepoint, 'left is: ', vol)
        if (args.right == True): 
            vol = cal_muscle_vol(target_path + '/' + timepoint + '_right_mask_downsampled_cal', spec[1], args.calb, args.calt)
            # vol = cal_vol(target_path + '/' + timepoint + '_right_mask_downsampled_cal', spec[1])
            print('The muscle volume for', args.s, timepoint, 'right is: ', vol)
    else:
        print('########                Skip calculating...              ########')


############################################ 
#            Support functions             *
############################################

#
# Normalize the ROI images
#
def roi_norm(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    # opt_dat = np.divide(roi_dat, 255.0)
    roi_dat *= 1/roi_dat.max()
    opt_dat = roi_dat
    print('max: ',np.max(opt_dat))
    print('min: ',np.min(opt_dat))
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, roi_fln + '_norm.nii.gz')

    return 1

def roi_norm_prev(roi_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = np.zeros_like(roi_dat)
    opt_dat = roi_dat - roi_dat.mean()
    opt_dat = opt_dat / roi_dat.std()
    print('max: ',np.max(opt_dat))
    print('min: ',np.min(opt_dat))
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, roi_fln + '_norm.nii.gz')

    return 1

#
# Binarize the ROI images by choosing a threshold
#
def roi_thr(roi_fln, min, max, opt_fln):

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    opt_dat = np.zeros_like(roi_dat)
    roi_coords = (roi_dat >= np.float(min)) & (roi_dat <= np.float(max))
    # opt_dat[roi_coords] = roi_dat[roi_coords]
    opt_dat[roi_coords] = 1.0
    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)

    if opt_fln is None:
        opt_fln = roi_fln + '_thr_'+ str(min) + '_' + str(max)

    nib.save(opt_img, opt_fln + '.nii.gz')

    return 1
#
# Select the struct to use
#
def roi_struct(option = 2):

    # the 26 neighbor system
    struct_26 = np.zeros((3, 3, 3))

    # 18 neighbors, take 8 corners out
    struct_18 = np.zeros((3, 3, 3))
    struct_18[:, 1, :] = 1
    struct_18[1, :, :] = 1
    struct_18[:, :, 1] = 1

    # 6 neighbors, only keep the 6 facet centers 
    struct_6 = np.zeros((3, 3, 3))
    struct_6[1, 1, 0] = 1
    struct_6[1, 1, 2] = 1
    struct_6[0, 1, 1] = 1
    struct_6[2, 1, 1] = 1
    struct_6[1, 0, 1] = 1
    struct_6[1, 2, 1] = 1

    # 26 neighbors
    if option == 1:
        return struct_26

    # 18 neighbors, take 8 corners out
    elif option == 2:   
        return struct_18     

    # 6 neighbors, only keep the 6 facet centers 
    elif option == 3:
        return struct_6
    # return 18 neighbor if option is not specified
    else:
        return struct_18

#
# Dilation
#
def roi_dilate(roi_fln, opt_name, option = 2):


    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')

    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
   
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_dilation(opt_dat_bin, structure = struct).astype(DTYPE)
    opt_fln = opt_name

    opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    nib.save(opt_img, opt_fln + '.nii.gz')
    
    return 1 

def new_roi_dil(roi_dat, option=2):
    struct = roi_struct(option)
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_dilation(opt_dat_bin, structure = struct).astype(DTYPE)

    return opt_dat

def new_roi_ero(roi_dat, option=2):
    struct = roi_struct(option)
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_erosion(opt_dat_bin, structure = struct).astype(DTYPE)

    return opt_dat

#
# Erosion
#
def roi_erode(roi_fln, opt_name, option = 2):


    struct = roi_struct(option)
    print('INFO: the kernel ', struct)

    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')


    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
   
    opt_dat_bin = np.zeros_like(roi_dat)
    opt_dat_bin[np.where(roi_dat > 0)] = 1.0
    opt_dat = ndimage.binary_erosion(opt_dat_bin, structure = struct).astype(DTYPE)
    opt_fln = opt_name

    # opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
    # nib.save(opt_img, opt_fln + '.nii.gz')
    
    return 1 

#
# Intersect with another ROI
#
def roi_intersect(roi1_fln, roi2_fln, opt_fln):

    try:      
        roi1_img = nib.load(roi1_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi1_img = nib.load(roi1_fln + '.nii')
    roi1_dat = roi1_img.get_data()
    roi1_aff = roi1_img.affine
    roi1_hdr = roi1_img.header


    try:
        roi2_img = nib.load(roi2_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the ROI image in NII.GZ. Trying to load the .NII instead.')
        roi2_img = nib.load(roi2_fln + '.nii')
    roi2_dat = roi2_img.get_data()
    roi2_aff = roi2_img.affine
    roi2_hdr = roi2_img.header

    if not (roi1_img.shape == roi2_img.shape):
        print('ERROR: image dimension not match!!!')
        print('INFO: 1st lesion mask ', roi1_img.shape, roi1_img.get_data_dtype())
        print('INFO: 2nd lesion mask ', roi2_img.shape, roi2_img.get_data_dtype())
        return -1

    else:
        opt_dat = np.zeros_like(roi1_dat)
        opt_dat[(roi1_dat > 0) & (roi2_dat > 0)] = 1.

    opt_img = nib.Nifti1Image(opt_dat, roi1_aff, header = roi1_hdr)

    if opt_fln is None: 
        opt_fln = roi_fln + '.nii.gz'
    else: 
        opt_fln = opt_fln + '.nii.gz'

    nib.save(opt_img, opt_fln)
    
    return 1    


#
# Find the cut slice of the image
#
def crop_middle(roi_fln, bottom, top):
    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')

    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    projection_z = np.mean(roi_dat,axis=2)
    projection_z = np.rot90(projection_z)

    projection_zx = np.mean(projection_z, axis=0)
    left_max = np.argmax(projection_zx[0:int(np.shape(projection_zx)[0]/2)])
    right_max = np.argmax(projection_zx[int(np.shape(projection_zx)[0]/2):int(np.shape(projection_zx)[0])]) + int(np.shape(projection_zx)[0]/2)
    middle_point = np.argmin(projection_zx[left_max:right_max])
    middle_point += left_max

    # # plotting the result
    # fig = plt.figure(figsize=(6.5,3))

    # plt1 = fig.add_subplot(121)
    # plt2 = fig.add_subplot(122)
    # plt1.imshow(projection_z)
    # ind = 1
    # for i in projection_zx:
    #     plt2.scatter([ind],[i],c='red')
    #     ind += 1 

    # plt2.scatter(right_max, projection_zx[right_max], c='blue')
    # ind += 1
    # plt2.scatter(left_max, projection_zx[left_max], c='blue')
    # ind += 1
    # plt2.scatter(middle_point, projection_zx[middle_point], c='black')

    # plt1.plot([middle_point,middle_point], [0,roi_dat.shape[1]-1], 'k-')
    # plt1.axis('off')
    # plt2.plot([middle_point,middle_point], [0,8000], 'k-')
    # plt.show()

    print('########           Cropping ' + roi_fln + ' left         ########')
    # crop_command = ["fslroi", roi_fln, roi_fln + "_left.nii.gz", str(middle_point), str(roi_dat.shape[0]-middle_point), '0', str(roi_dat.shape[1]), str(bottom), str(top-bottom)]
    crop_command = ["fslroi", roi_fln, roi_fln + "_left.nii.gz", str(middle_point), str(roi_dat.shape[0]-middle_point), '0', str(roi_dat.shape[1]), '0', str(roi_dat.shape[2])]
    print(crop_command)
    subprocess.call(crop_command)
    print('########          Cropping ' + roi_fln + ' right         ########')
    # crop_command = ["fslroi", roi_fln, roi_fln + "_right.nii.gz", '0', str(middle_point), '0', str(roi_dat.shape[1]), str(bottom), str(top-bottom)]
    crop_command = ["fslroi", roi_fln, roi_fln + "_right.nii.gz", '0', str(middle_point), '0', str(roi_dat.shape[1]), '0', str(roi_dat.shape[2])]
    subprocess.call(crop_command)
    print(crop_command)
    return 1

# snip, only keep the part of interest
def snip(roi_fln, bottom, top):
    try:      
        roi_img = nib.load(roi_fln + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(roi_fln + '.nii')

    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    crop_command = ["fslroi", roi_fln, roi_fln + "_snip.nii.gz", '1', str(roi_dat.shape[0]), '1', str(roi_dat.shape[1]), str(bottom), str(top-bottom)]
    subprocess.call(crop_command)


# 
# get the how many milimeters in each direction (x,y,z)
# result is in the format of [x_left, y_left, z_left; x_right, y_right, z_right]
# 
def retrieve_spec(target_path,timepoint,left,right):
    output = np.array([])
    result = np.zeros((2, 3))
    if (left == True):
        stdout = Popen(['fslinfo ' + target_path + '/' + timepoint + '_left.nii.gz'], shell=True, stdout=PIPE).stdout
        for line in stdout.readlines():
            if (output.size == 0):
                output = np.array(line)
            else:
                output = np.append(output,line) 

        x = str(output[6])
        result[0][0] = x[17:-3]
        y = str(output[7])
        result[0][1] = y[17:-3]
        z = str(output[8])
        result[0][2] = z[17:-3]

    if (right == True):
        output = np.array([])
        stdout = Popen(['fslinfo ' + target_path + '/' + timepoint + '_right.nii.gz'], shell=True, stdout=PIPE).stdout
        for line in stdout.readlines():
            if (output.size == 0):
                output = np.array(line)
            else:
                output = np.append(output,line) 

        x = str(output[6])
        result[1][0] = x[17:-3]
        y = str(output[7])
        result[1][1] = y[17:-3]
        z = str(output[8])
        result[1][2] = z[17:-3]

    return result

# 
# Calculate the muscle volume
# 
def cal_muscle_vol(path, spec, bottom, top):
    try:      
        roi_img = nib.load(path + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(path + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    temp = roi_dat.copy()
    temp = temp[:,:,bottom-1:top-1]
    # print('295: ',np.count_nonzero(temp[:,:,295]==1))
    voxel_no = np.count_nonzero(temp==1)
    result = float(voxel_no) * float(spec[0]) * float(spec[1]) * float(spec[2]) * 10
    print(spec[0], spec[1], spec[2])

    return result

def cal_vol(path,spec):
    try:      
        roi_img = nib.load(path + '.nii.gz')
    except FileNotFoundError:
        print('INFO: Cannot find the .nii.gz file, loading .nii instead')
        roi_img = nib.load(path + '.nii')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    voxel_no = np.count_nonzero(roi_dat==1)
    result = float(voxel_no) * float(spec[0]) * float(spec[1]) * float(spec[2]) * 10
    print(spec[0], spec[1], spec[2])
    return result




if __name__ == "__main__":
    main()