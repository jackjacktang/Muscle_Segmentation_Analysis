import subprocess, os, sys
from subprocess import Popen, PIPE
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import argparse
from scipy import ndimage

DTYPE = np.int



def main():
    input = '/Volumes/Studies/LEGmuscle/Analysis/MSTHIGH_04_Jack/FU/new/FU_right'

    print('########                 N3 correction...                ########')
    subprocess.call(['N4BiasFieldCorrection', '-i', input + '.nii.gz', '-o', input + '_n3.nii.gz'])

    input = input + '_n3'

    roi_norm(input)

    input = input + '_norm'

    up = 0.5
    low = 0.1

    roi_thr(input, low, up, input + '_' + str(low) + '_' + str(up))

    input = thr = input + '_' + str(low) + '_' + str(up)

    roi_img = nib.load(input + '.nii.gz')
    roi_dat = roi_img.get_data()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header
    print('########             Morphological processing...         ########')
    opt_dat = np.copy(roi_dat)

    eron = 1
    diln = 10
    if (int(eron) >= 1):
        for i in range(eron):
            print('########               Erosion iteration:' + str(i+1) + '             ########')
            opt_dat = new_roi_ero(opt_dat)

    if (int(diln) >= 1):
        for i in range(diln):
            print('########               Dilation iteration:' + str(i+1) + '             ########')
            opt_dat = new_roi_dil(opt_dat)

        input = mor = input + '_mor'
        opt_img = nib.Nifti1Image(opt_dat, roi_aff, header = roi_hdr)
        nib.save(opt_img, input + '.nii.gz')


    print('########                   Intersection...               ########')
    roi_intersect(thr, mor, input + '_mask')



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
    main()                        