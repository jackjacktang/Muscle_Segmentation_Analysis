import subprocess, os, sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from scipy import ndimage

def main():
	path="/Volumes/Studies/Research/LEGmuscle/Analysis/MSTHIGH_06_Jack/replicate/BL_left_processed_0.5-3.3.nii.gz"
	bone_path="/Volumes/Studies/Research/LEGmuscle/Analysis/MSTHIGH_06_Jack/replicate/BL_left_bonemask.nii.gz"
	bone_mask = nib.load(bone_path)
	roi_img = nib.load(path)
	segment_bone(roi_img,bone_mask)
	# option = sys.argv[2]

	

def calculate_slice(slice):
	return

def segment_bone(input, bone_mask):
	path="/Volumes/Studies/Research/LEGmuscle/Analysis/MSTHIGH_06_Jack/replicate/BL_left_bonemask.nii.gz"
	input_dat = input.get_data()
	input_aff = input.affine
	input_hdr = input.header

	bone_dat = bone_mask.get_data()
	bone_aff = bone_mask.affine
	bone_hdr = bone_mask.header

	# fill bone hole
	print(bone_dat.shape)
	# for i in range(0,bone_dat.shape[2]):
	for i in range(200,201):
		temp = bone_dat[:,:,i]
		temp = ndimage.binary_fill_holes(temp, structure=np.ones(20,20)).astype(int)
		# bonedat = ndimage.binary_fill_holes(bone_dat, structure=np.ones((5,5,5))).astype(int)
	opt_img = nib.Nifti1Image(bone_dat, bone_aff, header = bone_hdr)
	plt.scatter(temp[:,0],temp[:,1])
	nib.save(opt_img, "/Volumes/Studies/Research/LEGmuscle/Analysis/MSTHIGH_06_Jack/replicate/BL_left_bonemask_filled.nii.gz")

	return

def remove_outliers(input):
	path="/Volumes/Studies/Research/LEGmuscle/Analysis/MSTHIGH_06_Jack/replicate/BL_left_processed_0.5-3.3.nii.gz"
	# path="../replicate/BL_left_processed_0.5-3.3.nii.gz"
	# path="../replicate/BL_left.nii.gz"
	roi_img = nib.load(path)
	# option = sys.argv[2]

	roi_dat = roi_img.get_data()
	roi_aff = roi_img.affine
	roi_hdr = roi_img.header

	print(roi_dat.shape)
	print("roi_dat: ",np.count_nonzero(roi_dat))

	# for i in roi_dat.shape[2]:
	for i in range(0,roi_dat.shape[2]):
	# for i in range(253,254):
		temp = roi_dat[:,:,i]
		print(temp.shape)
		# temp[np.where(temp > 0)] = 1.0
		# print roi_dat.shape
		# print temp.shape
		# print temp.max

		fore = np.where(temp>0)
		fore = np.array(fore)
		# np.swapaxes(fore,0,1)
		# for i in fore:
		# 	print i
		fore_no = fore[1].shape[0]
		# print(fore_no)
		fore_temp = np.zeros((fore[1].shape[0],2))
		fore_temp[:,0] = fore[0]
		fore_temp[:,1] = fore[1]
		# fore = np.reshape(fore, (9508,2))
		# print fore_temp
		# fore.flatten()
		# print fore_temp.shape
		fore_mean_x = np.sum(fore[0])/fore[0].shape[0]
		fore_mean_y = np.sum(fore[1])/fore[1].shape[0]
		print(fore_mean_x)
		print(fore_mean_y)
		diff = np.zeros((fore[1].shape[0],1))
		ind = 0
		remain_ind = np.asarray([])
		for f in fore_temp:
			distance = np.linalg.norm(f-[fore_mean_x,fore_mean_y])
			# if distance < 63:
			# 	# print(distance)
			# 	if remain_ind.size == 0:
			# 		remain_ind = np.asarray(ind)
			# 	else:
			# 		remain_ind = np.append(remain_ind,ind)
			diff[ind] = distance
			ind += 1

		# get 10% outlier

		diff_sort = np.copy(diff)
		diff_sort = np.sort(diff_sort)
		outlier_start = fore_no * 0.2
		# thr = diff_sort[int(outlier_start)]
		thr = 70
		# thr = 100
		print(thr)

		print(diff_sort[0])
		print(diff_sort[-1])
		ind = 0
		for f in fore_temp:
			# print(diff[ind],thr)
			if diff[ind] < thr:
				if remain_ind.size == 0:
					remain_ind = np.asarray(ind)
				else:
					remain_ind = np.append(remain_ind,ind)
			ind += 1

		# remain = remain[data - np.mean(data) < m * np.std(data)]
		print(remain_ind.shape)
		print("max_distance: ",np.max(diff))
		print("min_distance: ",np.min(diff))
		print("mean_distance: ",np.mean(diff))
		remain = fore_temp[remain_ind,:]

		# np.concatenate((a, b), axis=0)

	# for f in fore_temp
	# print(remain.shape)

		mask = np.zeros((512,512))
		for r in remain:
			mask[int(r[0])][int(r[1])] = 1
		print("mask", mask.shape)
		print(mask)
		# sum_fore = (mask==1).sum
		print("fore: ",np.count_nonzero(mask))
		roi_dat[:,:,i] = mask

		# plt.scatter(remain[:,0],remain[:,1])
		# plt.scatter(fore_mean_x,fore_mean_y,c='red')


	opt_img = nib.Nifti1Image(roi_dat, roi_aff, header = roi_hdr)
	nib.save(opt_img, "/Volumes/Studies/Research/LEGmuscle/Analysis/MSTHIGH_06_Jack/replicate/BL_left_mor_0.5-3.3.nii.gz")


if __name__ == "__main__":
	main()