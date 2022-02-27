import subprocess
import nibabel as nib


path = '/Volumes/Studies/LEGmuscle/Analysis/MSTHIGH_04_Jack/old/'
# sub = 'BL'
# leg = 'right'
c_b = 2
c_t = 18

# BL_leg_downsampled
# BL_leg_mask_downsampled_cal
# FU_left_ref2bl_downsampled.nii.gz
# FU_left_mask_downsampled_cal.nii.gz

for sub in ['FU']:
	subprocess.call(['mkdir',path+sub+'/final'])
	for leg in ['right']:

		img = ''
		new_img = ''
		mask = ''
		new_mask = ''
		if sub == 'BL':
			mask = path + sub + '/Analysis/' + sub + '_' + leg + '_mask_downsampled_cal.nii.gz'
			new_mask = path + sub + '/final/' + sub + '_' + leg + '_mask_downsampled_cal.nii.gz'
			img = path + sub + '/Analysis/' + sub + '_' + leg + '_downsampled.nii.gz'
			new_img = path + sub + '/final/' + sub + '_' + leg + '_downsampled.nii.gz'
		else:
			mask = path + sub + '/Analysis/' + sub + '_' + leg + '_mask_downsampled_cal.nii.gz'
			new_mask = path + sub + '/final/' + sub + '_' + leg + '_mask_downsampled_cal.nii.gz'
			img = path + sub + '/Analysis/' + sub + '_' + leg + '_ref2bl_downsampled.nii.gz'
			new_img = path + sub + '/final/' + sub + '_' + leg + '_ref2bl_downsampled.nii.gz'

		roi_img = nib.load(img)
		roi_dat = roi_img.get_data()
		subprocess.call(['fslroi',img,new_img,'1',str(roi_dat.shape[0]),'1',str(roi_dat.shape[1]),str(c_b),str(c_t-c_b+1)])

		roi_mask = nib.load(mask)
		roi_dat = roi_mask.get_data()
		subprocess.call(['fslroi',mask,new_mask,'1',str(roi_dat.shape[0]),'1',str(roi_dat.shape[1]),str(c_b),str(c_t-c_b+1)])