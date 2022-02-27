import subprocess

p = '/Volumes/Studies/LEGmuscle/Analysis/MSTHIGH_04_Jack/'
tp = ['FU']
leg = ['right']
for t in tp:
	for l in leg:
		subprocess.call(['flirt', '-ref', p + 'BL/Analysis/BL_' + l + '.nii.gz', '-in', p + 'BL/final/BL_' + l + '_downsampled.nii.gz', '-omat', p + 'BL/Analysis/down2ori_' + l + '.mat', '-nosearch', '-dof', '6', '-out', p + 'BL/Analysis/BL_' + l + '_low2high.nii.gz'])
		subprocess.call(['convert_xfm', '-omat', p + 'BL/Analysis/down2ori_' + l + '_inverse.mat', '-inverse', p + 'BL/Analysis/down2ori_' + l + '.mat'])
		subprocess.call(['flirt', '-ref', p + 'BL/final/BL_' + l + '_downsampled.nii.gz', '-in', p + t + '/Analysis/' + t + '_' + l + '_ref2bl.nii.gz', '-out', p + t + '/final/' + t + '_' + l + '_ref2bl_downsampled.nii.gz', '-applyxfm', '-init', p + 'BL/Analysis/down2ori_' + l + '_inverse.mat'])
		# subprocess.call(['flirt', '-ref', p + 'BL/final/BL_' + l + '_downsampled.nii.gz', '-in', p + t + '/Analysis/' + t + '_' + l + '_mask.nii.gz', '-out', p + t + '/final/' + t + '_' + l + '_updated_cal.nii.gz', '-applyxfm', '-init', p + 'BL/Analysis/down2ori_' + l + '_inverse.mat'])
