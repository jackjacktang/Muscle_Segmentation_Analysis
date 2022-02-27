################## Sample Pipeline to show volume################## 
# MSTHIGH_07/FU_left.nii.gz
## Femur seg
### BL
CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/pipeline_demo/BL_left.nii.gz --output /raid/roger/dataset/pipeline_demo/
### FU
CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/pipeline_demo/FU_left.nii.gz --output /raid/roger/dataset/pipeline_demo/ --followup --baseline_femur /raid/roger/dataset/pipeline_demo/BL_left_femur_mask.nii.gz


PARAMS=(
    --save_path /raid/roger/dataset/Femur/pipeline_stats
    --mode volume
    --subject MSTHIGH_07_BL_left
    --nifti_1 /raid/roger/dataset/pipeline_demo/BL_left_thigh_mask.nii.gz
)
python stats.py ${PARAMS[@]}

PARAMS=(
    --save_path /raid/roger/dataset/Femur/pipeline_stats
    --mode volume
    --subject MSTHIGH_07_FU_left
    --nifti_1 /raid/roger/dataset/pipeline_demo/FU_left_thigh_mask.nii.gz
)
python stats.py ${PARAMS[@]}



##################  Scripts for all tests ################## 
##################  Running Pipeline ################## 

# # MSTHIGH_07
# ## Ground truth femur registration
# ### Left
# #### BL
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/BL_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/
# #### FU
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU2_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU3_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU4_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
# ### Right
# #### BL
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/BL_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/
# #### FU
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU2_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU3_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU4_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
# ## Predicted femur registration
# ### Left
# #### BL
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/BL_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/
# #### FU
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_left_femur_mask.nii.gz
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU2_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_left_femur_mask.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU3_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_left_femur_mask.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU4_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_left_femur_mask.nii.gz
# ### Right
# #### BL
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/BL_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/
# #### FU
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_right_femur_mask.nii.gz
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU2_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_right_femur_mask.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU3_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_right_femur_mask.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_07/FU4_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_right_femur_mask.nii.gz

# # MSTHIGH_06
# ## Ground truth femur registration
# ### Left
# #### FU (as BL)
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU2_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU3_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU4_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
# ### Right
# #### FU (as BL)
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU2_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU3_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU4_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
# ## Predicted femur registration
# ### Left
# #### FU (as BL)
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU2_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_left_femur_mask.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU3_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_left_femur_mask.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU4_left.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_left_femur_mask.nii.gz
# ### Right
# #### FU (as BL)
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/
# #### FU2
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU2_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_right_femur_mask.nii.gz
# #### FU3
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU3_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_right_femur_mask.nii.gz
# #### FU4
# CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/Femur/MSTHIGH_06/FU4_right.nii.gz --output /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/ --followup --baseline_femur /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_right_femur_mask.nii.gz


##################  Generate SSIM Stats ################## 
# # MSTHIGH_07
# ## left
# ### pred
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU2_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU2_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU3_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU3_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU4_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU4_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ### ground truth
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU2_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU2_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU3_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU3_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_left_FU4_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU4_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## right
# ### pred
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU2_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU2_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}


# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU3_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU3_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU4_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU4_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ### ground truth
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU2_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU2_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}


# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU3_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU3_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_07_right_FU4_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_07/BL_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU4_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# # MSTHIGH_06
# ## left
# ### pred
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_left_FU2_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU2_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_left_FU3_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU3_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_left_FU4_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU4_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ### ground truth
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_left_FU2_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU2_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_left_FU3_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU3_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_left_FU4_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU4_left_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## right
# ### pred
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_right_FU2_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU2_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_right_FU3_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU3_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_right_FU4_pred
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU4_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ### ground truth
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_right_FU2_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU2_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_right_FU3_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU3_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode ssim
#     --subject MSTHIGH_06_right_FU4_truth
#     --nifti_1 
#     /raid/roger/dataset/Femur/MSTHIGH_06/FU_right_femur.nii.gz
#     --niftis 
#     /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU4_right_ref2bl_bone.nii.gz
# )
# python stats.py ${PARAMS[@]}

##################  Volume ################## 
# # MSTHIGH_07
# ## left
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_BL_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU2_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU2_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU3_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU3_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU4_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU4_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## right
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_BL_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU2_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU2_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU3_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU3_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU4_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/FU4_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}


# # MSTHIGH_06
# ## left

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU2_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU2_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU3_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU3_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU4_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU4_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## right

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU2_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU2_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU3_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU3_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU4_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-pred-reg/FU4_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## ground truth
# # MSTHIGH_07
# ## left
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_BL_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/BL_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU2_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU2_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU3_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU3_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU4_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU4_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## right
# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_BL_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/BL_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU2_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU2_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU3_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU3_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_07_FU4_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-truth-reg/FU4_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}


# # MSTHIGH_06
# ## left

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU2_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU2_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU3_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU3_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU4_left
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU4_left_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# ## right

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU2_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU2_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU3_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU3_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}

# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/pipeline_stats
#     --mode volume
#     --subject MSTHIGH_06_FU4_right
#     --nifti_1 /raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_06/use-truth-reg/FU4_right_thigh_mask.nii.gz
# )
# python stats.py ${PARAMS[@]}
