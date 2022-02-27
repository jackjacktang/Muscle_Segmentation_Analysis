# PARAMS=(
#     --save_path /raid/roger/dataset/Femur/compare
#     --subject MSTHIGH_07_BL_left
#     --pred /raid/roger/dataset/Femur/nii_test_out/MSTHIGH_07_BL_left_resunet.nii.gz
#     --gt /raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz
# )
# python compare.py ${PARAMS[@]}

PARAMS=(
    --save_path /raid/roger/dataset/Femur/compare
    --subject MSTHIGH_07_FU3_right
    --pred /raid/roger/dataset/Femur/nii_test_out/MSTHIGH_07_FU3_right_resunet.nii.gz
    --gt /raid/roger/dataset/Femur/MSTHIGH_07/FU3_right_femur.nii.gz
)
python compare.py ${PARAMS[@]}

PARAMS=(
    --save_path /raid/roger/dataset/Femur/compare
    --subject MSTHIGH_06_FU_left
    --pred /raid/roger/dataset/Femur/nii_test_out/MSTHIGH_06_FU_left_resunet.nii.gz
    --gt /raid/roger/dataset/Femur/MSTHIGH_06/FU_left_femur.nii.gz
)
python compare.py ${PARAMS[@]}

PARAMS=(
    --save_path /raid/roger/dataset/Femur/compare
    --subject MSTHIGH_06_FU4_left
    --pred /raid/roger/dataset/Femur/nii_test_out/MSTHIGH_06_FU4_left_resunet.nii.gz
    --gt /raid/roger/dataset/Femur/MSTHIGH_06/FU4_left_femur.nii.gz
)
python compare.py ${PARAMS[@]}
