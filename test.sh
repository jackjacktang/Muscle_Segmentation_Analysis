# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0 >> ~/test_triplet_exp_1.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.1 >> ~/test_triplet_exp_2.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.3 >> ~/test_triplet_exp_3.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.5 >> ~/test_triplet_exp_4.out
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml >> ~/test_triplet_exp_5.out

# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.0 > ~/final_triplet_tests/resunet/test_triplet_exp_1.0_0.0.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.05 > ~/final_triplet_tests/resunet/test_triplet_exp_1.0_0.05.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.15 > ~/final_triplet_tests/test_triplet_exp_1.0_0.15.out

# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.01 > ~/final_triplet_tests/test_triplet_exp_1.0_0.01.out
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --dice_weight 0.9 --triplet_weight 0.05 > ~/final_triplet_tests/test_triplet_exp_0.9_0.05.out
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer.yml --triplet_weight 0.06 > ~/final_triplet_tests/test_triplet_exp_1.0_0.06.out


# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer1.yml --triplet_weight 0.05 > ~/final_triplet_tests/unet/test_triplet_exp_1.0_0.05.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer1.yml --triplet_mode no > ~/final_triplet_tests/unet/test_triplet_exp_1.0_0.0.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer2.yml --triplet_weight 0.05 > ~/final_triplet_tests/resAttUnet/test_triplet_exp_1.0_0.05.out 
# CUDA_VISIBLE_DEVICES=2,3 python train_amer.py --config configs/amer2.yml --triplet_mode no > ~/final_triplet_tests/resAttUnet/test_triplet_exp_1.0_0.0.out 

CUDA_VISIBLE_DEVICES=2,3 python test_amer.py --config configs/amer.yml
CUDA_VISIBLE_DEVICES=2,3 python test_amer.py --config configs/amer1.yml
CUDA_VISIBLE_DEVICES=2,3 python test_amer.py --config configs/amer2.yml