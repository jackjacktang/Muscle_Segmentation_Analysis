# Deep_Muscle_Segmentation

The pipeline takes a thigh a 3D magnetic resonance (MR) image in nifti format and outputs the volume of thigh muscle. The pipeline contains three main steps: femur segmentation, registration if the image is a follow-up case, thigh segmentation.

To run the pipeline:

## 1. Setup the dependencies (Mac & Ubuntu)

Install the packeges required:

```bash
pip install -r requirements.txt
```

## 2. Train models for femur and thigh

Edit `femur.yml` and `unet.yml` or `resunet.yml` or `res_attn_unet.yml`, depending on which deep neural networks you want to use. After that, train the network. E.g.:

Femur:
```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --config configs/femur.yml
```

Thigh:

```bash
CUDA_VISIBLE_DEVICES=2,3 python train.py --config configs/resunet.yml
```

## 3. Run main.py with stats (the pipeline main program)

Run `main.py` with correct arguments (with correct femur and thigh muscle weights path showed in outputs from step 2). Then run `stats.py` with correct arguments to check volumes of generated thigh muscle segmentation masks.

E.g.

```bash
# MSTHIGH_05/FU_left.nii.gz
## Femur seg
### BL
CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/pipeline_demo/BL_left.nii.gz --output /raid/roger/dataset/pipeline_demo/
### FU
CUDA_VISIBLE_DEVICES=2,3 python main.py --input /raid/roger/dataset/pipeline_demo/FU_left.nii.gz --output /raid/roger/dataset/pipeline_demo/ --followup --baseline_femur /raid/roger/dataset/pipeline_demo/BL_left_femur_mask.nii.gz


PARAMS=(
    --save_path /raid/roger/dataset/Femur/pipeline_stats
    --mode volume
    --subject MSTHIGH_05_BL_left
    --nifti_1 /raid/roger/dataset/pipeline_demo/BL_left_thigh_mask.nii.gz
)
python stats.py ${PARAMS[@]}

PARAMS=(
    --save_path /raid/roger/dataset/Femur/pipeline_stats
    --mode volume
    --subject MSTHIGH_05_FU_left
    --nifti_1 /raid/roger/dataset/pipeline_demo/FU_left_thigh_mask.nii.gz
)
python stats.py ${PARAMS[@]}
```

**Configrable parameters for main.py:**

- --input: path to input thigh nifti, can be baseline/followup
- --output: path to store the output
- --followup: indicates if an input is a followup
- --baseline_femur: path to baseline femur nifti for registration
- --femur_weights: path to the weights for pre-trained femur segmentation model
- --thigh_weights: path to the weights for pre-trained thigh muscle segmentation model

**Configrable parameters for stats.py:**

- --save_path: path to save stats as csv
- --mode: path to save stats as csv
- --nifti_1: path to FU (after registration with ground truth) nifti / path to thigh mask nifiti to test volumenetwork training
- --subject_name: subject name used for output files naming
- ----niftis: path to FU (after registration with predicted femur) nifti files to calculate ssim

**Notes**:

Sample running scripts can be found in *stats.sh*

## Other network variants

### Two seperate networks with triplet loss (Amer)

To run the triplet version for thigh muscle model traing, use train_amer.py and test_amer.py instead.

**Configrable parameters for train_amer.py:**

- --config: Configuration file to use
- --dice_weight: weight of dice loss
- --triplet_mode: 'yes' or 'no. Set as 'yes' to turn on triplet network training.
- --triplet_weight: weight of triplet loss
