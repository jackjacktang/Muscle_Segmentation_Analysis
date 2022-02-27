import torch
# import visdom
import argparse
import yaml
import cv2
import nibabel as nib
from PIL import Image
import csv
import os

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore, averageMeter
from scipy.ndimage.interpolation import zoom
from torch.utils import data
import timeit
import time

from ptsemseg.utils import *
from ptsemseg.functions import *
from ptsemseg.processing.preprocessing import *
from ptsemseg.processing.postprocessing import largest_connected_region


def log(s):
    if DEBUG:
        print(s)

def test(cfg, cp_path=None):
    PATCH_SIZE = cfg['data']['patch_size']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_path = cfg['data']['path']
    # patch_size = [256,256]

    log('n_classes is: {}'.format(cfg['training']['n_classes']))
    model = get_model(cfg['model'], cfg['training']['n_classes'])
    cp_path = cfg['test']['cp_path']
    state = convert_state_dict(torch.load(cp_path)["model_state"])
    if device.type == 'cuda': # run on gpu
        model.load_state_dict(state)
    else: # run on cpu
        model.load_state_dict(torch.load(cfg['test']['cp_path'], map_location=torch.device('cpu')), strict=False) 

    model.eval()
    model.to(device)

    create_folder(data_path + '/test/', cfg['model']['arch'])
    print('model: ', cfg['model']['arch'])

    # Femur preduction and save as nifti
    if cfg['data']['dataset'] == 'amer':
        # test_subjects = ['MSTHIGH_06', 'MSTHIGH_07']
        test_subjects = [f"MSTHIGH_{i:02d}" for i in range(3, 16) if i != 8 and i != 13]
        skip_files = []

        start = time.process_time()
        test_paths = {name:(cfg['test']['nifti']['nifti_in']+name) for name in test_subjects}
        out_path = cfg['test']['nifti']['nifti_out']
        create_folder(out_path)
        for (subject, test_path) in test_paths.items():
            for t in os.listdir(test_path):
                if (subject, t) in skip_files:
                    continue
                if 'femur' in t or 'mask' in t: # skip label files
                    continue
                running_metrics_val = runningScore(2)
                roi_img = nib.load(os.path.join(test_path, t))
                roi_dat = roi_img.get_fdata()
                pred_roi = np.zeros(roi_dat.shape)
                roi_aff = roi_img.affine
                roi_hdr = roi_img.header

                slice_no = roi_dat.shape[2]
                for s in range(slice_no):
                    roi_slice = roi_dat[:, :, s]
                    if len(np.unique(roi_slice)) <= 1:
                        continue
                    padded_slice, pad_info = pad_slice(roi_slice)
                    padded_slice = norm_slice(padded_slice)
                    padded_slice = padded_slice.astype(np.float64)
                    padded_slice = np.expand_dims(padded_slice, 0)
                    padded_slice = np.expand_dims(padded_slice, 0)

                    padded_slice = torch.from_numpy(padded_slice).float()
                    padded_slice = padded_slice.to(device)

                    outputs = model(padded_slice)
                    pred_slice = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

                    result_slice = unpad_slice(pred_slice, pad_info)
                    pred_roi[:, :, s] = result_slice

                # pred_roi = largest_connected_region(pred_roi) # largest connection region processing
                pred_roi = pred_roi.astype(np.int64)

                # Run metrics
                surfix = 'mask'
                lbl_img = nib.load(os.path.join(test_path, t).rstrip('.nii.gz')+ '_' + surfix + '.nii.gz')
                lbl_dat = lbl_img.get_fdata()
                for s in range(slice_no):
                    pred_slice = pred_roi[:, :, s]
                    lbl_slice = lbl_dat[:, :, s]
                    # print(np.unique(lbl_slice), np.unique(result_slice))
                    running_metrics_val.update(lbl_slice, pred_slice)
                
                pred_nii = nib.Nifti1Image(pred_roi, roi_aff, header=roi_hdr)
                save_as = os.path.join(out_path, subject + '_' + t.rstrip('.nii.gz') + '_' + cfg['model']['arch'] + '.nii.gz')
                nib.save(pred_nii, save_as)
                print("Saved", save_as)
            
                score = running_metrics_val.get_scores()
                print(score)
                scores_to_csv(score[0], f'{subject}/{t}')
        
        end = time.process_time()
        print(f'Total time: {end-start} seconds.')
        return

def scores_to_csv(score_dict, model_name):
    fields = ['Subject & Case'] + list(score_dict.keys())
    dw = { model_name: score_dict}

    def mergedict(a,b):
        a.update(b)
        return a

    create_folder(cfg['test']['csv_path'])
    path = f"{cfg['test']['csv_path']}/test_score_stats.csv"
    file_exists = os.path.isfile(path) 
    
    with open(path, "a") as f:
        w = csv.DictWriter(f, fields)
        if not file_exists:
            w.writeheader()
        for k,d in sorted(dw.items()):
            w.writerow(mergedict({'Subject & Case': k},d))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/femur.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        '--debug',
        action='store_true', 
        help='print debug messages'
    )

    args = parser.parse_args()

    DEBUG = args.debug

    with open(args.config) as fp:
        cfg = yaml.safe_load(fp)
    
    test(cfg)
