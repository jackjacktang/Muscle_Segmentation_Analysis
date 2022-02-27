import os
import yaml
import torch
import argparse
import torch.nn as nn
import random
import time
import subprocess

from ptsemseg.loader import get_loader
from ptsemseg.models import get_model
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.utils import get_logger, convert_state_dict
from ptsemseg.functions import time_converter
from ptsemseg.processing.postprocessing import largest_connected_region
from ptsemseg.processing.preprocessing import *

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = args.input
    print('Input path is', data_path)

    case_name = data_path[data_path.rindex('/')+1:].rstrip('.nii.gz')
    create_folder(args.output)
    output_path = args.output + case_name
    print('output_path prefix is', output_path)

    # Femur preduction and save as nifti
    femur_seg(device, args, case_name)

    if args.followup: # need registration
        start = time.time()
        baseline_femur = args.baseline_femur
        print('########             Register to baseline...             ########')
        if (os.path.isfile(output_path + '_ref2bl.nii.gz') == False):
            subprocess.call(['flirt', 
                '-in', 
                output_path + '_femur_mask.nii.gz', 
                # '/raid/roger/dataset/Femur/MSTHIGH_07/FU_left_femur.nii.gz',
                '-ref', 
                baseline_femur,
                '-out', 
                output_path + '_ref2bl_bone.nii.gz', 
                '-omat',
                output_path + '_ref2bl.mat',
                '-dof', 
                '6'])
            print(f'Saved {output_path}_ref2bl_bone.nii.gz and {output_path}_ref2bl.mat')
            subprocess.call(['flirt', 
                '-in', 
                data_path,
                '-applyxfm', 
                '-init', 
                output_path + '_ref2bl.mat',
                '-ref',
                baseline_femur,
                '-out',
                output_path + '_ref2bl.nii.gz'])
            print(f'Saved {output_path}_ref2bl.nii.gz')
        end = time.time()
        print(f'Registration total time: {time_converter(end-start)} seconds.')
        
    # Thigh muscle preduction and save as nifti
    thigh_muscle_seg(device, args, case_name, f'{output_path}_ref2bl.nii.gz' if args.followup else data_path)

def femur_seg(device, args, case_name):

    model_path = args.femur_weights
    model_file = model_path[model_path.rindex('/')+1:]
    model_name = model_file[:model_file.index('_')]
    print(model_name)
    model_dict = {
        'arch': model_name,
        'n_channels': 1
        }
    model = get_model(model_dict, 2)

    data_path = args.input
    output_path = args.output

    cp_path = args.femur_weights
    state = convert_state_dict(torch.load(cp_path)["model_state"])
    if device.type == 'cuda': # run on gpu
        model.load_state_dict(state)
    else: # run on cpu
        model.load_state_dict(torch.load(cp_path, map_location=torch.device('cpu')), strict=False) 

    model.eval()
    model.to(device)

    start = time.time()

    roi_img = nib.load(data_path)
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

    pred_roi = largest_connected_region(pred_roi) # largest connection region processing

    pred_nii = nib.Nifti1Image(pred_roi, roi_aff, header=roi_hdr)
    save_as = output_path + f'{case_name}_femur_mask.nii.gz'
    nib.save(pred_nii, save_as)
    print("Saved", save_as)
    
    end = time.time()
    print(f'Total time: {time_converter(end-start)} seconds.')
    return

def thigh_muscle_seg(device, args, case_name, data_path):
    start = time.time()

    model_path = args.thigh_weights
    model_file = model_path[model_path.rindex('/')+1:]
    model_name = model_file[:model_file.index('_')]
    model_dict = {
        'arch': model_name,
        'n_channels': 3
        }
    model = get_model(model_dict, 2)

    output_path = args.output

    cp_path = args.thigh_weights
    state = convert_state_dict(torch.load(cp_path)["model_state"])
    if device.type == 'cuda': # run on gpu
        model.load_state_dict(state)
    else: # run on cpu
        model.load_state_dict(torch.load(cp_path, map_location=torch.device('cpu')), strict=False) 

    model.eval()
    model.to(device)

    start = time.time()

    roi_img = nib.load(data_path)
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
        padded_slice = padded_slice.astype(np.float64)
        padded_slice /= 255.0
        padded_slice = np.expand_dims(padded_slice, 0)
        # padded_slice = padded_slice.transpose(2, 0, 1)
        temp = padded_slice.copy()
        padded_slice = np.concatenate((padded_slice, temp), axis=0)
        padded_slice = np.concatenate((padded_slice, temp), axis=0)
        padded_slice = np.expand_dims(padded_slice, 0)
        padded_slice = torch.from_numpy(padded_slice).float()
        # print('padded_slice', padded_slice.shape)
        padded_slice = padded_slice.to(device)

        outputs = model(padded_slice)
        pred_slice = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

        result_slice = unpad_slice(pred_slice, pad_info)
        pred_roi[:, :, s] = result_slice

    pred_roi = largest_connected_region(pred_roi) # largest connection region processing

    pred_nii = nib.Nifti1Image(pred_roi, roi_aff, header=roi_hdr)
    save_as = output_path + f'{case_name}_thigh_mask.nii.gz'
    nib.save(pred_nii, save_as)
    print("Saved", save_as)
    
    end = time.time()
    print(f'Total time: {time_converter(end-start)} seconds.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/MSTHIGH_07/FU_left.nii.gz',
        help='path to input thigh nifti, can be baseline/followup'
    )

    parser.add_argument(
        '--output',
        nargs='?',
        type=str,
        default="/raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/",
        help='path to store the output'
    )

    parser.add_argument(
        '--followup',
        action='store_true', 
        dest='followup',
        help='indicates if an input is a followup'
    )
    parser.set_defaults(followup=False)

    parser.add_argument(
        '--baseline_femur',
        nargs='?',
        type=str,
        default='/raid/roger/dataset/Femur/pipeline_nii_test_out/MSTHIGH_07/use-pred-reg/BL_left_femur_mask.nii.gz',
        # default='/raid/roger/dataset/Femur/MSTHIGH_07/BL_left_femur.nii.gz',
        help='path to baseline femur nifti for registration'
    )

    parser.add_argument(
        '--femur_weights',
        nargs='?',
        type=str,
        default="../runs/femur/80658/resunet_femur_model_best.pkl",
        help='path to the weights for femur segmentation model'
    )

    parser.add_argument(
        '--thigh_weights',
        nargs='?',
        type=str,
        default="../runs/resunet/23666/resunet_thigh_model_best.pkl",
        # default="../runs/resunet/24909/resunet_thigh_model_best.pkl",
        help='path to the weights for thigh muscle segmentation model'
    )

    args = parser.parse_args()

    main(args)