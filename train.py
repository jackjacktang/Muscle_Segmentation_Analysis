import os
import yaml
import torch
import argparse
import torch.nn as nn
from tensorboardX import SummaryWriter
import shutil
import random
import time

from torch.utils import data

from ptsemseg.loader import get_loader
from ptsemseg.models import get_model
from ptsemseg.metrics import runningScore, averageMeter
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from ptsemseg.schedulers import get_scheduler
from ptsemseg.optimizers import get_optimizer
from ptsemseg.utils import get_logger
from ptsemseg.utils import convert_state_dict
from ptsemseg.functions import *
from ptsemseg.processing.preprocessing import *

def log(s):
    if DEBUG:
        print(s)

def display(string):
    print(string)
    logger.info(string)

def data_split(root_path, subject_names, ratio="8:2", dataset_name="mask"):
    # Assume train/test splits only
    train_ratio, test_ratio = map(lambda x: int(x)/10, ratio.split(':'))
    slice_counter = {subject_name:0 for subject_name in subject_names}

    train_save_path_img = root_path+"/train/img/"
    train_save_path_lbl = root_path+"/train/lbl/"
    test_save_path_img = root_path+"/test/img/"
    test_save_path_lbl = root_path+"/test/lbl/"

    create_folder(train_save_path_img)
    create_folder(train_save_path_lbl)
    create_folder(test_save_path_img)
    create_folder(test_save_path_lbl)

    skip_files = []
    
    for subject_name in subject_names:
        nitfi_file_names = os.listdir(f'{root_path}/{subject_name}/')
        # print("nitfi_file_names", nitfi_file_names)
        for file_name in nitfi_file_names:
            if (subject_name, file_name) in skip_files:
                continue
            if ".nii.gz" not in file_name:
                continue
            if dataset_name not in file_name:
                slice_counter[subject_name] += load_nifti_to_dat(f'{root_path}/{subject_name}/{file_name}', train_save_path_img, subject_name)
            else:
                load_nifti_to_dat(f'{root_path}/{subject_name}/{file_name}', train_save_path_lbl, subject_name, is_lbl=True)

    total_slices = sum(slice_counter.values())
    test_subjects = find_subjects_to_test(slice_counter, total_slices, test_ratio)
    print(f"Test subjects are: {test_subjects}. Please use them during testing.") # by default, they're 06 and 07

    # Move all relevant subject slices to another folder
    files = []
    for s in test_subjects:
        files.extend(glob.glob(f'{train_save_path_img}/{s}*'))
    print(f"{len(files)} number of files to remove from train dataset.")
    for f in files:
        img_path_move = f
        img_path_to = img_path_move.replace('train', 'test')
        shutil.move(img_path_move, img_path_to) # remove files in train dataset
        print('Moving', img_path_move, 'to', img_path_to)
        tmp_idx = img_path_move.rindex('_')
        lbl_path_move = img_path_move.replace('img', 'lbl')[:tmp_idx] + '_' + dataset_name + img_path_move[tmp_idx:]
        lbl_path_to = lbl_path_move.replace('train', 'test')
        if os.path.isfile(lbl_path_move):
            shutil.move(lbl_path_move, lbl_path_to)
        print('Moving', lbl_path_move, 'to', lbl_path_to)

    print("Slice Counter:", slice_counter)
    print(f"Test subjects are: {test_subjects}. Please use them during testing.") # by default, they're 06 and 07

# Find which subjects should move to test dataset
def find_subjects_to_test(slice_counter, total_slices, test_ratio):
    approx_slices_to_test = int(total_slices*test_ratio)
    sorted_counter = sorted(slice_counter.items(), key=lambda x: x[1], reverse=True)
    result = []
    for (subject, count) in sorted_counter:
        if approx_slices_to_test <= 0:
            break
        approx_slices_to_test -= count
        result.append(subject)
    return result

    
def load_nifti_to_dat(file_path, save_path, subject_name, is_test=False, is_lbl=False):
    head, tail = os.path.split(file_path)
    
    roi_img = nib.load(file_path)
    roi_dat = roi_img.get_fdata()
    roi_aff = roi_img.affine
    roi_hdr = roi_img.header

    result_roi = np.zeros(roi_dat.shape)
    slice_no = roi_dat.shape[2]
    count = 0
    for s in range(slice_no):
        roi_slice = roi_dat[:, :, s]
        if len(np.unique(roi_slice)) <= 1:
            continue
        padded_slice, pad_info = pad_slice(roi_slice)

        if not is_lbl:
            padded_slice = norm_slice(padded_slice) # normalization

        padded_slice = padded_slice.astype(np.float64)
        padded_slice = np.expand_dims(padded_slice, 0)

        save_file_path= f'{save_path}{subject_name}_{tail.rstrip(".nii.gz")}_{s:03d}.dat'
        if np.isnan(padded_slice).any():
            print(f"NaN in {save_file_path}")
            raise RuntimeError()

        padded_slice = torch.from_numpy(padded_slice).float()

        torch.save(padded_slice, save_file_path)
        count+=1

        if s == 0 or s == slice_no - 1 or s % 100 == 0:
            print(f"Saved {save_file_path}")
    return count

def train(cfg, writer, logger):
    # Setup dataset split before setting up the seed for random
    if cfg['data']['dataset'] == 'thigh':
        # data_split_info = init_data_split(cfg['data']['path'], cfg['data'].get('split_ratio', 0), cfg['data'].get('compound', False))  # fly jenelia dataset'
        subject_names = [f"MSTHIGH_{i:02d}" for i in range(3, 16) if i != 8 and i != 13]
    elif cfg['data']['dataset'] == 'femur':
        subject_names = [f"MSTHIGH_{i:02d}" for i in range(3, 16) if i != 8 and i != 13]
        # femur_data_split(cfg['data']['path'], subject_names, ratio=cfg['data']['split_ratio'])

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Cross Entropy Weight
    weight = None
    if cfg['training']['loss'].get('name', None) != 'regression_l1':
        if 'cross_entropy_ratio' in cfg['training']:
            weight = prep_class_val_weights(cfg['training']['cross_entropy_ratio'])        
    
    log('Using loss : {}'.format(cfg['training']['loss']['name']))

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None) # if no augmentation => default None
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    t_loader = data_loader(
        data_path,
        split=cfg['data']['train_split'],
        augmentations=data_aug,
        n_classes=cfg['training'].get('n_classes', 2))

    # # If using validation, uncomment this block
    # v_loader = data_loader(
    #     data_path,
    #     split=cfg['data']['val_split'],
    #     data_split_info=data_split_info,
    #     n_classe=cfg['training'].get('n_classes', 1))

    n_classes = t_loader.n_classes
    log('n_classes is: {}'.format(n_classes))
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'],
                                  num_workers=cfg['training']['n_workers'],
                                  shuffle=False)

    print('trainloader len: ', len(trainloader))
    # Setup Metrics
    running_metrics_val = runningScore(n_classes) # a confusion matrix is created


    # Setup Model
    model = get_model(cfg['model'], n_classes)

    model = model.to(device)

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # print(range(torch.cuda.device_count()))
    count_parameters(model, verbose=False)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg['training']['optimizer'].items()
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))
    softmax_function = nn.Softmax(dim=1)

    # model_count = 0
    min_loss = None
    start_iter = 0
    if cfg['training']['resume'] is not None:
        log('resume saved model')
        if os.path.isfile(cfg['training']['resume']):
            display(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            min_loss = checkpoint["min_loss"]
            display(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            display("No checkpoint found at '{}'".format(cfg['training']['resume']))
            log('no saved model found')

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    # if cfg['training']['loss']['name'] == 'dice':
    #     loss_fn = dice_loss()


    i_train_iter = start_iter

    display('Training from {}th iteration\n'.format(i_train_iter))
    while i_train_iter < cfg['training']['train_iters']:
        i_batch_idx = 0
        train_iter_start_time = time.time()
        averageLoss = 0

        # training
        for (images, labels) in trainloader:
            start_ts = time.time()
            model.train()

            # images = images.cuda()
            # labels = labels.cuda()

            optimizer.zero_grad()

            # anchor_imgs, pos_imgs, neg_imgs = random_crop_triplet(images, labels)

            images = images.to(device)
            labels = labels.to(device)
	
            # outputs = model(images) 
            
            # print(outputs.shape, labels.shape)

            if cfg['training']['loss']['name'] in ['dice']:
                outputs = model(images)
                # print(outputs.unique)
                loss = loss_fn(outputs, labels)
                # print('loss match: ', loss, loss.item())
                averageLoss += loss.item()
            else:
                hard_loss = loss_fn(input=outputs, target=labels, weight=weight,
                                size_average=cfg['training']['loss']['size_average'])

                loss = hard_loss

                averageLoss += loss
            loss.backward()
            # print('{} optim: {}'.format(i, optimizer.param_groups[0]['lr']))
            optimizer.step()
            # print('{} scheduler: {}'.format(i, scheduler.get_lr()[0]))
            scheduler.step()

            time_meter.update(time.time() - start_ts)
            print_per_batch_check = True if cfg['training']['print_interval_per_batch'] else i_batch_idx+1 == len(trainloader)
            if (i_train_iter + 1) % cfg['training']['print_interval'] == 0 and print_per_batch_check:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i_train_iter + 1,
                                           cfg['training']['train_iters'],
                                           loss.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                display(print_str)
                writer.add_scalar('loss/train_loss', loss.item(), i_train_iter + 1)
                time_meter.reset()
            i_batch_idx += 1
        time_for_one_iteration = time.time() - train_iter_start_time

        display('EntireTime for {}th training iteration: {}  EntireTime/Image: {}'.format(i_train_iter+1, time_converter(time_for_one_iteration),
                                                                                          time_converter(time_for_one_iteration/(len(trainloader)*cfg['training']['batch_size']))))
        averageLoss /= (len(trainloader)*cfg['training']['batch_size'])
        print(averageLoss)
        # validation
        validation_check = (i_train_iter + 1) % cfg['training']['val_interval'] == 0 or \
                           (i_train_iter + 1) == cfg['training']['train_iters']
        if not validation_check:
            print('no validation check')
        else:

            '''
            This IF-CHECK is used to update the best model
            '''
            log('Validation: average loss for current iteration is: {}'.format(averageLoss))
            if min_loss is None:
                min_loss = averageLoss

            if averageLoss <= min_loss:
                min_loss = averageLoss
                state = {
                    "epoch": i_train_iter + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "min_loss": min_loss
                }

                # if cfg['training']['cp_save_path'] is None:
                save_path = os.path.join(writer.file_writer.get_logdir(),
                                         "{}_{}_model_best.pkl".format(
                                             cfg['model']['arch'],
                                             cfg['data']['dataset']))
                # else:
                #     save_path = os.path.join(cfg['training']['cp_save_path'],  writer.file_writer.get_logdir(),
                #                              "{}_{}_model_best.pkl".format(
                #                                  cfg['model']['arch'],
                #                                  cfg['data']['dataset']))
                print('save_path is: ' + save_path)

                torch.save(state, save_path)

            # model_count += 1

        i_train_iter += 1



if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s.yml",
        help="Configuration file to use"
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

    subject_names = [f"MSTHIGH_{i:02d}" for i in range(3, 16) if i != 8 and i != 13]
    # data_split(cfg['data']['path'], subject_names, ratio=cfg['data']['split_ratio'], dataset_name=('mask' if cfg['data']['dataset'] == 'thigh' else 'femur'))

    run_id = random.randint(1, 100000)
    logdir = os.path.join('../runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger)

    time_keeper(logger, start)