# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
# import tqdm
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler, Subset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
# from opencood.tools import train_utils
from opencood.utils import eval_utils

import numpy as np
import random

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    opt = parser.parse_args()
    return opt

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    opt = train_parser()

    # 设置随机数种子
    setup_seed(42)

    assert opt.fusion_method in ['late', 'early', 'intermediate']
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    multi_gpu_utils.init_distributed_mode(opt)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)
    # subset = Subset(opencood_train_dataset, range(4004,4400))

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
        infer_data_loader = DataLoader(opencood_validate_dataset,
                    sampler=sampler_val,
                    num_workers=16,
                    collate_fn=opencood_train_dataset.collate_batch_test,
                    drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params']['batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=False,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=False,
                                pin_memory=False,
                                drop_last=True)
        infer_data_loader = DataLoader(opencood_validate_dataset,
                            batch_size=1,
                            num_workers=16,
                            collate_fn=opencood_train_dataset.collate_batch_test,
                            shuffle=False,
                            pin_memory=False,
                            drop_last=False)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split = hypes['validate_dir'].split('/')[-1]
    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path,
                                                         model)

    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True)
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):

            # print(i)
            # if i > 5 :
            #     break
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(ouput_dict,
                                       batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])


            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes['train_params']['save_freq'] == 0:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if epoch % hypes['train_params']['eval_freq'] == 0:
            valid_ave_loss = []
            # Create the dictionary for evaluation.
            # also store the confidence score for each prediction
            result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
            with torch.no_grad():
                for j, batch_data_list in tqdm(enumerate(val_loader)):
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    ouput_dict = model(batch_data['ego'])

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    valid_ave_loss.append(final_loss.item())
            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

            with torch.no_grad():
                for j, batch_data in tqdm(enumerate(infer_data_loader)):
                    model.eval()
                    batch_data = train_utils.to_device(batch_data, device)
                    if opt.fusion_method == 'late':

                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_late_fusion(batch_data,
                                                                model,
                                                                opencood_validate_dataset)
                    elif opt.fusion_method == 'early':

                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_early_fusion(batch_data,
                                                                model,
                                                                opencood_validate_dataset)
                    elif opt.fusion_method == 'intermediate':
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            inference_utils.inference_intermediate_fusion(batch_data,
                                                                        model,
                                                                        opencood_validate_dataset)
                    else:
                        raise NotImplementedError('Only early, late and intermediate'
                                                'fusion is supported.')

                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.3)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.5)
                    eval_utils.caluclate_tp_fp(pred_box_tensor,
                                            pred_score,
                                            gt_box_tensor,
                                            result_stat,
                                            0.7)
            ap_30,ap_50,ap_70 = eval_utils.eval_final_results(result_stat,
                                opt.model_dir,
                                opt.global_sort_detections)

            with open(os.path.join(saved_path, 'result.txt'), 'a+') as f:
                msg = 'Split: {} | Epoch: {} | AP @0.3: {:.04f} | AP @0.5: {:.04f} | AP @0.7: {:.04f} \n'.format(split,epoch, ap_30, ap_50, ap_70)
                f.write(msg)
                print(msg)


            writer.add_scalar('AP_30', ap_30, epoch)
            writer.add_scalar('AP_50', ap_50, epoch)
            writer.add_scalar('AP_70', ap_70, epoch)

    print('Training Finished, checkpoints saved to %s' % saved_path)


if __name__ == '__main__':
    main()
