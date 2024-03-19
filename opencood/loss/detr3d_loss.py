# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DETR3DLoss(nn.Module):
    def __init__(self, args):
        super(DETR3DLoss, self).__init__()

        # self.alpha = 0.25
        # self.gamma = 2.0

        # self.cls_weight = args['cls_weight']
        # self.reg_coe = args['reg']
        # self.loss_dict = {}

    def logging(self, epoch, batch_id, batch_len, writer, loss_dict, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = loss_dict['loss']
        loss_cls = loss_dict['loss_cls']
        loss_bbox = loss_dict['loss_bbox']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || cls Loss: %.4f"
                " || bbox Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss, loss_cls, loss_bbox))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || cls Loss: %.4f"
                " || bbox Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss, loss_cls, loss_bbox))

        for key,value in loss_dict.items():
            writer.add_scalar(key, value,
                            epoch*batch_len + batch_id)
        # writer.add_scalar('cls_loss', loss_cls,
        #                   epoch*batch_len + batch_id)
        # writer.add_scalar('bbox_loss', loss_bbox,
        #                   epoch*batch_len + batch_id)