# -*- coding: utf-8 -*-
# Author: OpenPCDet, Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class FlowLoss(nn.Module):
    def __init__(self, args):
        super(FlowLoss, self).__init__()


        self.simi_weight = args['simi_weight']
        self.loss_dict = {}

    def forward(self, output_dict, target_dict, prefix=''):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """

        simi_loss = output_dict['similarity_loss'] * self.simi_weight

        total_loss = simi_loss

        self.loss_dict.update({'total_loss{}'.format(prefix): total_loss,
                               'similarity_loss{}'.format(prefix): simi_loss})

        return total_loss




    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
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
        total_loss = self.loss_dict['total_loss']
        simi_loss = self.loss_dict['similarity_loss']
        if pbar is None:
            print("[epoch %d][%d/%d], || Loss: %.4f || Simi Loss: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss.item(), simi_loss.item()))
        else:
            pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Simi Loss: %.4f" % (
                      epoch, batch_id + 1, batch_len,
                      total_loss.item(), simi_loss.item()))

        writer.add_scalar('Total_loss', total_loss.item(),
                          epoch*batch_len + batch_id)
        writer.add_scalar('Similarity_loss', simi_loss.item(),
                          epoch*batch_len + batch_id)