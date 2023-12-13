# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone


class PointPillarIntermediateIrregular(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateIrregular, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)

    def forward(self, data_dict):



        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']
        record_frames = data_dict['past_k_time_interval']                       #(B, )
        pairwise_t_matrix = data_dict['pairwise_t_matrix']                      #(B, L, k, 4, 4)
        
        BS,_,K,_,_ = pairwise_t_matrix.shape

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        

        spatial_features = batch_dict['spatial_features']

        # from IPython import embed
        # embed(header='forward!!!')
        split_x = self.regroup(spatial_features, record_len * K)
        cur_spatial_features_list = list()
        for i in range(BS):
            cav_num = record_len[i]
            delay = 1
            cur_index = list(range(delay,cav_num*K+delay,K))
            cur_index[0] = 0

            # print(cur_index)
            # print(split_x[i].shape[0])
            # if split_x[i].shape[0] <= cur_index[-1]:
            #     from IPython import embed
            #     embed(header='<')
            cur_spatial_feature = split_x[i][cur_index,...]
            cur_spatial_features_list.append(cur_spatial_feature)

        cur_spatial_features = torch.cat(cur_spatial_features_list,dim=0)
        batch_dict['spatial_features'] = cur_spatial_features
        batch_dict = self.backbone(batch_dict)
        spatial_features_2d = batch_dict['spatial_features_2d']

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {'psm': psm,
                       'rm': rm}

        return output_dict
    
    def regroup(self, x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x