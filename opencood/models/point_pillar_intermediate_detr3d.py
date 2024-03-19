# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import torch
import torch.nn as nn


from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone
from opencood.models.futr3d.models import FUTR3D






from mmcv.ops import Voxelization
from mmdet3d.models import builder
from mmcv import Config
from torch.nn import functional as F

# class PointPillarIntermediateDETR3D(nn.Module):
#     def __init__(self, args):
#         super(PointPillarIntermediateDETR3D, self).__init__()

#         self.detr3d_model = FUTR3D()
#         cfg = Config.fromfile("/home/wangz/wangzhe21/V2XPerception/OpenCOOD/opencood/models/futr3d/configs/lidar_0075v_900q.py")

#         pts_voxel_layer = cfg.model.pts_voxel_layer
#         pts_voxel_encoder = cfg.model.pts_voxel_encoder
#         pts_middle_encoder = cfg.model.pts_middle_encoder
#         pts_backbone = cfg.model.pts_backbone
#         pts_neck = cfg.model.pts_neck

#         if pts_voxel_layer:
#             self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
#         if pts_voxel_encoder:
#             self.pts_voxel_encoder = builder.build_voxel_encoder(
#                 pts_voxel_encoder)
#         if pts_middle_encoder:
#             self.pts_middle_encoder = builder.build_middle_encoder(
#                 pts_middle_encoder)
#         if pts_backbone:
#             self.pts_backbone = builder.build_backbone(pts_backbone)
#         # if pts_fusion_layer:
#         #     self.pts_fusion_layer = builder.build_fusion_layer(
#         #         pts_fusion_layer)
#         if pts_neck is not None:
#             self.pts_neck = builder.build_neck(pts_neck)

#     def voxelize(self, points):
#         """Apply dynamic voxelization to points.

#         Args:
#             points (list[torch.Tensor]): Points of each sample.

#         Returns:
#             tuple[torch.Tensor]: Concatenated points, number of points
#                 per voxel, and coordinates.
#         """
#         voxels, coors, num_points = [], [], []
#         for res in points:
#             res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
#             voxels.append(res_voxels)
#             coors.append(res_coors)
#             num_points.append(res_num_points)
#         voxels = torch.cat(voxels, dim=0)
#         num_points = torch.cat(num_points, dim=0)
#         coors_batch = []
#         for i, coor in enumerate(coors):
#             coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
#             coors_batch.append(coor_pad)
#         coors_batch = torch.cat(coors_batch, dim=0)
#         return voxels, num_points, coors_batch


#     def extract_pts_feat(self, pts):
#         """Extract features of points."""
#         # if not self.with_pts_bbox:
#         #     return None
#         # from IPython import embed
#         # embed(header="voxelize")
#         voxels, num_points, coors = self.voxelize(pts)

#         voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
#         batch_size = coors[-1, 0] + 1
       
#         x = self.pts_middle_encoder(voxel_features, coors, batch_size)
#         x = self.pts_backbone(x)
#         x = self.pts_neck(x)
#         return x
    
#     def forward(self, data_dict, return_loss=True):

#         # from IPython import embed
#         # embed(header='xxxxxx')

#         # voxel_features = data_dict['processed_lidar']['voxel_features']
#         # voxel_coords = data_dict['processed_lidar']['voxel_coords']
#         # voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
#         # record_len = data_dict['record_len']

#         points = data_dict['origin_lidar']
#         BS = points.shape[0]
#         pts = list()
#         for i in range(BS):
#             pts.append(points[i])
#         pts_feats = self.extract_pts_feat(pts)

#         gt_bboxes_3d = data_dict['gt_bboxes_3d']
#         gt_labels_3d = data_dict['gt_labels_3d']
#         img_metas = [None for _ in range(BS)]

#         # return_loss = True
#         if return_loss:
#             return self.detr3d_model.forward_train(pts_feats,img_metas,gt_bboxes_3d,gt_labels_3d)
#         else:
#             return self.detr3d_model.forward_test(pts_feats,img_metas,gt_bboxes_3d,gt_labels_3d)



class PointPillarIntermediateDETR3D(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateDETR3D, self).__init__()

        self.detr3d_model = FUTR3D()

        cfg = Config.fromfile("/home/wangz/wangzhe21/V2XPerception/OpenCOOD/opencood/models/futr3d/configs/lidar_0075v_900q.py")

        # pts_voxel_layer = cfg.model.pts_voxel_layer
        # pts_voxel_encoder = cfg.model.pts_voxel_encoder
        # pts_middle_encoder = cfg.model.pts_middle_encoder
        pts_backbone = cfg.model.pts_backbone
        pts_neck = cfg.model.pts_neck

        if pts_backbone:
            self.pts_backbone = builder.build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        # self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
        #                           kernel_size=1)
        # self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
        #                           kernel_size=1)

    
    def forward(self, data_dict, return_loss=True):

        # from IPython import embed
        # embed(header='xxxxxx')

        voxel_features = data_dict['processed_lidar']['voxel_features']
        voxel_coords = data_dict['processed_lidar']['voxel_coords']
        voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
        record_len = data_dict['record_len']

        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points,
                      'record_len': record_len}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)
        x = batch_dict['spatial_features_2d']

        x = self.pts_backbone(x)
        pts_feats = self.pts_neck(x)

        points = data_dict['origin_lidar']
        BS = points.shape[0]
        # pts = list()
        # for i in range(BS):
        #     pts.append(points[i])
        # pts_feats = self.extract_pts_feat(pts)

        gt_bboxes_3d = data_dict['gt_bboxes_3d']
        gt_labels_3d = data_dict['gt_labels_3d']
        img_metas = [None for _ in range(BS)]

        # return_loss = True
        if return_loss:
            return self.detr3d_model.forward_train(pts_feats,img_metas,gt_bboxes_3d,gt_labels_3d)
        else:
            return self.detr3d_model.forward_test(pts_feats,img_metas,gt_bboxes_3d,gt_labels_3d)