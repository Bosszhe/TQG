from numpy import record
import torch.nn as nn

from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.att_bev_backbone import AttBEVBackbone

import torch
from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
import numpy as np
from torch.nn import functional as F

    
class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class PointPillarIntermediateFFNetFreeze(nn.Module):
    def __init__(self, args):
        super(PointPillarIntermediateFFNetFreeze, self).__init__()
        print('PointPillarIntermediateFFNetFreeze.__init__')

        # PIllar VFE
        # self.pillar_vfe = PillarVFE(args['pillar_vfe'],
        #                             num_point_features=4,
        #                             voxel_size=args['voxel_size'],
        #                             point_cloud_range=args['lidar_range'])
        # self.scatter = PointPillarScatter(args['point_pillar_scatter'])
        # self.backbone = AttBEVBackbone(args['base_bev_backbone'], 64)

        # self.cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
        #                           kernel_size=1)
        # self.reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
        #                           kernel_size=1)

        self.encoder = double_conv(64,64)
        
        self.async_mode = args['async_mode']
        self.mse_loss = nn.MSELoss()
        self.simi_loss_type = args['simi_loss_type']

        self.cur_pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.cur_scatter = PointPillarScatter(args['point_pillar_scatter'])
        self.cur_backbone = AttBEVBackbone(args['base_bev_backbone'], 64)
        self.cur_cls_head = nn.Conv2d(128 * 3, args['anchor_number'],
                                  kernel_size=1)
        self.cur_reg_head = nn.Conv2d(128 * 3, 7 * args['anchor_num'],
                                  kernel_size=1)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        filename = '/home/wangz/wangzhe21/V2XPerception/OpenCOOD/opencood/logs/duibi/OPV2V_point_pillar_Attentive_Fusion_sync_gpu2_2023_11_16_19_19_52/net_epoch20.pth'
        state_dict = torch.load(filename, map_location = device)

        for module_name in ['cur_pillar_vfe','cur_backbone','cur_cls_head','cur_reg_head']:
        # for module_name in ['cur_pillar_vfe']:
            model_dict = getattr(self, module_name).state_dict()
            model_dict_new = {k.replace(module_name[4:]+'.', ''): v for k, v in state_dict.items() if k.startswith(module_name[4:])}
            model_dict.update(model_dict_new)
            getattr(self, module_name).load_state_dict(model_dict_new)

        self._freeze_cur_model()


    def _freeze_cur_model(self):

        # for module_name in ['cur_pillar_vfe','cur_backbone','cur_cls_head','cur_reg_head']:
        for module_name in ['cur_pillar_vfe']:
            getattr(self, module_name).eval()
            for param in getattr(self, module_name).parameters():
                param.requires_grad = False
    
    def forward(self, data_dict_list):

        cur_data_dict = data_dict_list[0]['ego']
        past_data_dict = data_dict_list[1]['ego']
        
        cur_ego_flag = []
        for i in cur_data_dict['ego_flag']:
            cur_ego_flag = cur_ego_flag +i
        
        past_ego_flag = []
        for i in past_data_dict['ego_flag']:
            past_ego_flag = past_ego_flag +i

        # past_not_ego_flag = [not i for i in past_ego_flag]
        cur_voxel_features = cur_data_dict['processed_lidar']['voxel_features']
        cur_voxel_coords = cur_data_dict['processed_lidar']['voxel_coords']
        cur_voxel_num_points = cur_data_dict['processed_lidar']['voxel_num_points']
        cur_record_len = cur_data_dict['record_len']

        past_voxel_features = past_data_dict['processed_lidar']['voxel_features']
        past_voxel_coords = past_data_dict['processed_lidar']['voxel_coords']
        past_voxel_num_points = past_data_dict['processed_lidar']['voxel_num_points']
        past_record_len = past_data_dict['record_len']

        assert torch.equal(cur_record_len,past_record_len)
        # if not torch.equal(cur_record_len,past_record_len):
        #     from IPython import embed
        #     embed()
        cur_car_nums = cur_voxel_coords[:, 0].max().int().item() + 1
        past_car_nums = past_voxel_coords[:, 0].max().int().item() + 1
        batch_size = cur_record_len.shape[0]


        voxel_features_list = list()
        voxel_coords_list = list()
        voxel_num_points_list = list()

        for car_id in range(cur_car_nums):
            batch_mask = cur_voxel_coords[:, 0] == car_id
            voxel_features_list.append(cur_voxel_features[batch_mask, :,:])
            voxel_coords_list.append(cur_voxel_coords[batch_mask, :])
            voxel_num_points_list.append(cur_voxel_num_points[batch_mask])

        past_voxel_features_list = list()
        past_voxel_coords_list = list()
        past_voxel_num_points_list = list()
        for car_id in range(past_car_nums):
            past_batch_mask = past_voxel_coords[:, 0] == car_id
            past_voxel_features_list.append(past_voxel_features[past_batch_mask, :,:])
            past_voxel_coords_list.append(past_voxel_coords[past_batch_mask, :])
            past_voxel_num_points_list.append(past_voxel_num_points[past_batch_mask])

        cur_cum_sum_len = torch.cumsum(cur_record_len, dim=0)
        cur_ego_index = [0] + list((cur_cum_sum_len[:-1]).cpu().numpy())

        past_cum_sum_len = torch.cumsum(past_record_len, dim=0)
        past_ego_index = [0] + list((past_cum_sum_len[:-1]).cpu().numpy())


        assert all([cur_ego_flag[i] for i in cur_ego_index])

        for i in range(batch_size):

            # cur_point_num = voxel_coords_list[cur_ego_index[i]].shape[0]
            

            # voxel_coords_list[cur_ego_index[i]][:,0] = torch.from_numpy(np.ones([cur_point_num])*past_ego_index[i])
            past_voxel_features_list[past_ego_index[i]] = voxel_features_list[cur_ego_index[i]]
            past_voxel_coords_list[past_ego_index[i]] = voxel_coords_list[cur_ego_index[i]]
            past_voxel_num_points_list[past_ego_index[i]] = voxel_num_points_list[cur_ego_index[i]]

        
        async_voxel_features = torch.cat(past_voxel_features_list,dim=0)
        async_voxel_coords = torch.cat(past_voxel_coords_list,dim=0)
        async_voxel_num_points = torch.cat(past_voxel_num_points_list,dim=0)

        # from IPython import embed
        # embed(header='forward')


        cur_batch_dict = {'voxel_features': cur_voxel_features,
            'voxel_coords': cur_voxel_coords,
            'voxel_num_points': cur_voxel_num_points,
            'record_len': cur_record_len}
        
        if self.async_mode:
            batch_dict = {'voxel_features': async_voxel_features,
                    'voxel_coords': async_voxel_coords,
                    'voxel_num_points': async_voxel_num_points,
                    'record_len': past_record_len}
        
        else:
            batch_dict = cur_batch_dict
        
        # batch_dict = self.pillar_vfe(batch_dict)
        # batch_dict = self.scatter(batch_dict)

        batch_dict = self.cur_pillar_vfe(batch_dict)
        batch_dict = self.cur_scatter(batch_dict)


        cur_batch_dict = self.cur_pillar_vfe(cur_batch_dict)
        cur_batch_dict = self.cur_scatter(cur_batch_dict)

        past_bev = batch_dict['spatial_features']
        cur_bev = cur_batch_dict['spatial_features']
        
        past_bev_trans = self.encoder(past_bev)
        batch_dict['spatial_features'] = past_bev_trans

        if self.simi_loss_type == 'similarity_loss' :
            similarity = torch.cosine_similarity(torch.flatten(past_bev_trans, start_dim=1, end_dim=3),
                                             torch.flatten(cur_bev, start_dim=1, end_dim=3), dim=1)
        
            label = torch.ones(past_record_len.sum(), requires_grad=False).cuda(device=past_bev.device)

            simi_loss = self.mse_loss(similarity, label)
        elif self.simi_loss_type == 'mse_loss':
            simi_loss = self.mse_loss(past_bev_trans, cur_bev)



        batch_dict = self.cur_backbone(batch_dict)
        # cur_batch_dict = self.cur_backbone(cur_batch_dict)



        spatial_features_2d = batch_dict['spatial_features_2d']
                
        psm_single = self.cur_cls_head(spatial_features_2d)
        rm_single = self.cur_reg_head(spatial_features_2d)

        output_dict = {'psm': psm_single,
                    'rm': rm_single,
                    'similarity_loss': simi_loss
                    }

        return output_dict
    
    # def forward(self, data_dict_list):
    #     # from IPython import embed
    #     # embed()
    #     # batch_dict_list = [] 
    #     feature_list = []  
    #     feature_2d_list = []  
    #     # matrix_list = []
    #     # regroup_feature_list = []  
    #     # regroup_feature_list_large = []  
    #     for origin_data in data_dict_list:  
    #         data_dict = origin_data['ego']
    #         voxel_features = data_dict['processed_lidar']['voxel_features']
    #         voxel_coords = data_dict['processed_lidar']['voxel_coords']
    #         voxel_num_points = data_dict['processed_lidar']['voxel_num_points']
    #         record_len = data_dict['record_len']

    #         batch_dict = {'voxel_features': voxel_features,
    #                     'voxel_coords': voxel_coords,
    #                     'voxel_num_points': voxel_num_points,
    #                     'record_len': record_len}
    #         batch_dict = self.pillar_vfe(batch_dict)
    #         batch_dict = self.scatter(batch_dict)
    #         batch_dict = self.backbone(batch_dict)
    #         spatial_features_2d = batch_dict['spatial_features_2d']
            
                
    #         # batch_dict_list.append(batch_dict)
    #         spatial_features = batch_dict['spatial_features']
    #         feature_list.append(spatial_features)
    #         feature_2d_list.append(spatial_features_2d)
     
        
    #     spatial_features = feature_list[0]
    #     spatial_features_2d = feature_2d_list[0]
    #     # batch_dict = batch_dict_list[0]
    #     # record_len = batch_dict['record_len']
        
        
    #     psm_single = self.cls_head(spatial_features_2d)
    #     rm_single = self.reg_head(spatial_features_2d)



    #     output_dict = {'psm': psm_single,
    #                 'rm': rm_single
    #                 }

    #     return output_dict