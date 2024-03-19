import warnings

import torch
from torch.nn import functional as F

from mmcv.ops import Voxelization
from mmcv.runner import force_fp32

from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors import Base3DDetector, MVXTwoStageDetector
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
# from opencood.models.futr3d.models.utils.grid_mask import GridMask

from mmcv import Config

# @DETECTORS.register_module()
class FUTR3D(MVXTwoStageDetector):
    def __init__(self,
                 aux_weight=1.0,
                 init_cfg=None):
        super(FUTR3D, self).__init__(init_cfg=init_cfg)

        cfg = Config.fromfile("/home/wangz/wangzhe21/V2XPerception/OpenCOOD/opencood/models/futr3d/configs/lidar_0075v_900q.py")
        # from IPython import embed
        # embed(header='init')
        pts_bbox_head = cfg.model.pts_bbox_head
        train_cfg = cfg.model.train_cfg
        test_cfg = cfg.model.test_cfg
        aux_weight = cfg.model.aux_weight

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)


        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aux_weight = aux_weight

        # if freeze_backbone:
        #     self._freeze_backbone()

    # def _freeze_backbone(self):
    #     for modules in [self.img_backbone, self.img_neck, self.pts_backbone, \
    #                     self.pts_middle_encoder, self.pts_neck]: 
    #         if modules is not None:
    #             modules.eval()
    #             for param in modules.parameters():
    #                 param.requires_grad = False


    def forward_train(self,
                      points_feats=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_bboxes_ignore=None):
        # from IPython import embed
        # embed(header="f_train")
        # pts_feats = tuple([points_feats])
        pts_feats = points_feats

        losses = dict()
        losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        # from IPython import embed
        # embed(header='forward_train')
        loss, log_vars = self._parse_losses(losses)

        return loss, log_vars

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        img_feats = None
        radar_feats = None
        outputs_classes, outputs_coords, aux_outs = \
            self.pts_bbox_head(pts_feats, img_feats, radar_feats, img_metas)
        loss_inputs = (outputs_classes, outputs_coords, gt_bboxes_3d, gt_labels_3d, img_metas)
        
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if aux_outs is not None:
            aux_loss_inputs = [gt_bboxes_3d, gt_labels_3d, aux_outs]
            aux_losses = self.pts_bbox_head.aux_head.loss(*aux_loss_inputs)
            for k, v in aux_losses.items():
                losses[f'aux_{k}'] = v * self.aux_weight


        return losses
    
    def forward_test(self,
                     points_feats=None,
                     img_metas=None,
                     gt_bboxes_3d=None,
                     gt_labels_3d=None,
                     gt_bboxes_ignore=None):
        # pts_feats = tuple([points_feats])
        pts_feats = points_feats
        bbox_list = [dict() for i in range(len(img_metas))]
        
        bbox_pts = self.simple_test_pts(pts_feats, img_metas)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        
        return bbox_list



    def simple_test_pts(self, pts_feats, img_metas, rescale=False):
        """Test function of point cloud branch."""
        img_feats = None
        radar_feats = None
        
        outputs_classes, outputs_coords, aux_outs = \
            self.pts_bbox_head(pts_feats, img_feats, radar_feats, img_metas)
        outs = (outputs_classes, outputs_coords)
        
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        # from IPython import embed
        # embed(header="ssssss")
        return bbox_results


    
    # def aug_test(self, img_metas, points=None, imgs=None, radar=None, rescale=False):
    #     """Test function with augmentaiton."""
    #     img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

    #     bbox_list = dict()
    #     if pts_feats and self.with_pts_bbox:
    #         bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
    #         bbox_list.update(pts_bbox=bbox_pts)
    #     return [bbox_list]

    # def aug_test_pts(self, feats, img_metas, rescale=False):
    #     """Test function of point cloud branch with augmentaiton."""
    #     # only support aug_test for one sample
    #     aug_bboxes = []
    #     for x, img_meta in zip(feats, img_metas):
    #         outs = self.pts_bbox_head(x)
    #         bbox_list = self.pts_bbox_head.get_bboxes(
    #             *outs, img_meta, rescale=rescale)
    #         bbox_list = [
    #             dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
    #             for bboxes, scores, labels in bbox_list
    #         ]
    #         aug_bboxes.append(bbox_list[0])

    #     # after merging, bboxes will be rescaled to the original image size
    #     merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
    #                                         self.pts_bbox_head.test_cfg)
    #     return merged_bboxes