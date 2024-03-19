

class_names = [
    'car'
]
voxel_size = [0.4, 0.4, 4]
point_cloud_range = [-140.8, -40.0, -3.0, 140.8, 40, 1.0]

center_head = dict(
    type='CenterHead',
    in_channels=256,
    tasks=[
        dict(num_class=1, class_names=['car']),
    ],
    common_heads=dict(
        reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
    share_conv_channel=64,
    bbox_coder=dict(
        type='CenterPointBBoxCoder',
        pc_range=point_cloud_range[:2],
        post_center_range=[-150.8, -50.0, -10.0, 150.8, 50.0, 10.0],
        max_num=500,
        score_threshold=0.1,
        out_size_factor=2,
        voxel_size=voxel_size[:2],
        code_size=9),
    separate_head=dict(
        type='SeparateHead', init_bias=-2.19, final_kernel=3),
    loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
    loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
    norm_bbox=True)

model = dict(
    type='FUTR3D',
    aux_weight=0.5,
    pts_voxel_layer=dict(
        max_num_points=10, 
        voxel_size=voxel_size, 
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 200, 704],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=False),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        add_extra_convs=True,
        num_outs=4,
        relu_before_extra_convs=True,
        ),
    pts_bbox_head=dict(
        type='FUTR3DHead',
        use_dab=True,
        anchor_size=3,
        use_aux=False,
        aux_head=center_head,
        mix_selection=False,
        num_query=900,
        num_classes=1,
        in_channels=256,
        pc_range=point_cloud_range,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='FUTR3DTransformer',
            use_dab=True,
            num_feature_levels = 1,
            decoder=dict(
                type='FUTR3DTransformerDecoder',
                num_layers=6,
                use_dab=True,
                anchor_size=3,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='FUTR3DAttention',
                            embed_dims=256)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0)
        ),
    # training and testing settings
    train_cfg=dict(pts=dict(
        point_cloud_range=point_cloud_range,
        pc_range=point_cloud_range,
        grid_size=[704, 200, 1],
        voxel_size=voxel_size,
        out_size_factor=2,
        dense_reg=1,
        gaussian_overlap=0.1,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0)))),
    test_cfg=dict(pts=dict(
        pc_range=point_cloud_range[:2],
        post_center_limit_range=[-150.8, -50.0, -10.0, 150.8, 50.0, 10.0],
        max_per_img=500,
        max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.85, 0.175],
        out_size_factor=2,
        voxel_size=voxel_size[:2],
        nms_type='circle',
        pre_max_size=1000,
        post_max_size=83,
        nms_thr=0.2,
        max_num=100,
        score_threshold=0,
        post_center_range=[-150.8, -50.0, -10.0, 150.8, 50.0, 10.0],
    )))




