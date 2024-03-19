# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.late_fusion_dataset import LateFusionDataset
from opencood.data_utils.datasets.early_fusion_dataset import EarlyFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset import IntermediateFusionDataset
from opencood.data_utils.datasets.intermediate_fusion_dataset_v2 import IntermediateFusionDatasetV2
from opencood.data_utils.datasets.intermediate_fusion_dataset_scope import IntermediateFusionDatasetSCOPE
from opencood.data_utils.datasets.intermediate_fusion_dataset_irregular import IntermediateFusionDatasetIrregular
from opencood.data_utils.datasets.intermediate_fusion_dataset_irregular_detr3d import IntermediateFusionDatasetIrregularDETR3D
from opencood.data_utils.datasets.intermediate_fusion_dataset_detr3d import IntermediateFusionDatasetDETR3D
from opencood.data_utils.datasets.early_fusion_dataset_detr3d import EarlyFusionDatasetDETR3D

__all__ = {
    'LateFusionDataset': LateFusionDataset,
    'EarlyFusionDataset': EarlyFusionDataset,
    'IntermediateFusionDataset': IntermediateFusionDataset,
    'IntermediateFusionDatasetV2': IntermediateFusionDatasetV2,
    'IntermediateFusionDatasetSCOPE':IntermediateFusionDatasetSCOPE,
    'IntermediateFusionDatasetIrregular':IntermediateFusionDatasetIrregular,
    'IntermediateFusionDatasetIrregularDETR3D':IntermediateFusionDatasetIrregularDETR3D,
    'IntermediateFusionDatasetDETR3D':IntermediateFusionDatasetDETR3D,
    'EarlyFusionDatasetDETR3D':EarlyFusionDatasetDETR3D
}

# the final range for evaluation
GT_RANGE = [-140, -40, -3, 140, 40, 1]
# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in ['LateFusionDataset', 'EarlyFusionDataset',
                            'IntermediateFusionDataset', 'IntermediateFusionDatasetV2','IntermediateFusionDatasetSCOPE',
                            'IntermediateFusionDatasetIrregular',
                            'IntermediateFusionDatasetIrregularDETR3D',
                            'IntermediateFusionDatasetDETR3D','EarlyFusionDatasetDETR3D'], error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train
    )

    return dataset
