# CONFIG_FILE="opencood/hypes_yaml/point_pillar_intermediate_multi_time.yaml"
# FUSION_METHOD="intermediate"
# CONFIG_FILE="opencood/hypes_yaml/point_pillar_detr3d_early_fusion.yaml"
# FUSION_METHOD="early"
# PORT=${PORT:-29503}

# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch \
#     --nproc_per_node=2  \
#     --use_env \
#     --master_port=$PORT \
#     opencood/tools/train0.py \
#     --hypes_yaml ${CONFIG_FILE} \
#     --fusion_method ${FUSION_METHOD}


# CONFIG_FILE="opencood/hypes_yaml/point_pillar_detr3d_early_fusion.yaml"
# FUSION_METHOD="detr3d"
# PORT=${PORT:-29503}

# CUDA_VISIBLE_DEVICES=4,5 \
# python -m torch.distributed.launch \
#     --nproc_per_node=2  \
#     --use_env \
#     --master_port=$PORT \
#     opencood/tools/train_detr3d.py \
#     --hypes_yaml ${CONFIG_FILE} \
#     --fusion_method ${FUSION_METHOD}



CONFIG_FILE="opencood/hypes_yaml/point_pillar_early_fusion.yaml"
FUSION_METHOD="early"
PORT=${PORT:-29503}

CUDA_VISIBLE_DEVICES=5,6 \
python -m torch.distributed.launch \
    --nproc_per_node=2  \
    --use_env \
    --master_port=$PORT \
    opencood/tools/train0.py \
    --hypes_yaml ${CONFIG_FILE} \
    --fusion_method ${FUSION_METHOD}