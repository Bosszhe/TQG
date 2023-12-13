CONFIG_FILE="opencood/hypes_yaml/point_pillar_intermediate_multi_time.yaml"
FUSION_METHOD="intermediate"
PORT=${PORT:-29501}

CUDA_VISIBLE_DEVICES=4,5 \
python -m torch.distributed.launch \
    --nproc_per_node=2  \
    --use_env \
    --master_port=$PORT \
    opencood/tools/train0.py \
    --hypes_yaml ${CONFIG_FILE} \
    --fusion_method ${FUSION_METHOD}