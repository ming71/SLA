#!/usr/bin/env bash

PYTHON=${PYTHON:-'python'}

GPUS='2'

# DATASET='HRSC2016'
# DATASET='UCAS_AOD'
# DATASET='DOTA'
# DATASET='DOTA1.5'
DATASET='DOTA2'

CKPT='50000.pth'


if   [ $DATASET = 'HRSC2016' ]; then
    ROOT='hrsc2016'
    VOC07_METRIC=True

elif [ $DATASET = 'UCAS_AOD' ]; then
    ROOT='ucas_aod'
    VOC07_METRIC=True

elif [ $DATASET = 'DOTA' ]; then
    ROOT='dota'
    VOC07_METRIC=True

elif [ $DATASET = 'DOTA1.5' ]; then
    ROOT='dota1_5'
    VOC07_METRIC=True

elif [ $DATASET = 'DOTA2' ]; then
    ROOT='dota2'
    VOC07_METRIC=True

fi


# python run/$ROOT/evaluate.py --gpus $GPUS --ckpt $CKPT  --use_voc07   $VOC07_METRIC

python -m torch.distributed.launch  run/$ROOT/evaluate_dist.py  --gpus $GPUS --ckpt $CKPT  --use_voc07   $VOC07_METRIC