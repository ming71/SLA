#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS="2"

DATASET="HRSC2016"
# DATASET="UCAS_AOD"
# DATASET="DOTA"
# DATASET="DOTA1.5"
# DATASET="DOTA2"


if   [ $DATASET = "HRSC2016" ]; then
    ROOT="hrsc2016"

elif [ $DATASET = "UCAS_AOD" ]; then
    ROOT="ucas_aod"

elif [ $DATASET = "DOTA" ]; then
    ROOT="dota"

elif [ $DATASET = 'DOTA1.5' ]; then
    ROOT='dota1_5'

elif [ $DATASET = 'DOTA2' ]; then
    ROOT='dota2'
fi


# python run/$ROOT/train.py --gpus $GPUS

python -m torch.distributed.launch  run/$ROOT/train_dist.py  --gpus $GPUS
