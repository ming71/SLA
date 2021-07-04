#!/usr/bin/env bash

PYTHON=${PYTHON:-'python'}

DATASET='HRSC2016'
# DATASET='UCAS_AOD'
# DATASET='DOTA'
# DATASET='DOTA1.5'
# DATASET='DOTA2'


CKPT='3000.pth'

if   [ $DATASET = 'HRSC2016' ]; then
    ROOT='hrsc2016'

elif [ $DATASET = 'UCAS_AOD' ]; then
    ROOT='ucas_aod'

elif [ $DATASET = 'DOTA' ]; then
    ROOT='dota'

elif [ $DATASET = 'DOTA1.5' ]; then
    ROOT='dota1_5'

elif [ $DATASET = 'DOTA2' ]; then
    ROOT='dota2'
fi



CUDA_VISIBLE_DEVICES=0 python run/$ROOT/demo.py  --ckpt $CKPT 

