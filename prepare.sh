#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}


# DATASET="HRSC2016"
# DATASET="UCAS_AOD"
# DATASET="DOTA"
# DATASET="DOTA1.5"
DATASET="DOTA2"


if   [ $DATASET = "HRSC2016" ]; then
    ROOT="hrsc2016"
    DIR='data/HRSC2016'

elif [ $DATASET = "UCAS_AOD" ]; then
    ROOT="ucas_aod"
    DIR='data/UCAS_AOD'

elif [ $DATASET = "DOTA" ]; then
    ROOT="dota"
    DIR='data/DOTA'

elif [ $DATASET = 'DOTA1.5' ]; then
    ROOT='dota1_5'
    DIR='data/DOTA1_5'

elif [ $DATASET = 'DOTA2' ]; then
    ROOT='dota2'
    DIR='data/DOTA2'
fi


python run/$ROOT/prepare.py  $DIR


