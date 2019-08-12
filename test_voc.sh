#!/bin/bash


if [ $2 = "y" ]
then
    python setup.py build develop
fi

CUDA_VISIBLE_DEVICES="2" python tools/test_net.py \
                                --config-file $1 \
                                MODEL.WEIGHT FCOS_R_50_FPN_1x.pth \
                                TEST.IMS_PER_BATCH 4    