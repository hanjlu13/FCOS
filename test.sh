#!/bin/bash


if [ $2 = "y" ]
then
    python setup.py build develop
fi

CUDA_VISIBLE_DEVICES="2" python tools/test_net.py \
                                --config-file $1 \
                                MODEL.WEIGHT /home/hancock/data/training_records/fcos/VOC/FCOS_R50_0.0025/model_final.pth \
                                TEST.IMS_PER_BATCH 4    