#!/bin/bash


if [ $2 = "y" ]
then
    python setup.py build develop
fi

CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch \
                                  --nproc_per_node=2 \
                                  --master_port=40000 \
                                  tools/train_net.py \
                                  --config-file $1 \