#!/bin/bash

python pruning.py --arch fpn --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch pan --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch deeplabv3 --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch unet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch manet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch linknet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch pspnet --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch unet++ --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8
python pruning.py --arch deeplabv3+ --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.8