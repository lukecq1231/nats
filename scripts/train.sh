#!/bin/bash

# use CUDNN
export LD_LIBRARY_PATH=/disk1/$USER/src/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/disk1/$USER/src/cuda/include:$CPATH
export LIBRARY_PATH=/disk1/$USER/src/cuda/lib64:$LD_LIBRARY_PATH
export THEANO_FLAGS=device=gpu0,floatX=float32, optimizer_including=cudnn

# no CUDNN
# export THEANO_FLAGS=device=gpu0,floatX=float32

python -u ./train_nats.py > log.txt 2>&1 &




