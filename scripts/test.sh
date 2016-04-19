#!/bin/bash -x 

export THEANO_FLAGS=device=cpu,floatX=float32

# some variable
KL=0
CTX=0
STATE=0
ROOT=/disk1/$USER/nats
MODEL=$ROOT/models/model.npz
DIC=$ROOT/data/toy_train_input.txt.pkl
INPUT=$ROOT/data/toy_test_input.txt
TEMP=./temp.txt
GEN=./final.txt
REF=$ROOT/data/toy_test_output.txt

# generate summarys
python gen.py -n -p 10 -k 5 -l ${KL} -x ${CTX} -s ${STATE} $MODEL $DIC $INPUT $TEMP

# replace unk
python replace_unk.py $INPUT $TEMP $GEN

# calculate rouge score
perl ROUGE.pl 1 N $REF $GEN
perl ROUGE.pl 2 N $REF $GEN
perl ROUGE.pl 1 L $REF $GEN
