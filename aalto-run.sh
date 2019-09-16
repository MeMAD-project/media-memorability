#! /bin/bash

arg=--target=short

python3 -u aalto-predict.py $arg --train_fold=0/4 --output=a
python3 -u aalto-predict.py $arg --train_fold=1/4 --output=b
python3 -u aalto-predict.py $arg --train_fold=2/4 --output=c
python3 -u aalto-predict.py $arg --train_fold=3/4 --output=d

python3 -u aalto-predict.py $arg --train_fold=0/1 --output=8k

arg=--target=long

python3 -u aalto-predict.py $arg --train_fold=0/4 --output=a
python3 -u aalto-predict.py $arg --train_fold=1/4 --output=b
python3 -u aalto-predict.py $arg --train_fold=2/4 --output=c
python3 -u aalto-predict.py $arg --train_fold=3/4 --output=d

python3 -u aalto-predict.py $arg --train_fold=0/1 --output=8k

