#! /bin/bash

#SBATCH --time 04:00:00
#SBATCH --mem  8G

. ./venv/bin/activate

./aalto-predict.py --target $1 --hidden_size $2 --picsom_features $3

# ./aalto-predict-with-dev.py --epochs 100 --target $1 --hidden_size $2 --picsom_features $3

