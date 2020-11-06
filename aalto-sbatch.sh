#! /bin/bash

#SBATCH --time 04:00:00
#SBATCH --mem  8G

. ./venv/bin/activate

./aalto-predict.py --target long --hidden_size $1 --picsom_features $2

