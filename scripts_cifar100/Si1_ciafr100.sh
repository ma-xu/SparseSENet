#!/bin/bash
#BATCH --job-name=Sigmoid1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xuma@my.unt.edu
#SBATCH --ntasks=1
#SBATCH --qos=large
#SBATCH -p public
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -t 500:00:00
#SBATCH --output=outlog/out_%j.log

module load python/3.6.5
module load pytorch/1.1.0


python3 /home/qc0025/Mara/SparseSENet/cifar.py --netName=SEResNet50 --activation=Sigmoid --factor=1 --bs=512
