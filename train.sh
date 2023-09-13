#!/bin/sh
#SBATCH --account=vision
#SBATCH --partition=tibet
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/ayushsingla/humor_dev/GaitForeMer/logs/sbatch/gaitforemer-focal-loss-10-fold-%A.log

##SBATCH --exclude=deep[17-24]
cd /home/ayushsingla/humor_dev/GaitForeMer
source ../venv/bin/activate
python training/transformer_model_fn.py --non_autoregressive --action=all --pad_decoder_inputs --focal_loss
