import numpy as np
import os
import sys
import argparse
import json
import time
# from potr.data.Gait17JointsDataset import Gait17JointsDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, precision_recall_fscore_support
from pathlib import Path
from numpyencoder import NumpyEncoder


thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, thispath+"/../")

import training.seq2seq_model_fn as seq2seq_model_fn
import models.PoseTransformer as PoseTransformer
import models.PoseEncoderDecoder as PoseEncoderDecoder
import data.NTURGDDataset as NTURGDDataset
import data.GaitJointsDataset as GaitJointsDataset
import data.SMPLJointsDataset as SMPLJointsDataset
import utils.utils as utils

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WEIGHT_DECAY = 0.00001
_NSEEDS = 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default="/home/ayushsingla/humor_dev/GaitForeMer/logs")
    parser.add_argument('--model_type', type=str, default='RAI')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_path', type=str, default="/home/ayushsingla/humor_dev/data/clinical")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--action', nargs='*', type=str, default=None)
    parser.add_argument('--use_one_hot',  action='store_true')
    parser.add_argument('--init_fn', type=str, default='xavier_init')
    parser.add_argument('--include_last_obs', action='store_true')
    parser.add_argument('--task', type=str, default='downstream', choices=['pretext', 'downstream'])
    parser.add_argument('--downstream_strategy', default='both_then_class', choices=['both', 'class', 'both_then_class'])
    # pose transformers related parameters
    parser.add_argument('--model_dim', type=int, default=256)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dim_ffn', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--source_seq_len', type=int, default=50)                  
    parser.add_argument('--target_seq_len', type=int, default=25)
    parser.add_argument('--max_gradient_norm', type=float, default=0.1)
    parser.add_argument('--lr_step_size',type=int, default=400)
    parser.add_argument('--learning_rate_fn',type=str, default='step')
    parser.add_argument('--warmup_epochs', type=int, default=100)
    parser.add_argument('--pose_format', type=str, default='3D')
    parser.add_argument('--remove_low_std', action='store_true')
    parser.add_argument('--remove_global_trans', action='store_true')
    parser.add_argument('--loss_fn', type=str, default='l1')
    parser.add_argument('--pad_decoder_inputs', action='store_true')  # used in GaitForeMer
    parser.add_argument('--pad_decoder_inputs_mean', action='store_true')
    parser.add_argument('--use_wao_amass_joints', action='store_true')
    parser.add_argument('--non_autoregressive', action='store_true')
    parser.add_argument('--pre_normalization', action='store_true')
    parser.add_argument('--use_query_embedding', action='store_true')  # broken
    parser.add_argument('--predict_activity',  type=bool, default=True)
    parser.add_argument('--use_memory', action='store_true')
    parser.add_argument('--query_selection',action='store_true')  # broken
    parser.add_argument('--activity_weight', type=float, default=1.0)
    parser.add_argument('--pose_embedding_type', type=str, default='gcn_enc')
    parser.add_argument('--encoder_ckpt', type=str, default="/home/ayushsingla/humor_dev/GaitForeMer/checkpoints/pre-trained_ntu_ckpt_epoch_0099.pt")
    parser.add_argument('--dataset', type=str, default='tug_gait')
    parser.add_argument('--skip_rate', type=int, default=5)
    parser.add_argument('--eval_num_seeds', type=int, default=_NSEEDS)
    parser.add_argument('--copy_method', type=str, default=None)
    parser.add_argument('--finetuning_ckpt', type=str, default=None)
    parser.add_argument('--pos_enc_alpha', type=float, default=10)
    parser.add_argument('--pos_enc_beta', type=float, default=500)
    args = parser.parse_args()
    
  params = vars(args)

  if params['task'] == 'downstream':
    num_folds = 54
  else:
    num_folds = 1
  total_preds = []
  total_gts = []

  preds_votes = []
  preds_probs = []

  all_folds = range(1, 55)
  for fold in all_folds:

    print(f'Fold {fold} out of {num_folds}')

    utils.create_dir_tree(params['model_prefix']) # moving this up because dataset mean and std stored under it

    train_dataset_fn, eval_dataset_fn = dataset_factory(params, fold, params['model_prefix'])

    params['input_dim'] = train_dataset_fn.dataset._data_dim
    params['pose_dim'] = train_dataset_fn.dataset._pose_dim
    pose_encoder_fn, pose_decoder_fn = \
        PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

    config_path = os.path.join(params['model_prefix'], 'config', 'config.json')        
    with open(config_path, 'w') as file_:
      json.dump(params, file_, indent=4)

    model_fn = POTRModelFn(
        params, train_dataset_fn, 
        eval_dataset_fn, 
        pose_encoder_fn, pose_decoder_fn
    )
    if params['task'] == 'downstream':
      predictions, gts, pred_probs = model_fn.train()

      print('predicitons:', predictions)

      # save predicted classes
      preds_votes.append(predictions.tolist())

      # save predicted probabilities
      preds_probs.append(pred_probs.tolist())

      # save final predictions and true labels
      if np.shape(gts)[0] == 1: # only 1 clip
        pred = int(predictions)
      else:
        pred = single_vote(predictions)
      gt = gts[0]
      total_preds.append(pred)
      total_gts.append(int(gt))

      del model_fn, pose_encoder_fn, pose_decoder_fn

      attributes = [preds_votes, total_preds, preds_probs, total_gts]
      names = ['predicted_classes', 'predicted_final_classes', 'prediction_list', 'true_labels']
      jsonfilename = os.path.join(params['model_prefix'], 'results.json')        
      save_json(jsonfilename, attributes, names)
    else:
      model_fn.train()

  if params['task'] == 'downstream':
    print(classification_report(total_gts, total_preds))
