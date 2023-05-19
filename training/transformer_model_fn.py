###############################################################################
# Pose Transformers (POTR): Human Motion Prediction with Non-Autoregressive 
# Transformers
# 
# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
# Written by 
# Angel Martinez <angel.martinez@idiap.ch>,
# 
# This file is part of 
# POTR: Human Motion Prediction with Non-Autoregressive Transformers
# 
# POTR is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# POTR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with POTR. If not, see <http://www.gnu.org/licenses/>.
###############################################################################

# python training/transformer_model_fn.py --non_autoregressive --action=all --pad_decoder_inputs --focal_loss

"""Implments the model function for the POTR model."""


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
from datetime import datetime
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
import utils.losses as losses

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_WEIGHT_DECAY = 0.00001
_NSEEDS = 8

class POTRModelFn(seq2seq_model_fn.ModelFn):
    def __init__(self,
                             params,
                             train_dataset_fn,
                             eval_dataset_fn,
                             pose_encoder_fn=None,
                             pose_decoder_fn=None):
        super(POTRModelFn, self).__init__(
            params, train_dataset_fn, eval_dataset_fn, pose_encoder_fn, pose_decoder_fn)
        self._loss_fn = self.layerwise_loss_fn
        self.task = params['task']
        self.use_focal_loss = params['focal_loss']
        if self.task == 'downstream':
            #weights = torch.tensor([141., 24.])
            #weights = weights / weights.sum() # turn into percentage
            #weights = 1.0 / weights # inverse
            #weights = weights / weights.sum()
            if self.use_focal_loss:
                self._downstream_loss = losses.FocalLoss(alpha=0.25, gamma=2)
                print(f'Using Focal Loss for gait impairment score prediction.')
            else:
                weights = torch.tensor(np.unique(train_dataset_fn.dataset.get_sampler_weights())).float()
                self._loss_weights = weights.to(_DEVICE)
                self._downstream_loss = nn.CrossEntropyLoss(weight=self._loss_weights)
                print(f'Using a weighted CE loss for gait impairment score prediction with weights={weights}.')
        else:
            print('Using a standard CE loss for activity prediction.')

    def smooth_l1(self, decoder_pred, decoder_gt):
        l1loss = nn.SmoothL1Loss(reduction='mean')
        return l1loss(decoder_pred, decoder_gt)

    def loss_l1(self, decoder_pred, decoder_gt):
        return nn.L1Loss(reduction='mean')(decoder_pred, decoder_gt)

    def loss_activity(self, logits, class_gt):                                     
        """Computes entropy loss from logits between predictions and class."""
        if self.task == 'downstream':
            return self._downstream_loss(logits, class_gt)
        else:
            return nn.functional.cross_entropy(logits, class_gt, reduction='mean')

    def compute_class_loss(self, class_logits, class_gt):
        """Computes the class loss for each of the decoder layers predictions or memory."""
        class_loss = 0.0
        for l in range(len(class_logits)):
            class_loss += self.loss_activity(class_logits[l], class_gt)

        return class_loss/len(class_logits)

    def select_loss_fn(self):
        if self._params['loss_fn'] == 'mse':
            return self.loss_mse
        elif self._params['loss_fn'] == 'smoothl1':
            return self.smooth_l1
        elif self._params['loss_fn'] == 'l1':
            return self.loss_l1
        else:
            raise ValueError('Unknown loss name {}.'.format(self._params['loss_fn']))

    def layerwise_loss_fn(self, decoder_pred, decoder_gt, class_logits=None, class_gt=None):
        """Computes layerwise loss between predictions and ground truth."""
        pose_loss = 0.0
        loss_fn = self.select_loss_fn()

        for l in range(len(decoder_pred)):
            pose_loss += loss_fn(decoder_pred[l], decoder_gt)

        pose_loss = pose_loss/len(decoder_pred)
        if class_logits is not None:
            return pose_loss, self.compute_class_loss(class_logits, class_gt)

        return pose_loss, None

    def init_model(self, pose_encoder_fn=None, pose_decoder_fn=None):
        self._model = PoseTransformer.model_factory(
                self._params, 
                pose_encoder_fn, 
                pose_decoder_fn
        )

    def select_optimizer(self):
        optimizer = optim.AdamW(
                self._model.parameters(), lr=self._params['learning_rate'],
                betas=(0.9, 0.999),
                weight_decay=_WEIGHT_DECAY
        )

        return optimizer


def dataset_factory(params, fold=None):
    if params['dataset'] == 'ntu_rgbd':
        return NTURGDDataset.dataset_factory(params)
    elif params['dataset'] == 'pd_gait':
        return GaitJointsDataset.dataset_factory(params, fold)
    elif params['dataset'] == 'tug_gait':
        return SMPLJointsDataset.dataset_factory(params)
    else:
        raise ValueError('Unknown dataset {}'.format(params['dataset']))

def single_vote(pred):
    """
    Get majority vote of predicted classes for the clips in one video.
    :param preds: list of predicted class for each clip of one video
    :return: majority vote of predicted class for one video
    """
    p = np.array(pred)
    counts = np.bincount(p)
    max_count = 0
    max_index = 0
    for i in range(len(counts)):
        if max_count < counts[i]:
            max_index = i
            max_count = counts[i]
    return max_index

def save_json(filename, attributes, names):
        """
        Save training parameters and evaluation results to json file.
        :param filename: save filename
        :param attributes: attributes to save
        :param names: name of attributes to save in json file
        """
        with open(filename, "w", encoding="utf8") as outfile:
                d = {}
                for i in range(len(attributes)):
                        name = names[i]
                        attribute = attributes[i]
                        d[name] = attribute
                json.dump(d, outfile, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='binary_combined')
    parser.add_argument('--num_activities', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_path', type=str, default="/home/ayushsingla/humor_dev/GaitForeMer/logs")
    parser.add_argument('--data_path', type=str, default="/home/ayushsingla/humor_dev/GaitForeMer/data/smpl_k_fold")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--action', nargs='*', type=str, default=None)
    parser.add_argument('--use_one_hot',  action='store_true')
    parser.add_argument('--init_fn', type=str, default='xavier_init')
    parser.add_argument('--include_last_obs', action='store_true')
    parser.add_argument('--task', type=str, default='downstream', choices=['pretext', 'downstream'])
    parser.add_argument('--downstream_strategy', default='both_then_class', choices=['both', 'class', 'both_then_class'])
    # pose transformers related parameters
    parser.add_argument('--model_dim', type=int, default=128)
    parser.add_argument('--num_encoder_layers', type=int, default=4)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dim_ffn', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--source_seq_len', type=int, default=40)                  
    parser.add_argument('--target_seq_len', type=int, default=20)
    parser.add_argument('--max_gradient_norm', type=float, default=0.1)
    parser.add_argument('--lr_step_size',type=int, default=400)
    parser.add_argument('--learning_rate_fn',type=str, default='step')
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
    parser.add_argument('--focal_loss', action='store_true')
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
    params['data_path'] = f"{params['data_path']}/{params['model_type']}"
    params['time'] = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")

    num_folds = 10
    
    #total_gts = []
    #total_preds = []
    #preds_votes = []
    #preds_probs = []
    precisions, recalls, f1s = [], [], []
    class_precisions, class_recalls, class_f1s= [], [], []

    #print(f"Training using leave-one-out cross validation.")
    print(f"Training using {num_folds}-fold cross validation.")
    
    for fold in range(num_folds):
        print(f'Fold {fold + 1} out of {num_folds}:')

        params['model_prefix'] = str(Path(params['base_path']) / params['model_type'] / params['time'] / f"_Fold_{fold}")
        
        utils.create_dir_tree(params['model_prefix']) # moving this up because dataset mean and std stored under it

        params['load_type'] = fold
        train_dataset_fn, eval_dataset_fn = dataset_factory(params)

        params['input_dim'] = train_dataset_fn.dataset._data_dim
        params['pose_dim'] = train_dataset_fn.dataset._pose_dim
        pose_encoder_fn, pose_decoder_fn = \
                PoseEncoderDecoder.select_pose_encoder_decoder_fn(params)

        config_path = Path(params['model_prefix']) / 'config' / f'config.json'
        with open(config_path, 'w') as file_:
            json.dump(params, file_, indent=4)

        model_fn = POTRModelFn(
                params, train_dataset_fn, 
                eval_dataset_fn, 
                pose_encoder_fn, pose_decoder_fn
        )
        predictions, gts, pred_probs = model_fn.train()

        # save predicted classes
        #preds_votes.append(predictions.tolist())

        # save predicted probabilities
        #preds_probs.append(pred_probs.tolist())
        
        # save final predictions and true labels
        #if np.shape(gts)[0] == 1: # only 1 clip
        #  pred = int(predictions)
        #else:
        #  pred = single_vote(predictions)
        #gt = gts[0]
        #total_preds.append(pred)
        #total_gts.append(int(gt))
        
        #attributes = [preds_votes, total_preds, preds_probs, total_gts]  # outputs everything thus far – bug, but keeping it for now as it is useful for debugging
        #names = ['predicted_classes', 'predicted_final_classes', 'prediction_list', 'true_labels']
        #jsonfilename = os.path.join(params['model_prefix'], 'results.json')        
        #save_json(jsonfilename, attributes, names)
        
        del model_fn, pose_encoder_fn, pose_decoder_fn
        
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(gts, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(gts, predictions, average='macro')
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        class_precisions.append(class_precision)
        class_recalls.append(class_recall)
        class_f1s.append(class_f1)
        
        print("\n")
        print(f"Fold {fold} macro-precision: {precision}")
        print(f"Fold {fold} macro-recall:    {recall}")
        print(f"Fold {fold} macro-F1:        {f1}")
        print("\n")

    #print(classification_report(total_gts, total_preds))
    
    #final_cls_report = classification_report(total_gts, total_preds, output_dict=True)
    #jsonfilename = str(Path(params['base_path']) / params['model_type'] / params['time'] / 'results.json')        
    #save_json(jsonfilename, list(final_cls_report.values()), list(final_cls_report.keys()))

    averages = [
        f"Average macro-precision: {np.mean(precisions)*100   }% (+/- {np.std(precisions)*100 })",
        f"Average macro-recall:    {np.mean(recalls)*100      }% (+/- {np.std(recalls)*100    })",
        f"Average macro-F1:        {np.mean(f1s)*100          }% (+/- {np.std(f1s)*100        })"
    ]
    attributes = [class_precisions, class_recalls, class_f1s, precisions, recalls, f1s, averages]
    names = ['class_precisions', 'class_recalls', 'class_f1s', 'precisions', 'recalls', 'f1s', 'macro-averages']
    jsonfilename = str(Path(params['base_path']) / params['model_type'] / params['time'] / 'results.json')        
    save_json(jsonfilename, attributes, names)
    
    print(averages, sep='\n')
