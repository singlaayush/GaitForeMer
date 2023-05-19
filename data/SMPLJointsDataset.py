import argparse
import joblib
import random
import pandas as pd
from pathlib import Path

import torch
import numpy as np

_TOTAL_ACTIONS = 2
_NSEEDS = 8

# mapping from 17 SMPL joints to NTU (1-base)
# 'spine'/'spine1' = 3/6 -> 2 = 'middle of the spine' (only one spine joint recorded in NTU)
#  NTU Order:    1, 2,  3,  4,  5,  6,  7,  9, 10, 11, 13, 14, 15, 17, 18, 19, 21
_MAJOR_JOINTS = [0, 3, 12, 15, 13, 16, 18, 14, 17, 19,  1,  4,  7,  2,  5, 11,  9]  # SMPL Order

_NMAJOR_JOINTS = len(_MAJOR_JOINTS)
_MIN_STD = 1e-4
_SPINE_ROOT = 0 # after only taking major joints (i.e. spine's index in _MAJOR_JOINTS)

def collate_fn(batch):
    """Collate function for data loaders."""
    e_inp = torch.from_numpy(np.stack([e['encoder_inputs'] for e in batch]))
    d_inp = torch.from_numpy(np.stack([e['decoder_inputs'] for e in batch]))
    d_out = torch.from_numpy(np.stack([e['decoder_outputs'] for e in batch]))
    action_id = torch.from_numpy(np.stack([e['action_id'] for e in batch]))
    action = [e['action_str'] for e in batch]

    batch_ = {
        'encoder_inputs': e_inp,
        'decoder_inputs': d_inp,
        'decoder_outputs': d_out,
        'action_str': action,
        'action_ids': action_id
    }

    return batch_

class SMPLJointsDataset(torch.utils.data.Dataset):
    def __init__(self, params=None, load_type='csv', mode='train'):
        super(SMPLJointsDataset, self).__init__()
        self._params = params
        self._mode = mode
        self.thisname = self.__class__.__name__
        self.data_dir = self._params['data_path']
        self.load_data(load_type)
    
    def get_sampler_weights(self):
        len_Y = len(self.Y)
        weight_per_class = len_Y / np.bincount(self.Y)
        weights = [0] * len_Y
        for i in range(len_Y):
            image_class = self.Y[i]
            weights[i] = weight_per_class[image_class]
        return weights
        # return weights, weight_per_class

    def load_data(self, load_type):
        suffix = f"{load_type if type(load_type) == type(1) else ''}"
        pkl_file = f"train{suffix}.pkl" if self._mode == 'train' else f"test{suffix}.pkl"
        csv_file = "binned_RAI.csv"
        pkl_data = joblib.load(open(Path(self.data_dir) / pkl_file, "rb"))
        X_1, Y = self.data_generator_csv(pkl_data, csv_file) if suffix == "" else self.data_generator_pkl(pkl_data)    
        
        self.X_1 = X_1
        self.Y = Y
        self._monitor_action = 'robust'
        if "binary" in self._params['model_type']:
            self._action_str = ['robust', 'frail']
        elif "combined" in self._params['model_type']:
            self._action_str = ['robust', 'moderate', 'severe']
        else:
            self._action_str = ['robust', 'mild', 'moderate', 'severe']

        self._pose_dim = 3 * _NMAJOR_JOINTS
        self._data_dim = self._pose_dim
        
        self.compute_norm_stats()  # compute dataset mean and std
        self.normalize_data()  # normalize joints stored in _data dict using computed mean and std

    def compute_norm_stats(self):
        self._norm_stats = {}
        data = np.concatenate(self.X_1, axis=0)  # shape: [total_frames_in_dataset, 16 * 3]
        data = np.reshape(data, (data.shape[0], -1))
        mean = np.mean(data, axis=0)  # computes the mean [16 * 3] across all frames in the dataset (i.e. mean across all joints in all frames concatenated from all videos) 
        std = np.std(data, axis=0)  # was np.mean before – fixed it assuming that is vv likely a typo 
        std[np.where(std<_MIN_STD)] = 1

        self._norm_stats['mean'] = mean.ravel() # flattens mean into a 1-D array
        self._norm_stats['std'] = std.ravel() # flattens std into a 1-D array

    def normalize_data(self):
        for i, k in enumerate(self.X_1):
            tmp_data = np.reshape(k, (k.shape[0], -1))
            tmp_data = tmp_data - self._norm_stats['mean']
            tmp_data = np.divide(tmp_data, self._norm_stats['std'])
            self.X_1[i] = tmp_data

    def data_generator_csv(self, data_dict, csv_file):
        df = pd.read_csv(Path(self.data_dir) / csv_file, index_col=0)
        X_1 = []
        Y = []
        for key in data_dict.keys():
            p = np.copy(data_dict[key][:,_MAJOR_JOINTS,:])
            label = int(df.loc[df['ID'] == int(key)]['Y'].values[0])
            X_1.append(p)
            Y.append(label)
        return X_1, np.stack(Y)
    
    def data_generator_pkl(self, pkl_data):
        X_1, Y = pkl_data
        return X_1, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x = self.X_1[idx]  # [all_frames, n_joints*joint_dim]
        y = self.Y[idx]
        
        action_id = y
        N = np.shape(x)[0]  # all_frames
        
        source_seq_len = self._params['source_seq_len']
        target_seq_len = self._params['target_seq_len']
        pose_dim = self._norm_stats['std'].shape[-1]  # input and output dims are the same
        total_frames = source_seq_len + target_seq_len
        src_seq_len = source_seq_len - 1

        encoder_inputs = np.zeros((src_seq_len, pose_dim), dtype=np.float32)
        decoder_inputs = np.zeros((target_seq_len, pose_dim), dtype=np.float32)
        decoder_outputs = np.zeros((target_seq_len, pose_dim), dtype=np.float32)

        # apparently, np.random.randint() did not change the start frame between epochs
        start_frame = random.randint(0, N - total_frames) # high inclusive
        data_sel = x[start_frame:(start_frame + total_frames), :]  # [total_frames, n_joints*joint_dim]

        encoder_inputs[:, 0:pose_dim] = data_sel[0:src_seq_len,:]
        decoder_inputs[:, 0:pose_dim] = \
                data_sel[src_seq_len:src_seq_len+target_seq_len, :]
        decoder_outputs[:, 0:pose_dim] = data_sel[source_seq_len:, 0:pose_dim]

        if self._params['pad_decoder_inputs']:
            query = decoder_inputs[0:1, :]
            decoder_inputs = np.repeat(query, target_seq_len, axis=0)

        return {
                'encoder_inputs': encoder_inputs, 
                'decoder_inputs': decoder_inputs, 
                'decoder_outputs': decoder_outputs,
                'action_id': action_id,
                'action_str': self._action_str[action_id],
        }

def dataset_factory(params):
    """Defines the datasets that will be used for training and testing/validation."""
    # params['load_type'] = 0
    # params['num_activities'] = _TOTAL_ACTIONS
    params['virtual_dataset_size'] = params['steps_per_epoch'] * params['batch_size']
    params['n_joints'] = _NMAJOR_JOINTS

    train_dataset = SMPLJointsDataset(params, load_type=params['load_type'], mode='train')
    sampler_weights = train_dataset.get_sampler_weights()
    train_dataset_fn = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            num_workers=0,
            sampler=torch.utils.data.WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True),
            collate_fn=collate_fn,
    )
    
    eval_dataset = SMPLJointsDataset(params, load_type=params['load_type'], mode='test')
    eval_dataset_fn = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
    ) 

    return train_dataset_fn, eval_dataset_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='binary_combined')
    parser.add_argument('--num_activities', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_path', type=str, default="/home/ayushsingla/humor_dev/GaitForeMer/logs")
    parser.add_argument('--data_path', type=str, default="/home/ayushsingla/humor_dev/GaitForeMer/data/smpl_k_fold/binary_combined")
    parser.add_argument('--learning_rate', type=float, default=1e-4)
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
    parser.add_argument('--warmup_epochs', type=int, default=10)
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
    params['load_type'] = 0
    train_loader, test_loader = dataset_factory(params)
    train_batch = iter(train_loader).next()
    print(f"test_loader length: {len(test_loader)}")
    print("Sample train batch:")
    print(train_batch['encoder_inputs'].shape)
    print(train_batch['decoder_inputs'].shape)
    print(train_batch['decoder_outputs'].shape)
    print(np.bincount(train_batch['action_ids']))
    print("Sample test batch:")
    #for test_batch in test_loader:
    test_batch = iter(test_loader).next()
    print(test_batch['encoder_inputs'].shape)
    print(test_batch['decoder_inputs'].shape)
    print(test_batch['decoder_outputs'].shape)
    print(test_batch['action_ids'])
    print(test_batch['action_str'])
    print(np.unique(train_loader.dataset.get_sampler_weights()))