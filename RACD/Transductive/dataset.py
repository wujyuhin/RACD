import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    '''
    the dataset for distillation.
    '''
    def __init__(self, df_log, n_user, n_item, theta, psi, Q_mat = None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        self.theta = torch.Tensor(theta.to_numpy())
        self.psi = torch.Tensor(psi.to_numpy())
        pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] \
                = (row['score'] - 0.5) * 2
            pbar.update(1)
        pbar.close()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        user_log = torch.Tensor(self.log_mat[user_id,:])
        item_log = torch.Tensor(self.log_mat[:, item_id])
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        theta_log = self.theta[user_id, :]
        psi_log = self.psi[item_id, :]
        score = torch.FloatTensor([self.score[index]])
        # user_log_mask = user_log.clone()
        # item_log_mask = item_log.clone()
        # user_log_mask[item_id] = 0
        # item_log_mask[user_id] = 0

        return user_log, item_log, user_id, item_id, score, theta_log, psi_log
    
    def __len__(self):
        return self.user_id.shape[0]


class RCDataset(Dataset):
    '''
    the dataset of RCD.
    '''
    def __init__(self, df_log: pd.DataFrame, n_user:int, n_item:int, Q_mat = None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] \
                = (row['score'] - 0.5) * 2
            pbar.update(1)
        pbar.close()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        user_log = torch.Tensor(self.log_mat[user_id,:])
        item_log = torch.Tensor(self.log_mat[:, item_id])
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        score = torch.FloatTensor([self.score[index]])

        # user_log_mask = user_log.clone()
        # item_log_mask = item_log.clone()
        # user_log_mask[item_id] = 0
        # item_log_mask[user_id] = 0

        return user_log, item_log, user_id, item_id, score
    
    def __len__(self):
        return self.user_id.shape[0]