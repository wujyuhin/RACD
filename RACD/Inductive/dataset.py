import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class DRDataset(Dataset):
    '''
    the dataset for dr.
    '''
    def __init__(self, df_log, n_user, n_item, theta, psi, valid_type=None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        self.theta = torch.Tensor(theta.to_numpy())
        self.psi = torch.Tensor(psi.to_numpy())
        self.valid_type = valid_type
        pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] \
                = (row['score'] - 0.5) * 2
            pbar.update(1)
        pbar.close()
        if valid_type == 'inductive':
            self.log_mat_train = self.log_mat.copy()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        user_log = torch.Tensor(self.log_mat[user_id,:])
        if self.valid_type == 'inductive':
            item_log = torch.Tensor(self.log_mat_train[:, item_id])
        else:
            item_log = torch.Tensor(self.log_mat[:, item_id])
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        theta_log = self.theta[user_id, :]
        psi_log = self.psi[item_id, :]
        score = torch.FloatTensor([self.score[index]])

        return user_log, item_log, user_id, item_id, score, theta_log, psi_log
    
    def __len__(self):
        return self.user_id.shape[0]


class RACDataset(Dataset):
    '''
    the dataset of RACD.
    '''
    def __init__(self, df_log: pd.DataFrame, n_user:int, n_item:int, valid_type=None):
        self.df_log = df_log
        self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        self.valid_type = valid_type
        pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        for i, row in df_log.iterrows():
            self.log_mat[int(row['user_id']), int(row['item_id'])] \
                = (row['score'] - 0.5) * 2
            pbar.update(1)
        pbar.close()
        if valid_type == 'inductive':
            self.log_mat_train = self.log_mat.copy()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        user_log = torch.Tensor(self.log_mat[user_id,:])
        if self.valid_type == 'inductive':
            item_log = torch.Tensor(self.log_mat_train[:, item_id])
        else:
            item_log = torch.Tensor(self.log_mat[:, item_id])
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        score = torch.FloatTensor([self.score[index]])

        return user_log, item_log, user_id, item_id, score
    
    def __len__(self):
        return self.user_id.shape[0]
    

# s_temb = pd.read_csv(args.data_root + "stu_embeddings.csv", header=None)
# e_temb = pd.read_csv(args.data_root + "item_embeddings.csv", header=None)

class DRDataset_T(Dataset):
    '''
    the dataset for dr.
    '''
    def __init__(self, df_log, s_temb, e_temb, n_user, n_item, theta, psi, valid_type=None):
        self.df_log = df_log
        self.s_emb = torch.tensor(s_temb.values) #dtype=torch.float32
        self.e_emb = torch.tensor(e_temb.values) #dtype=torch.float32
        # self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        self.theta = torch.Tensor(theta.to_numpy())
        self.psi = torch.Tensor(psi.to_numpy())
        # self.valid_type = valid_type
        # pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        # for i, row in df_log.iterrows():
        #     self.log_mat[int(row['user_id']), int(row['item_id'])] \
        #         = (row['score'] - 0.5) * 2
        #     pbar.update(1)
        # pbar.close()
        # if valid_type == 'inductive':
        #     self.log_mat_train = self.log_mat.copy()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        user_log = self.s_emb[user_id, :] # torch.Tensor(self.log_mat[user_id,:])
        # if self.valid_type == 'inductive':
        #     item_log = torch.Tensor(self.log_mat_train[:, item_id])
        # else:
        #     item_log = torch.Tensor(self.log_mat[:, item_id])
        item_log = self.e_emb[item_id, :]
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        theta_log = self.theta[user_id, :]
        psi_log = self.psi[item_id, :]
        score = torch.FloatTensor([self.score[index]])

        return user_log, item_log, user_id, item_id, score, theta_log, psi_log
    
    def __len__(self):
        return self.user_id.shape[0]
    
class RACDataset_T(Dataset):
    '''
    the dataset of RACD.
    '''
    def __init__(self, df_log: pd.DataFrame, s_temb, e_temb, n_user:int, n_item:int, valid_type=None):
        self.df_log = df_log
        self.s_emb = torch.tensor(s_temb.values) #dtype=torch.float32
        self.e_emb = torch.tensor(e_temb.values) #dtype=torch.float32
        # self.log_mat = np.zeros((n_user, n_item))
        self.user_id = df_log['user_id'].values
        self.item_id = df_log['item_id'].values
        self.score = df_log['score'].values
        # self.valid_type = valid_type
        # pbar = tqdm(total = df_log.shape[0],desc='Loading data')
        # for i, row in df_log.iterrows():
        #     self.log_mat[int(row['user_id']), int(row['item_id'])] \
        #         = (row['score'] - 0.5) * 2
        #     pbar.update(1)
        # pbar.close()
        # if valid_type == 'inductive':
        #     self.log_mat_train = self.log_mat.copy()
    
    def __getitem__(self, index):
        user_id = self.user_id[index]
        item_id = self.item_id[index]
        # user_log = torch.Tensor(self.log_mat[user_id,:])
        # if self.valid_type == 'inductive':
        #     item_log = torch.Tensor(self.log_mat_train[:, item_id])
        # else:
        #     item_log = torch.Tensor(self.log_mat[:, item_id])
        user_log = self.s_emb[user_id, :]
        item_log = self.e_emb[item_id, :]
        user_id = torch.LongTensor([user_id])
        item_id = torch.LongTensor([item_id])
        score = torch.FloatTensor([self.score[index]])

        return user_log, item_log, user_id, item_id, score
    
    def __len__(self):
        return self.user_id.shape[0]