import numpy as np
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.distributions as distributions
from torch.autograd import Function
from torch.distributions import Normal

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
from utils.hamming import find_topk_similar_hashes_batch



class PosLinear(nn.Linear):
    def forward(self, input):
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class EAD(nn.Module):
    def __init__(self, n_user, n_item, n_know, user_dim, item_dim, Q_mat, device,
                  monotonicity_assumption = True):
        '''
        :param n_user: int, the number of learners.
        :param n_item: int, the number of test items.
        :param n_know: int, the number of knowledge concepts.
        :param user_dim: int, the dimension of mixed user representations.
        :param item_dim: int, the dimension of mixed item representations.
        :param Q_mat: np.array [n_item,n_know], the binary Q-matrix.
        :param device: torch.device
        :param monotonicity_assumption: bool, whether to apply
                the monotonicity assumption to the diagnostic module. If True,
                the monotonicity assumption is applied.
        '''
        super(EAD, self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.n_know = n_know
        self.user_dim = user_dim 
        self.item_dim = item_dim
        self.itf = self.ncd_func
        self.device = device

        self.Q_mat = torch.Tensor(Q_mat) if Q_mat is not None else torch.ones((n_item, n_know))
        self.Q_mat = self.Q_mat.to(device)

        # Buffer
        self.Theta_buf = nn.Parameter(torch.zeros((n_user, n_know)),
                                      requires_grad=False).to(device)
        self.Psi_buf = nn.Parameter(torch.zeros((n_item, n_know)),
                                      requires_grad=False).to(device)
        
        # monotonicity
        f_linear = nn.Linear if monotonicity_assumption is False else PosLinear


        '''
        Encoder
        '''
        self.f_nn = nn.Sequential(
            OrderedDict(
                [   
                    ('f_layer_1', nn.Linear(n_item, 512)),
                    ('f_activate_1', nn.Sigmoid()),
                    ('f_layer_2', nn.Linear(512, 256)),
                    ('f_activate_2', nn.Sigmoid()),
                    ('f_layer_3', nn.Linear(256, 128)),
                    ('f_activate_3', nn.Sigmoid()),
                    ('f_layer_4', nn.Linear(128, n_know))
                ]
            )
        ).to(device)

        self.g_nn = nn.Sequential(
            OrderedDict(
                [
                    ('g_layer_1', nn.Linear(n_user, 512)),
                    ('g_activate_1', nn.Sigmoid()),
                    ('g_layer_2', nn.Linear(512, 256)),
                    ('g_activate_2', nn.Sigmoid()),
                    ('g_layer_3', nn.Linear(256, 128)),
                    ('g_activate_3', nn.Sigmoid()),
                    ('g_layer_4', nn.Linear(128, n_know))
                ]
            )
        ).to(device)


        '''
        Decoder
        '''
        self.theta_agg_mat = nn.Linear(n_know, user_dim).to(device)
        self.psi_agg_mat = nn.Linear(n_know, item_dim).to(device)

        self.ncd = nn.Sequential(
            OrderedDict([
                ('pred_layer_1', f_linear(user_dim, 64)),
                ('pred_activate_1', nn.Sigmoid()),
                ('pred_dropout_1', nn.Dropout(p=0.5)),
                ('pred_layer_2', f_linear(64, 32)),
                ('pred_activate_2', nn.Sigmoid()),
                ('pred_dropout_2', nn.Dropout(p=0.5)),
                ('pred_layer_3', f_linear(32, 1)),
                ('pred_activate_3', nn.Sigmoid()),
            ])
        ).to(device)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)


    '''
    Encode
    '''
    def diagnose_theta(self, user_log):
        '''
        Encoding learner cognitive states from their logs.

        :param user_log: torch.Tensor [batch_size, n_item]
        '''
        theta = self.f_nn(user_log)
        return theta


    def diagnose_psi(self, item_log):
        '''
        Encoding item features from their logs.

        :param item_log: torch.Tensor [batch_size, n_user]
        '''
        psi = self.g_nn(item_log)
        return psi


    def encode(self, user_log, item_log):
        '''
        Encoding learner cognitive and item features from their logs simultaneously.

        :param user_log: torch.Tensor [batch_size, n_item]
        :param item_log: torch.Tensor [batch_size, n_user]
        '''
        theta = self.diagnose_theta(user_log)
        psi = self.diagnose_psi(item_log)
        return theta, psi
    

    '''
    Decode
    '''
    def ncd_func(self, theta, psi):
        '''
        Three layers MLP.

        :param theta: torch.Tensor [batch_size, user_dim]
        :param psi: torch.Tensor [batch_size, item_dim]
        :return: y_pred: torch.Tensor [batch_size, 1]
        '''
        assert(self.user_dim == self.item_dim)
        y_pred = self.ncd(theta - psi)
        return y_pred
    

    def decode(self, theta, psi, Q_batch):
        '''
        
        :param theta: torch.Tensor [batch_size, user_dim]
        :param psi: torch.Tensor [batch_size, item_dim]
        :return: output: torch.Tensor [batch_size, 1]
        '''
        theta_agg = self.theta_agg_mat(theta * Q_batch)
        theta_agg = torch.sigmoid(theta_agg)

        psi_agg = self.psi_agg_mat(psi * Q_batch)
        psi_agg = torch.sigmoid(psi_agg)

        output = self.itf(theta_agg, psi_agg)
        return output
    

    '''
    Buffer
    '''
    def get_Theta_buf(self):
        return self.Theta_buf.detach().cpu()
    
    def update_Theta_buf(self, theta_new, user_id):
        self.Theta_buf[user_id] = theta_new
    
    def get_Psi_buf(self):
        return self.Psi_buf.detach().cpu()

    def update_Psi_buf(self, psi_new, item_id):
        self.Psi_buf[item_id] = psi_new
    

    '''
    Forward
    '''
    def forward(self, user_log, item_log, user_id, item_id):
        '''
        This is a regular forward without using any data augmentation.
        '''
        # encode
        theta, psi = self.encode(user_log, item_log)

        # decode
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.decode(theta, psi, Q_batch)
        return output, theta, psi