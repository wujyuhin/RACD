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
from utils.hamming import find_topk_similar_hashes_batch,find_topk_similar_hashes_batch_custom, find_topk_similar_theta_batch


class hash(Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def hash_layer(input):
    return hash.apply(input)


class PosLinear(nn.Linear):
    def forward(self, input):
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class RCD(nn.Module):
    def __init__(self, n_user, n_item, n_know, user_dim, item_dim, Q_mat, Master_matrix, bits, k, device,
                 train, monotonicity_assumption = True):
        '''
        :param n_user: int, the number of learners.
        :param n_item: int, the number of test items.
        :param n_know: int, the number of knowledge concepts.
        :param user_dim: int, the dimension of mixed user representations.
        :param item_dim: int, the dimension of mixed item representations.
        :param Q_mat: np.array [n_item,n_know], the binary Q-matrix.
        :param bits: int, the number of bits in hashing.
        :param k: int, the number of samples retrieved from hashing used for aggregation.
        :param device: torch.device
        :param monotonicity_assumption: bool, whether to apply
                the monotonicity assumption to the diagnostic module. If True,
                the monotonicity assumption is applied.
        '''
        super(RCD,self).__init__()
        self.n_user = n_user 
        self.n_item = n_item 
        self.n_know = n_know
        self.user_dim = user_dim 
        self.item_dim = item_dim
        self.itf = self.ncd_func
        self.device = device
        self.k = k

        self.Q_mat = torch.Tensor(Q_mat) if Q_mat is not None else torch.ones((n_item, n_know))
        self.Q_mat = self.Q_mat.to(device)
        if train:
            self.Master_matrix = torch.Tensor(Master_matrix).to(device)

        # Buffer of student traits
        self.Theta_buf = nn.Parameter(torch.zeros((n_user, n_know)),
                                      requires_grad=False).to(device)
        # self.agg_Theta_buf =
        
        # Buffer of student hashes
        self.Hash_buf = nn.Parameter(torch.zeros((n_user, bits)),
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
        Hashing module
        '''
        self.hashing = nn.Sequential(
            OrderedDict(
                [
                    ('h_layer_1', nn.Linear(n_know, 512)),
                    ('h_activate_1', nn.ReLU()),
                    ('h_layer_2', nn.Linear(512, bits)),
                    # ('h_activate_2', nn.ReLU()),
                    # ('h_layer_3', nn.Linear(64, bits)),
                ]
            )
        ).to(device)


        '''
        Decoder
        '''
        self.theta_agg_mat = PosLinear(n_know, user_dim).to(device)
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
    Hashing
    '''
    def truncated_normal_sample(self, mean, std, a, b):
        """
        Generate samples from a normal distribution and clip them to the range [a, b].

        :param mean: torch.Tensor, the mean of the normal distribution
        :param std: torch.Tensor, the standard deviation of the normal distribution
        :param a: float, the lower bound of the range
        :param b: float, the upper bound of the range
        :return: torch.Tensor, the clipped samples
        """
        samples = Normal(mean, std).sample()
        samples = torch.clamp(samples, a, b)
        return samples

    def skill_shake(self, theta, master_matrix, p=0.5):
        '''
        A data augmentation method that allows the student feature vectors to shake within a reasonable range.

        :param theta: torch.Tensor [batch_size, n_know]
        :param master_matrix: torch.Tensor [n_know, 8]
        :param p: float the probability of elements shaking 
        :return: theta_i [batch_size, n_know], theta_j [batch_size, n_know]
        '''
        b, n_know = theta.shape
        
        # Create a mask for which elements to shake
        mask = torch.rand(b, n_know) < p
        mask = mask.to(self.device)
        
        # Extract the intervals
        lower1 = master_matrix[:, 0].unsqueeze(0).expand(b, -1)
        upper1 = master_matrix[:, 1].unsqueeze(0).expand(b, -1)
        lower2 = master_matrix[:, 2].unsqueeze(0).expand(b, -1)
        upper2 = master_matrix[:, 3].unsqueeze(0).expand(b, -1)
        
        # Extract the means and standard deviations for the normal distributions
        mean1 = master_matrix[:, 4].unsqueeze(0).expand(b, -1)
        std1 = master_matrix[:, 6].unsqueeze(0).expand(b, -1)
        mean2 = master_matrix[:, 5].unsqueeze(0).expand(b, -1)
        std2 = master_matrix[:, 7].unsqueeze(0).expand(b, -1)
        
        # Determine which interval each element falls into
        in_interval1 = (theta >= lower1) & (theta <= upper1)
        in_interval2 = (theta >= lower2) & (theta <= upper2)
        
        # Sample from the truncated normal distributions
        sampled_theta_i = torch.where(in_interval1, self.truncated_normal_sample(mean1, std1, lower1, upper2), theta)
        sampled_theta_i = torch.where(in_interval2, self.truncated_normal_sample(mean2, std2, lower1, upper2), sampled_theta_i).to(self.device)
        
        sampled_theta_j = torch.where(in_interval1, self.truncated_normal_sample(mean1, std1, lower1, upper2), theta)
        sampled_theta_j = torch.where(in_interval2, self.truncated_normal_sample(mean2, std2, lower1, upper2), sampled_theta_j).to(self.device)
        
        # Apply the mask to decide which elements to shake
        theta_i = torch.where(mask, sampled_theta_i, theta)  # theta_i = theta
        theta_j = torch.where(mask, sampled_theta_j, theta)
        
        return theta_i, theta_j
    

    def get_hashing_ij(self, theta, Master_matrix):
        '''
        Two variants of theta are obtained by skill_shake, 
        and ultimately, the corresponding two hash codes are generated.

        :param theta: torch.Tensor [batch_size, n_know]
        :return: stu_hashing_i [batch_size, bits], stu_hashing_j [batch_size, bits]
        '''
        theta_i, theta_j = self.skill_shake(theta, Master_matrix)

        prob_i = torch.sigmoid(self.hashing(theta_i))
        stu_hashing_i = hash_layer(prob_i - 0.5)

        prob_j = torch.sigmoid(self.hashing(theta_j))
        stu_hashing_j = hash_layer(prob_j - 0.5)

        return stu_hashing_i, stu_hashing_j


    def get_hashing(self, theta):
        '''
        From theta to hashing.

        :param theta: torch.Tensor [batch_size, n_know]
        :return: stu_hashing [batch_size, bits]
        '''
        prob = torch.sigmoid(self.hashing(theta))
        stu_hashing = hash_layer(prob - 0.5)

        return stu_hashing
    

    def get_hashing_bylog(self, user_log):
        '''
        From user_log to hashing.

        :param user_log: torch.Tensor [batch_size, n_item]
        :return: stu_hashing [batch_size, bits]
        '''
        theta = self.diagnose_theta(user_log)
        prob = torch.sigmoid(self.hashing(theta))
        stu_hashing = hash_layer(prob - 0.5)

        return stu_hashing
    
    
    def agg_by_hash(self, hashes, k):
        """
        Aggregate the theta by hashing.

        :param hashes: torch.Tensor [batch_size, n_know]
        :param k: number of top similar hashes to return
        :return: agg_theta [batch_size, n_know]
        """
        # get current database_hashes and database_theta
        DataHashes = self.Hash_buf.detach()
        Datatheta = self.Theta_buf.detach()

        # get indices, distances, softmax weights of topk neighbors
        # topk_indices, topk_weights = find_topk_similar_hashes_batch_custom(hashes, DataHashes, k, distance_metric='euclidean')
        topk_indices, topk_weights = find_topk_similar_hashes_batch(hashes, DataHashes, k)
        # Gather the topk theta values
        topk_theta = Datatheta[topk_indices]  # [b, k, n_know]
        
        # Compute the weighted sum
        agg_theta = torch.sum(topk_theta * topk_weights.unsqueeze(-1), dim=1)  # [b, n_know]
        
        return agg_theta

    # 现在不用hash的方法，是直接对self.Theta_buf进行计算与当前输入的theta最接近的k个Theta_buf进行聚合
    def agg_by_theta(self, theta, k):
        """
        Aggregate the theta by theta.

        :param theta: torch.Tensor [batch_size, n_know]
        :param k: number of top similar hashes to return
        :return: agg_theta [batch_size, n_know]
        """
        # get current database_theta
        DataTheta = self.Theta_buf.detach()

        # get indices, distances, softmax weights of topk neighbors
        topk_indices, topk_weights = find_topk_similar_theta_batch(theta, DataTheta, k, distance_metric='euclidean')

        # Gather the topk theta values
        topk_theta = DataTheta[topk_indices]

        # Compute the weighted sum
        agg_theta = torch.sum(topk_theta * topk_weights.unsqueeze(-1), dim=1)  # [b, n_know]
        return agg_theta



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
    def update_Theta_buf(self, theta_new, user_id):
        self.Theta_buf[user_id] = theta_new

    def update_Hash_buf(self, hash_new, user_id):
        self.Hash_buf[user_id] = hash_new

    def get_Theta_buf(self):
        return self.Theta_buf.detach().cpu()
    
    def get_Hash_buf(self):
        return self.Hash_buf.detach().cpu()
    

    '''
    Forward
    '''
    def train_forward(self, user_log, item_log, user_id, item_id):
        '''
        During the training period,
        samples will be transformed into two variants,
        which is different from direct forward propagation;
        therefore, this function is written separately.
        '''
        # encode
        theta, psi = self.encode(user_log, item_log)

        # hash and aggregate
        # theta_i_hash_code, theta_j_hash_code = self.get_hashing_ij(theta, self.Master_matrix)
        # agg_theta_i = self.agg_by_hash(theta_i_hash_code, self.k)
        # agg_theta_j = self.agg_by_hash(theta_j_hash_code, self.k)
        # aggregate
        weight = 0.5  # weight = 1 只用检索的theta融合
        theta_i_hash_code, theta_j_hash_code = self.get_hashing_ij(theta, self.Master_matrix)
        agg_theta_i = self.agg_by_hash(theta_i_hash_code, self.k)*weight+theta*(1-weight)
        agg_theta_j = self.agg_by_hash(theta_j_hash_code, self.k)*weight+theta*(1-weight)

        # decode
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output_i = self.decode(agg_theta_i, psi, Q_batch)
        output_j = self.decode(agg_theta_j, psi, Q_batch)

        return output_i, output_j, theta_i_hash_code, theta_j_hash_code



    def forward(self, user_log, item_log, user_id, item_id):
        '''
        This is a regular forward without using any data augmentation.
        '''
        # encode
        theta, psi = self.encode(user_log, item_log)

        # hash and aggregate
        theta_hash_code = self.get_hashing(theta)
        # agg_theta = self.agg_by_hash(theta_hash_code, self.k)

        # aggregate
        weight = 0.5  # weight = 1 只用检索的theta融合
        agg_theta = self.agg_by_hash(theta_hash_code, self.k)*weight+theta*(1-weight)

        # decode
        Q_batch = self.Q_mat[item_id].squeeze(dim=1)
        output = self.decode(agg_theta, psi, Q_batch)
        return output


    # def forward(self, user_log, item_log, user_id, item_id):
    #     '''
    #     This is a regular forward without using any data augmentation.
    #     '''
    #     # encode
    #     theta, psi = self.encode(user_log, item_log)
    #
    #     # hash and aggregate
    #     # theta_hash_code = self.get_hashing(theta)
    #     weight = 0.5
    #     agg_theta = self.agg_by_theta(theta, self.k) * weight + theta * (1 - weight)
    #
    #     # decode
    #     Q_batch = self.Q_mat[item_id].squeeze(dim=1)
    #     output = self.decode(agg_theta, psi, Q_batch)
    #     return output