"""
这是测试文件，将我们实验数据放入其中进行测试，修改他们的data_loader.py文件，使其能够读取我们的数据
"""
from distutils.command.config import config

import torch
import json
import sys
import pandas as pd
import numpy as np
from sympy.stats.sampling.sample_numpy import numpy


class TrainDataLoader(object):
    '''
    data_triple loader for training
    '''

    def __init__(self):
        self.batch_size = 256
        self.ptr = 0
        self.data = []

        data_file = '../data/junyi_more50/TransData/train.csv'
        q_file = '../data/junyi_more50/TransData/Q_mat.npy'
        config_file = '../data/junyi_more50/TransData/config.txt'
        self.data = pd.read_csv(data_file)
        self.q_data = np.load(q_file)
        # data_file = '../data/junyi/train_set.json'
        # config_file = 'config.txt'
        # with open(data_file, encoding='utf8') as i_f:
        #     self.data = json.load(i_f)

        with open(config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data.iloc[self.ptr + count]  # log = self.data[self.ptr + count]
            # 下面已经改成q矩阵
            # knowledge_emb = [0.] * self.knowledge_dim
            # for knowledge_code in log['knowledge_code']:
            #     knowledge_emb[knowledge_code - 1] = 1.0
            knowledge_emb = list(self.q_data[log['item_id']])  # our q matrix exer_id is from 0

            y = log['score']  # y = log['score']
            input_stu_ids.append(log['user_id'])  # input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['item_id'])  # input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(
            input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


if __name__ == '__main__':
    data_loader = TrainDataLoader()
    input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
    # data_file = '../data/junyi/junyi/TransData/train.csv'
