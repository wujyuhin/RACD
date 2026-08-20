# coding: utf-8
# 2021/4/1 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error,f1_score
from EduCDM import CDM
import os
import time

class EarlyStopping:
    def __init__(self, patience=8, delta=0, mode='min', verbose=False):
        assert mode in ['min', 'max']
        
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.best_score = np.Inf if mode == 'min' else -np.Inf

    def __call__(self, metric):
        """
        每次验证后调用此方法
        Returns:
            bool: 是否应该停止训练
        """
        if self.mode == 'min':
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self._update_best(score)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: Counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self._update_best(score)
            self.counter = 0

        return self.early_stop

    def _update_best(self, score):
        self.best_score = score
        if self.verbose:
            print(f'Validation metric improved ({self.mode} mode)')

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_difficulty = nn.Embedding(self.exer_n, 1)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, save_dir=None, early_stop=True):
        if early_stop == True:
            early_stopping = EarlyStopping(patience=5, delta=1e-4, mode='max', verbose=True)

        since = time.time()
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        f = open(os.path.join(save_dir, "record.txt"), "w")
        bestAccuracy,bestEpoch = 0,0
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s   " % epoch_i):
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.mean().item())

            # print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))
            accuracy_train, auc_train, rmse_train,f1_train = self.eval(train_data, device=device)
            if test_data is not None:
                accuracy, auc, rmse,f1 = self.eval(test_data, device=device)
                if accuracy > bestAccuracy:
                    bestEpoch = epoch_i +1
                    bestAccuracy = accuracy
                    if save_dir is not None:
                        self.save(os.path.join(save_dir, 'NCDM.pth'))
                    else:
                        self.save('NCDM.pth')
                print("[Epoch %d] train_acc: %.6f,val_acc: %.6f, auc: %.6f, rmse: %.6f, f1:%.6f" % (epoch_i,accuracy_train, accuracy, auc, rmse,f1))

            msg = "epoch:{} train_acc:{:.4f} val_acc:{:.4f} auc:{:.4f} rmse:{:.4f} f1:{:.4f} \n".format(
                epoch_i + 1, accuracy_train, accuracy, auc, rmse,f1)
            f.write(msg)
            f.flush()

            if early_stop and test_data is not None and early_stopping(accuracy):
                print(f'Early stopping triggered at epoch {epoch_i + 1}')
                break
        msg_best = "model:{} best epoch:{} val_acc:{:.4f} \n".format('NCDM', bestEpoch, bestAccuracy)
        time_elapsed = "training time: {:.2f} s \n".format(time.time() - since)
        print(msg_best)
        print(time_elapsed)
        f.write(msg_best)
        f.write(time_elapsed)
        # f.close()

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return (accuracy_score(y_true, np.array(y_pred) >= 0.5),
                roc_auc_score(y_true, y_pred),
                np.sqrt(mean_squared_error(y_true, y_pred)),
                f1_score(y_true, np.array(y_pred) >= 0.5))

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        # logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        # logging.info("load parameters from %s" % filepath)
