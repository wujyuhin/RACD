# coding: utf-8
# 2021/6/21 @ tongshiwei

import logging
import numpy as np
import torch
from torchmetrics.functional import f1_score

from EduCDM import CDM
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, f1_score
import torch.autograd as autograd
import torch.nn.functional as F
import time
import os

class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINANet(DINANet):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)


class DINA(CDM):
    def __init__(self, user_num, item_num, hidden_dim, ste=False):
        super(DINA, self).__init__()
        if ste:
            self.dina_net = STEDINANet(user_num, item_num, hidden_dim)
        else:
            self.dina_net = DINANet(user_num, item_num, hidden_dim)

    def train(self, train_data, test_data=None, *,epoch: int, device="cpu", lr=0.001,save_dir) -> ...:
        since = time.time()
        self.dina_net = self.dina_net.to(device)
        loss_function = nn.BCELoss()
        trainer = torch.optim.Adam(self.dina_net.parameters(), lr)
        f = open(os.path.join(save_dir, "record.txt"), "w")
        bestAccuracy,bestEpoch = 0,0
        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s   " % e):
                user_id, item_id, knowledge, response = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge: torch.Tensor = knowledge.to(device)
                predicted_response: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            # train acc
            accuracy_train,auc_train,rmse_train,f1_train = self.eval(train_data, device=device)
            # print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))
            if test_data is not None:
                accuracy, auc, rmse,f1 = self.eval(test_data, device=device)
                # model selection
                if accuracy > bestAccuracy:
                    bestEpoch = e + 1
                    bestAccuracy = accuracy
                    if save_dir is not None:
                        self.save(os.path.join(save_dir, 'best_dina.pth'))
                    else:
                        self.save('best_dina.pth')
                print("[Epoch %d] train_acc: %.6f, val_acc: %.6f, auc: %.6f, rmse: %.6f, f1:%.6f" % (e, accuracy_train, accuracy,auc,rmse,f1))

            msg = "epoch:{} train_acc:{:.4f} val_acc:{:.4f} auc:{:.4f} rmse:{:.4f} f1:{:.4f}\n".format(
                e + 1,accuracy_train,accuracy,auc,rmse,f1)
            f.write(msg)
            f.flush()
        msg_best = "model:{} best epoch:{} val_acc:{:.4f} \n".format('DINA',bestEpoch, bestAccuracy)
        time_elapsed = "training time: {:.2f} s \n".format(time.time() - since)
        print(msg_best)
        print(time_elapsed)
        f.write(msg_best)
        f.write(time_elapsed)
        f.close()

    def eval(self, test_data, device="cpu") -> tuple:
        self.dina_net = self.dina_net.to(device)
        self.dina_net.eval()
        y_pred = []
        y_true = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, knowledge, response = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge: torch.Tensor = knowledge.to(device)
            pred: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())

        self.dina_net.train()
        # calculate the accuracy, AUC, Rmse
        return (accuracy_score(y_true, np.array(y_pred) >= 0.5),
                roc_auc_score(y_true, y_pred),
                np.sqrt(mean_squared_error(y_true, y_pred)),
                f1_score(y_true, np.array(y_pred) >= 0.5))

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        # logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dina_net.load_state_dict(torch.load(filepath))
        # logging.info("load parameters from %s" % filepath)
