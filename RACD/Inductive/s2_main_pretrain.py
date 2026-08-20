import gc
import numpy as np
import pandas as pd
import torch
import os
import math
import random
import time
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.Encoder_Decoder import EAD
from dataset import DRDataset, RACDataset, RACDataset_T
import argparse

data_name = "math1"
remain = 20
def parse_args(data_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='set seed', default=42)
    parser.add_argument('--data', help='the name of dataset', default=f'{data_name}')
    parser.add_argument('--train_file', help='the path of the train file',
                        default=f'./data/{data_name}/InducData/train.csv')
    parser.add_argument('--theta_file', help='the path of the NCDM_theta file',
                        default=f'./data/{data_name}/InducData/NCDM_theta.csv')
    parser.add_argument('--psi_file', help='the path of the NCDM_psi file',
                        default=f'./data/{data_name}/InducData/NCDM_psi.csv')
    parser.add_argument('--valid_train_file', help='the path of the valid file',
                        default=f'./data/{data_name}/InducData/val/val_20_in.csv')
    parser.add_argument('--valid_test_file', help='the path of the test file',
                        default=f'./data/{data_name}/InducData/val/val_20_out.csv')
    parser.add_argument('--test_train_file', help='the path of the test_train file',
                        default=f'./data/{data_name}/InducData/test/test_{remain}_in.csv')
    parser.add_argument('--test_test_file', help='the path of the test_test file',
                        default=f'./data/{data_name}/InducData/test/test_{remain}_out.csv')
    parser.add_argument('--Q_matrix', help='the path of the q-matrix',
                        default=f'./data/{data_name}/InducData/Q_mat.npy')

    parser.add_argument('--save_path', help='the save path of all results',
                        default=f'./RACD/Inductive/report/EAD_report/')
    parser.add_argument('--user_dim', help='the dimension of user vector', default=64)
    parser.add_argument('--item_dim', help='the dimension of item vector', default=64)
    parser.add_argument('--epoch', help='the training epoch', default=20)  # default=20
    parser.add_argument('--batch_size', help='the batch size in the training phase', default=256)
    parser.add_argument('--lr', help='the learning rate in the training phase', default=7e-4)
    parser.add_argument('--alpha', help='the hyperparameters to balance the distillation loss', default=1)
    parser.add_argument('--device', help='the running device. cpu or gpu', default='cuda:0')
    args = parser.parse_args()
    return args


# 对args再补充一些参数 
def add_args(args):
    args.model_name = f'/Pretrain_model/'
    return args


class AverageMeter(object):
    '''
    Computes and stores the average and current value
    
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def add_knowledge_code(data, Q_mat):
    '''
    transform Q_matrix to triplet data (log) style
    '''
    knowledge = []
    for i in range(data.shape[0]):
        knowledge.append(Q_mat[data.loc[i, 'item_id']])
    data['knowledge'] = knowledge
    return data


def get_eval_result(y_true, y_pred, y_pred_label, test=False):
    '''
    Compute the the acc, auc, rmse in Val_set or Test_set

    :param y_true: label
    :param y_pred: the prediction from model
    :param y_pred_label: transfrom y_pred to 1 or 0
    :param test: testset or not
    '''
    acc = accuracy_score(y_true, y_pred_label)
    auc = roc_auc_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    f1 = f1_score(y_true, y_pred_label)
    if test:
        print('test_acc = %.4f auc = %.4f rmse = %.4f f1 = %.4f ' % (acc, auc, rmse, f1))
        result = {'test_acc': acc, 'auc': auc, 'rmse': rmse, 'f1': f1}
    else:
        print('val_acc = %.4f auc = %.4f rmse = %.4f f1 = %.4f ' % (acc, auc, rmse, f1))
        result = {'val_acc': acc, 'auc': auc, 'rmse': rmse, 'f1': f1}
    return result


def eval(model, dataloader, df_data, test=False):
    '''
    Evaluaion
    
    :param dataloader: valid_loader or test_loader
    :param df_data: a pd.Dataframe
    :param test: testset or not
    '''
    model.eval()
    device = model.device
    eval_result = {}
    y_pred = []
    for i, (user_log, item_log, user_id, item_id, score) \
            in enumerate(dataloader):
        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        pred_1_batch = model.forward(user_log, item_log, user_id, item_id)[0].detach().cpu().tolist()
        y_pred += pred_1_batch
    y_pred = np.array(y_pred)
    y_true = df_data['score'].values.astype(int)
    y_plab = (y_pred > 0.5).astype(int)
    eval_result = get_eval_result(y_true, y_pred, y_plab, test)
    return eval_result


def train_one_epoch(model, optimizer, train_loader, trainset, epoch_now, alpha):
    result_epoch = {}
    model.train()
    device = model.device
    Lce_recorder = AverageMeter()
    Lkd_recorder = AverageMeter()
    mse_loss = nn.MSELoss()
    # kl_loss = kl_div_loss = nn.KLDivLoss(reduction="batchmean")

    pbar = tqdm(total=len(train_loader), desc='Epoch %d' % epoch_now)
    score_all = []
    pred_all = []

    for i, (user_log, item_log, user_id, item_id, score, theta_log, psi_log) in enumerate(train_loader):
        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        score = score.to(device)
        theta_log = theta_log.squeeze(1).to(device)
        psi_log = psi_log.squeeze(1).to(device)
        pred, theta, psi = model.forward(user_log, item_log, user_id, item_id)
        lce = F.binary_cross_entropy(pred, score)
        lkd = (mse_loss(F.normalize(theta, dim=1), F.normalize(theta_log, dim=1)) +
               mse_loss(F.normalize(psi, dim=1), F.normalize(psi_log, dim=1))) * alpha
        loss = lce + lkd
        Lce_recorder.update(lce.item(), n=user_log.size(0))
        Lkd_recorder.update(lkd.item(), n=user_log.size(0) * 2)
        score_all += score.detach().cpu().numpy().reshape(-1, ).tolist()
        pred_all += pred.detach().cpu().numpy().reshape(-1, ).tolist()

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        pbar.update(1)
    pbar.close()

    model.eval()
    # Update examinee traits
    for i in range(math.ceil(trainset.log_mat.shape[0] / args.batch_size)):
        idx = np.arange(i * args.batch_size, min(trainset.log_mat.shape[0], (i + 1) * args.batch_size))
        model.update_Theta_buf(model.diagnose_theta(torch.Tensor(trainset.log_mat[idx, :]) \
                                                    .to(device)).detach(), torch.LongTensor(idx))

    # Update question features
    for i in range(math.ceil(trainset.log_mat.shape[1] / args.batch_size)):
        idx = np.arange(i * args.batch_size, min(trainset.log_mat.shape[1], (i + 1) * args.batch_size))
        model.update_Psi_buf(model.diagnose_psi(torch.Tensor(trainset.log_mat[:, idx].T) \
                                                .to(device)).detach(), torch.LongTensor(idx))

    score_all = np.array(score_all)
    pred_all = np.array(pred_all)
    train_acc = accuracy_score(score_all, pred_all > 0.5)
    Loss_ce = Lce_recorder.avg
    Loss_kd = Lkd_recorder.avg
    print('epoch = %d, train_acc = %.4f, Loss_ce = %.4f, Loss_kd = %.4f' % (epoch_now, train_acc, Loss_ce, Loss_kd))
    model.train()
    result_epoch['train_acc'] = train_acc
    result_epoch['Lce'] = Loss_ce
    result_epoch['Lkd'] = Loss_kd
    return result_epoch


def train(model, optimizer, trainloader, trainset, validloader, df_valid, exp_path, max_epoch, alpha):
    since = time.time()
    best_acc = -1
    bestEpoch = 0
    f = open(os.path.join(exp_path, "record.txt"), "w")

    for epoch in range(max_epoch):
        result_per_epoch = train_one_epoch(model, optimizer, trainloader, trainset, epoch + 1, alpha)
        eval_result = eval(model, validloader, df_valid)

        if eval_result['val_acc'] > best_acc:
            bestEpoch = epoch + 1
            best_acc = eval_result['val_acc']
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=eval_result['val_acc'])
            name = os.path.join(exp_path, "best.pth")
            torch.save(state_dict, name)
            theta = pd.DataFrame(model.Theta_buf.detach().cpu().numpy())
            theta_path = os.path.join(exp_path, "theta.csv")
            theta.to_csv(theta_path, index=False)

        msg = "epoch:{} train_acc:{:.4f} Lce:{:.4f}  Lkd:{:.4f}\n val_acc:{:.4f} auc:{:.4f} rmse:{:.4f} f1:{:.4f} \n".format(
            epoch + 1,
            result_per_epoch['train_acc'],
            result_per_epoch['Lce'],
            result_per_epoch['Lkd'],
            eval_result['val_acc'],
            eval_result['auc'],
            eval_result['rmse'],
            eval_result['f1']
        )

        f.write(msg)
        f.flush()

    msg_best = "model:{} best epoch {} val_acc:{:.4f} \n".format('EAD', bestEpoch, best_acc)
    time_elapsed = "traninng time: {:.2f} s \n".format(time.time() - since)
    print(msg_best)
    print(time_elapsed)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()


def calculate(test_acc_all, test_auc_all, test_rmse_all, test_f1_all, args, remain):
    # record_test_average = 'Summarize: \n test_acc = %.4f ± %.4f \n test_auc = %.4f ± %.4f \n test_rmse = %.4f ± %.4f \n test_f1 = %.4f ± %.4f' % (
    averageAccuracy = np.mean(test_acc_all) * 100
    stdAcc = np.std(test_acc_all) * 100
    averageAuc = np.mean(test_auc_all) * 100
    stdAuc = np.std(test_auc_all) * 100
    averageRmse = np.mean(test_rmse_all) * 100
    stdRmse = np.std(test_rmse_all) * 100
    averageF1 = np.mean(test_f1_all) * 100
    stdF1 = np.std(test_f1_all) * 100
    # 保留两位小数
    averageAccuracy = round(averageAccuracy, 2)
    stdAcc = round(stdAcc, 2)
    averageAuc = round(averageAuc, 2)
    stdAuc = round(stdAuc, 2)
    averageRmse = round(averageRmse, 2)
    stdRmse = round(stdRmse, 2)
    averageF1 = round(averageF1, 2)
    stdF1 = round(stdF1, 2)
    endText = f"  ============= dataset is {args.data} ============= \n" + \
              f"  ============= remain rate is {remain}% ============= \n" + \
              f"  ============= [the Best Epoch] test ============= \n" + \
              f"  =======  Average acc: {averageAccuracy}, std: {stdAcc} =======\n" + \
              f"  =======  Average auc: {averageAuc}, std: {stdAuc} =======\n" + \
              f"  ======= Average rmse: {averageRmse}, std: {stdRmse} =======\n" + \
              f"  =======   Average f1: {averageF1}, std: {stdF1} =======\n" + \
              f"  =================================================\n"
    return endText


if __name__ == '__main__':
    # model parameter
    args = parse_args(data_name)
    args = add_args(args)
    # traning parameter
    times = 5
    user_dim = int(args.user_dim)
    item_dim = int(args.item_dim)
    epoch = int(args.epoch)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    alpha = float(args.alpha)
    device = torch.device(args.device)

    # base dataset
    df_train = pd.read_csv(args.train_file)
    df_valid_train = pd.read_csv(args.valid_train_file)
    df_valid_test = pd.read_csv(args.valid_test_file)
    df_theta = pd.read_csv(args.theta_file)
    df_test_train = pd.read_csv(args.test_train_file)
    df_test_test = pd.read_csv(args.test_test_file)
    df_psi = pd.read_csv(args.psi_file)
    Q_mat = np.load(args.Q_matrix)

    n_user = len(df_train['user_id'].unique())
    n_item = np.max([np.max(df_train['item_id']), np.max(df_valid_train['item_id']),
                     np.max(df_valid_test['item_id']), np.max(df_test_train['item_id']),
                     np.max(df_test_test['item_id'])]) + 1
    n_know = Q_mat.shape[1]
    n_user_new0 = len(df_valid_train['user_id'].unique())
    n_user_new = len(df_test_train['user_id'].unique())

    df_train = add_knowledge_code(df_train, Q_mat)
    df_valid_train = add_knowledge_code(df_valid_train, Q_mat)
    df_valid_test = add_knowledge_code(df_valid_test, Q_mat)

    trainset = DRDataset(df_train, n_user, n_item, df_theta, df_psi)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    valid_trainset = RACDataset(df_valid_train, n_user_new0, n_item)
    valid_testset = RACDataset(df_valid_test, n_user_new0, n_item, valid_type='inductive')
    valid_testset.log_mat_train = trainset.log_mat
    valid_testset.log_mat = valid_trainset.log_mat.copy()
    valid_testloader = DataLoader(dataset=valid_testset, batch_size=batch_size, shuffle=False)

    # test
    df_test_train = add_knowledge_code(df_test_train, Q_mat)
    df_test_test = add_knowledge_code(df_test_test, Q_mat)
    test_trainset = RACDataset(df_test_train, n_user_new, n_item)
    test_testset = RACDataset(df_test_test, n_user_new, n_item, valid_type='inductive')
    test_testset.log_mat_train = trainset.log_mat
    test_testset.log_mat = test_trainset.log_mat.copy()
    test_testloader = DataLoader(dataset=test_testset, batch_size=batch_size, shuffle=False)

    #training
    accuracy_all, auc_all, rmse_all, f1_all = [], [], [], []
    out_path = "{}/{}/{}/".format(args.save_path, args.data, args.model_name)
    for i in range(times):
        #save text
        exp_path = "{}/{}/{}/{}/".format(args.save_path, args.data, args.model_name, str(i + 1))
        os.makedirs(exp_path, exist_ok=True)

        #set seed
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)

        print(
            f'================================================Run {i + 1}================================================')
        model = EAD(n_user, n_item, n_know, user_dim, item_dim,
                    Q_mat=Q_mat, monotonicity_assumption=True, device=device)
        optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)
        train(model, optimizer, trainloader, trainset, valid_testloader, df_valid_test, exp_path, epoch, alpha)
        # 导入模型进行测试
        model.load_state_dict(torch.load(os.path.join(exp_path, "best.pth"), weights_only=False)['model'])

        for remain in range(10, 100, 10):
            args.test_train_file = f'./data/{data_name}/InducData/test/test_{remain}_in.csv'
            args.test_test_file = f'./data/{data_name}/InducData/test/test_{remain}_out.csv'
            df_test_train = pd.read_csv(args.test_train_file)
            df_test_test = pd.read_csv(args.test_test_file)
            df_test_train = add_knowledge_code(df_test_train, Q_mat)
            df_test_test = add_knowledge_code(df_test_test, Q_mat)
            test_trainset = RACDataset(df_test_train, n_user_new, n_item)
            test_testset = RACDataset(df_test_test, n_user_new, n_item, valid_type='inductive')
            test_testset.log_mat_train = trainset.log_mat
            test_testset.log_mat = test_trainset.log_mat.copy()
            test_testloader = DataLoader(dataset=test_testset, batch_size=batch_size, shuffle=False)

            test_result = eval(model, test_testloader, df_test_test, test=True)
            accuracy_all.append(test_result['test_acc'])
            auc_all.append(test_result['auc'])
            rmse_all.append(test_result['rmse'])
            f1_all.append(test_result['f1'])

    # 做多少次remain time
    remain_time = 9
    for k in range(0, remain_time):
        accuracy_all1 = [accuracy_all[j] for j in range(k, times * remain_time, remain_time)]
        auc_all1 = [auc_all[j] for j in range(k, times * remain_time, remain_time)]
        rmse_all1 = [rmse_all[j] for j in range(k, times * remain_time, remain_time)]
        f1_all1 = [f1_all[j] for j in range(k, times * remain_time, remain_time)]
        endText = calculate(accuracy_all1, auc_all1, rmse_all1, f1_all1, args, remain=(k + 1) * 10)
        print(endText)
        with open(os.path.join(out_path, "test_performance.txt"), 'a') as f:
            f.write(endText)
            f.write("\n")
            f.close()

    with open(os.path.join(out_path, "test_performance.txt"), 'a') as f:
        f.write('\n')
        f.write("model:{} \n dataset:{} \n".format('EADRCD', data_name))
        f.close()

    gc.collect()
