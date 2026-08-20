import gc

import numpy as np
import pandas as pd
import torch
import os
import math
import random
import time

# from lightgbm import early_stopping
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, f1_score
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
# from examples.IRT.GD.IRT import accuracy
# from model_parser import parse_args
from model.RCD import RCD
from dataset import RACDataset
from utils.CLoss import NtXentLoss, InfoNCELoss
from utils.getMasterMatrix import statistics
import argparse

# math1
data_name = 'math1'
opt = 4  # random pick the pretrain model
runtimes = 5

def parse_args(data_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--times', help='the number of runs', default=runtimes)
    parser.add_argument('--seed', help='set seed', default=42)
    parser.add_argument('--device', help='the running device. cpu or gpu', default='cuda:0')
    parser.add_argument('--save_path', help='the save path of all results',
                        default='./RACD/Inductive/report/RACD_report/')
    parser.add_argument('--data', help='the name of dataset', default=f'{data_name}_Induc')
    parser.add_argument('--model_name', help='the name of model',default=f'/finetune_model/')
    parser.add_argument('--pretrain_model_path', help='the path of the weights of pertained model',
                        default=f'./RACD/Inductive/report/EAD_report/{data_name}/pretrain_model/{opt}/')
    parser.add_argument('--train_file', help='the path of the train file',default=f'./data/{data_name}/InducData/train.csv')
    parser.add_argument('--valid_train_file', help='the path of the valid file',default=f'./data/{data_name}/InducData/val/val_20_in.csv')
    parser.add_argument('--valid_test_file', help='the path of the valid file', default=f'./data/{data_name}/InducData/val/val_20_out.csv')
    parser.add_argument('--test_train_file', help='the path of the test file', default=f'./data/{data_name}/InducData/test/test_20_in.csv')
    parser.add_argument('--test_test_file', help='the path of the test file', default=f'./data/{data_name}/InducData/test/test_20_out.csv')
    parser.add_argument('--Q_matrix', help='the path of the q-matrix', default=f'./data/{data_name}/InducData/Q_mat.npy')
    parser.add_argument('--user_dim', help='the dimension of user vector', default=64)
    parser.add_argument('--item_dim', help='the dimension of item vector', default=64)
    parser.add_argument('--epoch', help='the training epoch', default=40)  # 40
    parser.add_argument('--change_epoch', help='the epoch of changing unsupervised to supervise', default=30)  #30
    parser.add_argument('--batch_size', help='the batch size in the training phase', default=512)
    parser.add_argument('--lr', help='the learning rate in the training phase', default=7e-4)
    parser.add_argument('--bits', help='the number of bits in hashing', default=16)
    parser.add_argument('--k', help='the number of samples retrieved from hashing used for aggregation',default=20)  # 20 better
    parser.add_argument('--T', help='the temperature in the contrasive learning', default=0.5)
    parser.add_argument('--weight', help='the weight of the contrastive loss', default=1)

    args = parser.parse_args()
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
        try:
            pred_1_batch = model.forward(user_log, item_log, user_id, item_id).detach().cpu().tolist()
        except:
            pred_1_batch = model.forward(user_log, item_log, user_id, item_id).detach().cpu().tolist()
        y_pred += pred_1_batch
    y_pred = np.array(y_pred)
    y_true = df_data['score'].values.astype(int)
    y_plab = (y_pred > 0.5).astype(int)
    eval_result = get_eval_result(y_true, y_pred, y_plab, test)
    return eval_result


def train_one_epoch(model, optimizer_hash, optimizer_decoder, train_loader, trainset,
                    epoch_now, weight, change_epoch):
    result_epoch = {}
    model.train()
    device = model.device
    contrasive_loss = NtXentLoss(batch_size, temperature)
    # contrasive_loss = InfoNCELoss(batch_size, temperature)

    Lce_recorder = AverageMeter()
    Lcontra_recorder = AverageMeter()

    pbar = tqdm(total=len(train_loader), desc='Epoch %d' % epoch_now)
    score_all = []
    pred_all = []

    for i, (user_log, item_log, user_id, item_id, score) in enumerate(train_loader):

        user_log = user_log.to(device)
        item_log = item_log.to(device)
        user_id = user_id.to(device)
        item_id = item_id.to(device)
        score = score.to(device)

        if epoch_now > change_epoch:
            pred = model.forward(user_log, item_log, user_id, item_id)
            score_all += score.detach().cpu().numpy().reshape(-1, ).tolist()
            pred_all += pred.detach().cpu().numpy().reshape(-1, ).tolist()
            Lce = F.binary_cross_entropy(pred, score)
            Lce_recorder.update(Lce.item(), n=user_log.size(0))
            optimizer_decoder.zero_grad()
            Lce.backward(retain_graph=True)
            optimizer_decoder.step()

        else:
            pred_i, pred_j, theta_i_hash_code, theta_j_hash_code = model.train_forward(user_log, item_log,
                                                                                       user_id, item_id)
            score_all += score.detach().cpu().numpy().reshape(-1, ).tolist()
            score_all += score.detach().cpu().numpy().reshape(-1, ).tolist()
            pred_all += pred_i.detach().cpu().numpy().reshape(-1, ).tolist()
            pred_all += pred_j.detach().cpu().numpy().reshape(-1, ).tolist()
            Lcontra = weight * contrasive_loss(theta_i_hash_code, theta_j_hash_code, device)
            Lcontra_recorder.update(Lcontra.item(), n=user_log.size(0) * 2)
            # Lcontra = weight*contrasive_loss(theta_i_hash_code, theta_j_hash_code, device)
            # Lcontra_recorder.update(Lcontra.item(), n=user_log.size(0))
            optimizer_hash.zero_grad()
            Lcontra.backward(retain_graph=True)
            optimizer_hash.step()

        # Update student Hash
        model.eval()
        for i in range(math.ceil(trainset.log_mat.shape[0] / batch_size)):
            idx = np.arange(i * batch_size, min(trainset.log_mat.shape[0] \
                                                , (i + 1) * batch_size))
            model.update_Hash_buf(model.get_hashing_bylog( \
                torch.Tensor(trainset.log_mat[idx, :]) \
                    .to(device)).detach(), torch.LongTensor(idx))
        model.train()

        pbar.update(1)
    pbar.close()

    # Update student traits
    model.eval()
    for i in range(math.ceil(trainset.log_mat.shape[0] / batch_size)):
        idx = np.arange(i * batch_size, min(trainset.log_mat.shape[0] \
                                            , (i + 1) * batch_size))
        model.update_Theta_buf(model.diagnose_theta( \
            torch.Tensor(trainset.log_mat[idx, :]) \
                .to(device)).detach(), torch.LongTensor(idx))

    if epoch_now > change_epoch:
        Loss_ce = Lce_recorder.avg
        result_epoch['Lce'] = Loss_ce
        score_all = np.array(score_all)
        pred_all = np.array(pred_all)
        train_acc = accuracy_score(score_all, pred_all > 0.5)
        # print('epoch = %d, train_acc = %.4f, Lce = %.4f'%(epoch_now, train_acc, Loss_ce))

    else:
        Loss_contra = Lcontra_recorder.avg
        result_epoch['Lcontra'] = Loss_contra
        score_all = np.array(score_all)
        pred_all = np.array(pred_all)
        train_acc = accuracy_score(score_all, pred_all > 0.5)
        # print('epoch = %d, train_acc = %.4f, Lcontra = %.4f'%(epoch_now, train_acc, Loss_contra))

    model.train()
    result_epoch['train_acc'] = train_acc
    return result_epoch


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience  # 耐心值，即允许验证损失不再下降的最大轮数，在我们蒸馏任务中，看val_acc不再增加的最大轮数
        self.min_delta = 0  # 最小变化量，只有当损失变化超过这个值时才认为有改进
        self.restore_best_weights = restore_best_weights  # 是否恢复到最好的权重
        self.best_val_acc = None  # 最佳验证损失
        self.counter = 0  # 记录验证损失不再下降的轮数,包括最大值自己
        self.best_model = None  # 记录最佳模型的权重

    def __call__(self, val_acc, model):
        if self.best_val_acc is None:
            # 第一次调用时初始化最佳val_acc
            self.best_val_acc = val_acc
            self.best_model = model.state_dict()
        elif self.best_val_acc - val_acc < self.min_delta:
            # 如果当前验证val_acc比最佳损失好，并且超过了最小变化量
            self.best_val_acc = val_acc
            self.best_model = model.state_dict()
            self.counter = 0  # 重置计数器
        else:
            # 如果当前验证val_acc没有显著改善
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, best val_acc: {self.best_val_acc}')
            if self.counter >= self.patience:
                # 如果计数器达到或超过耐心值，触发提前终止
                if self.restore_best_weights:
                    # 恢复到最佳权重
                    model.load_state_dict(self.best_model)
                return True  # 返回True表示需要提前终止训练
        return False  # 返回False表示继续训练


def train(model, optimizer_hash, optimizer_decoder, trainloader, trainset, validloader, df_valid,
          times, weight, change_epoch, early_stopping=None):
    since = time.time()
    best_acc = -1
    bestEpoch = 0

    f = open(os.path.join(exp_path, "record.txt"), "w")

    for epoch in range(change_epoch):
        # trainset.set_epoch(epoch)
        result_per_epoch = train_one_epoch(model, optimizer_hash, optimizer_decoder, trainloader, trainset,
                                           epoch + 1, weight, change_epoch)
        eval_result = eval(model, validloader, df_valid)
        # eval_result_test = eval(model, testloader, df_test, test=True)

        if eval_result['val_acc'] > best_acc:
            bestEpoch = epoch + 1
            best_acc = eval_result['val_acc']
            state_dict = dict(epoch=epoch + 1, model=model.state_dict(), acc=eval_result['val_acc'])
            name = os.path.join(exp_path, "best.pth")
            torch.save(state_dict, name)
            theta = pd.DataFrame(model.Theta_buf.detach().cpu().numpy())
            theta_path = os.path.join(exp_path, "theta.csv")
            theta.to_csv(theta_path, index=False)
            hashes = pd.DataFrame(model.Hash_buf.detach().cpu().numpy())
            hashes_path = os.path.join(exp_path, "hashes.csv")
            hashes.to_csv(hashes_path, index=False)

        msg = "epoch:{} train_acc:{:.4f} Lcontra:{:.4f} || val_acc:{:.4f} auc:{:.4f} rmse:{:.4f} f1:{:.4f} \n".format(
            epoch + 1,
            result_per_epoch['train_acc'],
            result_per_epoch['Lcontra'],
            eval_result['val_acc'],
            eval_result['auc'],
            eval_result['rmse'],
            eval_result['f1']
            # eval_result_test['test_acc'],
            # eval_result_test['auc'],
            # eval_result_test['rmse'],
            # eval_result_test['f1']
        )
        f.write(msg)
        f.flush()
        print(msg)

        if early_stopping is not None:
            if early_stopping(eval_result['val_acc'], model):
                print('Early stopping !!!')
                break

    for epoch_supervise in range(change_epoch, args.epoch):
        # trainset.set_epoch(epoch)
        result_per_epoch = train_one_epoch(model, optimizer_hash, optimizer_decoder, trainloader, trainset,
                                           epoch_supervise + 1, weight, change_epoch)
        eval_result = eval(model, validloader, df_valid)
        # eval_result_test = eval(model, testloader, df_test, test=True)

        if eval_result['val_acc'] > best_acc:
            bestEpoch = epoch_supervise + 1
            best_acc = eval_result['val_acc']
            state_dict = dict(epoch=epoch_supervise + 1, model=model.state_dict(), acc=eval_result['val_acc'])
            name = os.path.join(exp_path, "best.pth")
            torch.save(state_dict, name)
            theta = pd.DataFrame(model.Theta_buf.detach().cpu().numpy())
            theta_path = os.path.join(exp_path, "theta.csv")
            theta.to_csv(theta_path, index=False)
            hashes = pd.DataFrame(model.Hash_buf.detach().cpu().numpy())
            hashes_path = os.path.join(exp_path, "hashes.csv")
            hashes.to_csv(hashes_path, index=False)

        # if (epoch + 1) > change_epoch:
        msg = "epoch:{} train_acc:{:.4f} Lce:{:.4f}  ||  val_acc:{:.4f} auc:{:.4f} rmse:{:.4f} f1:{:.4f}  \n".format(
            epoch_supervise + 1,
            result_per_epoch['train_acc'],
            result_per_epoch['Lce'],
            eval_result['val_acc'],
            eval_result['auc'],
            eval_result['rmse'],
            eval_result['f1']
        )
        f.write(msg)
        f.flush()
        print(msg)

    msg_best = "model:{} best epoch{}  val_acc:{:.4f} \n".format('EADRCD', bestEpoch, best_acc)
    time_elapsed = "traninng time: {:.2f} s \n".format(time.time() - since)
    print(msg_best)
    print(time_elapsed)
    f.write(msg_best)
    f.write(time_elapsed)
    f.close()

def calculate_average_std(test_acc_all,test_auc_all,test_rmse_all,test_f1_all,args,remain):
    # record_test_average = 'Summarize: \n test_acc = %.4f ± %.4f \n test_auc = %.4f ± %.4f \n test_rmse = %.4f ± %.4f \n test_f1 = %.4f ± %.4f' % (
    averageAccuracy = np.mean(test_acc_all) * 100
    stdAcc = np.std(test_acc_all) * 100
    averageAuc = np.mean(test_auc_all) * 100
    stdAuc = np.std(test_auc_all) * 100
    averageRmse = np.mean(test_rmse_all) * 100
    stdRmse= np.std(test_rmse_all) * 100
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

    # traning parameter
    user_dim = int(args.user_dim)
    item_dim = int(args.item_dim)
    epoch = int(args.epoch)
    change_epoch = int(args.change_epoch)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    bits = int(args.bits)
    temperature = float(args.T)
    weight = float(args.weight)
    k = int(args.k)
    device = torch.device(args.device)

    # load data
    df_train = pd.read_csv(args.train_file)
    df_valid_train = pd.read_csv(args.valid_train_file)
    df_valid_test = pd.read_csv(args.valid_test_file)
    df_test_train = pd.read_csv(args.test_train_file)
    df_test_test = pd.read_csv(args.test_test_file)

    Q_mat = np.load(args.Q_matrix)
    # calculate all num
    n_user = int(len(df_train['user_id'].unique()))
    n_item = int(np.max([np.max(df_train['item_id']),
                         np.max(df_valid_train['item_id']), np.max(df_valid_test['item_id']),
                         np.max(df_test_train['item_id']), np.max(df_test_test['item_id'])]) + 1)
    n_know = int(Q_mat.shape[1])
    n_user_new0 = int(np.max(df_valid_train['user_id']) + 1)  # new student log : student num
    n_user_new = int(np.max(df_test_train['user_id']) + 1)  # new student log : student num

    # combine Q_matrix to data
    df_train = add_knowledge_code(df_train, Q_mat)
    # new student: valid data : remain p% item is df_valid_train, and the rest is df_valid_test
    df_valid_train = add_knowledge_code(df_valid_train, Q_mat)
    df_valid_test = add_knowledge_code(df_valid_test, Q_mat)
    # new student: test data : remain p% item is df_test_train, and the rest is df_test_test
    df_test_train = add_knowledge_code(df_test_train, Q_mat)
    df_test_test = add_knowledge_code(df_test_test, Q_mat)

    trainset = RACDataset(df_train, n_user, n_item)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)

    valid_trainset = RACDataset(df_valid_train, n_user_new0, n_item)
    valid_testset = RACDataset(df_valid_test, n_user_new0, n_item,valid_type='inductive')
    valid_testset.log_mat_train = trainset.log_mat
    valid_testset.log_mat = valid_trainset.log_mat.copy()
    valid_testloader = DataLoader(dataset=valid_testset, batch_size=batch_size, shuffle=False)

    # df_test_train = add_knowledge_code(df_test_train, Q_mat)
    # df_test_test = add_knowledge_code(df_test_test, Q_mat)
    # test_trainset = RCDataset(df_test_train, n_user_new, n_item)
    # test_testset = RCDataset(df_test_test, n_user_new, n_item,valid_type='inductive')
    # test_testset.log_mat_train = trainset.log_mat
    # test_testset.log_mat = test_trainset.log_mat.copy()
    # test_testloader = DataLoader(dataset=test_testset, batch_size=batch_size, shuffle=False)

    #training
    times = args.times
    # calculate master matrix
    theta_path = os.path.join(args.pretrain_model_path, "theta.csv")
    theta = pd.read_csv(theta_path)
    Master_matrix = statistics(theta, pd.DataFrame(trainset.log_mat), Q_mat).to_numpy()
    # print(Master_matrix)
    theta_buf = torch.tensor(theta.to_numpy(), dtype=torch.float32).to('cuda:0')

    test_acc_all,test_auc_all,test_rmse_all,test_f1_all = [],[],[],[]
    out_path = "{}/{}/{}/".format(args.save_path, args.data, args.model_name)
    os.makedirs(out_path, exist_ok=True)

    for i in range(times):
        # save text
        exp_path = "{}/{}/{}/{}/".format(args.save_path, args.data, args.model_name, str(i + 1))
        os.makedirs(exp_path, exist_ok=True)

        # set seed
        random.seed(args.seed + i)
        np.random.seed(args.seed + i)
        torch.manual_seed(args.seed + i)

        print(f'======================================Run {i + 1} RCD======================================')
        model = RCD(n_user, n_item, n_know, user_dim, item_dim, Q_mat, Master_matrix, bits, k, device,
                    train=True, monotonicity_assumption=True)
        model.Theta_buf = theta_buf

        # loading pertrain IDCDF and forzen encoder
        RCD_model_dict = model.state_dict()
        pretrained_dict = torch.load(os.path.join(args.pretrain_model_path, "best.pth"), \
                                     weights_only=False)['model']
        RCD_model_dict.update(pretrained_dict)
        model.load_state_dict(RCD_model_dict)
        model.f_nn.requires_grad_(False)
        model.g_nn.requires_grad_(False)

        # double optimizer
        optimizer_hash = torch.optim.Adam([{'params': model.hashing.parameters()}], lr=lr)
        decoder_params = list(model.theta_agg_mat.parameters()) + \
                         list(model.psi_agg_mat.parameters()) + \
                         list(model.ncd.parameters())
        optimizer_decoder = torch.optim.Adam([{'params': decoder_params}], lr=lr)
        early_stopping = EarlyStopping(patience=5, min_delta=0, restore_best_weights=True)
        train(model, optimizer_hash, optimizer_decoder, trainloader, trainset, valid_testloader, df_valid_test, i, weight,
              change_epoch,early_stopping=early_stopping)

        model.load_state_dict(torch.load(os.path.join(exp_path, "best.pth"),weights_only=False)['model'])

        # load database
        hashes = pd.read_csv(os.path.join(exp_path, "hashes.csv")).to_numpy()
        hashes = torch.tensor(hashes, dtype=torch.float32).to('cuda:0')
        model.Hash_buf = hashes
        theta = pd.read_csv(os.path.join(exp_path, "theta.csv")).to_numpy()
        theta = torch.tensor(theta, dtype=torch.float32).to('cuda:0')
        model.Theta_buf = theta

        remain_rate = [10,20,30,40,50,60,70,80,90]

        for remain in remain_rate:
            args.test_train_file = f'./data/{data_name}/InducData/test/test_{remain}_in.csv'
            args.test_test_file = f'./data/{data_name}/InducData/test/test_{remain}_out.csv'
            df_test_train = pd.read_csv(args.test_train_file)
            df_test_test = pd.read_csv(args.test_test_file)
            df_test_train = add_knowledge_code(df_test_train, Q_mat)
            df_test_test = add_knowledge_code(df_test_test, Q_mat)
            test_trainset = RACDataset(df_test_train, n_user_new, n_item)
            test_testset = RACDataset(df_test_test, n_user_new, n_item,valid_type='inductive')
            test_testset.log_mat_train = trainset.log_mat
            test_testset.log_mat = test_trainset.log_mat.copy()
            test_testloader = DataLoader(dataset=test_testset, batch_size=batch_size, shuffle=False)

            test_result = eval(model, test_testloader, df_test_test, test=True)

            test_acc_all.append(test_result['test_acc'])
            test_auc_all.append(test_result['auc'])
            test_rmse_all.append(test_result['rmse'])
            test_f1_all.append(test_result['f1'])

    remain_time = 9
    for k in range(0, remain_time):
        accuracyList1 = [test_acc_all[j] for j in range(k, times*remain_time, remain_time)]
        test_auc_all1 = [test_auc_all[j] for j in range(k, times*remain_time, remain_time)]
        test_rmse_all1 = [test_rmse_all[j] for j in range(k, times*remain_time, remain_time)]
        test_f1_all1 = [test_f1_all[j] for j in range(k, times*remain_time, remain_time)]
        endText = calculate_average_std(accuracyList1,test_auc_all1,test_rmse_all1,test_f1_all1,args,remain_rate[k])
        print(endText)
        with open(os.path.join(out_path, "test_performance.txt"), 'a') as f:
            f.write(endText)
            f.write('\n')
            f.close()
    with open(os.path.join(out_path, "test_performance.txt"), 'a') as f:
        f.write('\n')
        f.close()

    gc.collect()
