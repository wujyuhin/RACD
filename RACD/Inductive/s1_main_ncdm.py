from NCDM.NCDM import NCDM
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import argparse
import numpy as np
import random

model_name = 'NCDM'
data_name = 'math1'
# data_name = "assist_17"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_experiments', help='the number of experiments', default=1)
    parser.add_argument('--train_file', default=f"./data/{data_name}/InData/train.csv")
    parser.add_argument('--valid_file', default=f"./data/{data_name}/TransData/val.csv")   
    parser.add_argument('--test_file',  default=f"./data/{data_name}/TransData/test.csv")
    parser.add_argument('--Q_matrix', default=f"./data/{data_name}/TransData/Q_csv.csv")
    parser.add_argument('--model_name', help='the model name', default=f'{model_name}')
    parser.add_argument('--dataset', help='the dataset name', default=f'{data_name}')
    parser.add_argument('--save_root', help='the save path of all results', default="./inductive_result")
    parser.add_argument('--batch_size', help='the batch size in the training phase', default=64)
    parser.add_argument('--lr', help='the learning rate in the training phase', default=1e-3)
    parser.add_argument('--epoch', help='the training epoch', default=20)
    parser.add_argument('--device', help='the running device. cpu or gpu', default='cuda')
    parser.add_argument('--seed', help='the random seed', default=45)
    parser.add_argument('--early_stop', help='whether to use early stopping', action='store_true')
    args = parser.parse_args()
    return args


def get_knowledge_nums(item_data):
    item2knowledge = {}
    knowledge_set = set()
    for i, s in item_data.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)
    return np.max(list(knowledge_set))


def get_knowledge_nums_NCDM(item_data):
    item2knowledge = {}
    knowledge_set = set()
    for i, s in item_data.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)
    return len(list(knowledge_set)), item2knowledge


def transform_NCDM(user, item, item2knowledge, score, batch_size, knowledge_n):
    knowledge_emb = torch.zeros((len(item), knowledge_n))
    for idx in range(len(item)):
        knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0

    data_set = TensorDataset(
        torch.tensor(user, dtype=torch.int64),  # (0, user_n-1)
        torch.tensor(item, dtype=torch.int64),  # (0, item_n-1)
        knowledge_emb,
        torch.tensor(score, dtype=torch.float32)
    )
    return DataLoader(data_set, batch_size=batch_size, shuffle=True)


def code2vector(x, knowledge_num):
    vector = [0] * knowledge_num
    for k in eval(x):
        vector[k] = 1
    return vector


def train_model(train_data, valid_data, test_data, args, item_data=None, early_stop=True):
    # calculate all num
    user_num = np.max(train_data['user_id']) + 1
    item_num = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])]) + 1
    # device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # deal with the input data
    knowledge_num, item2knowledge = get_knowledge_nums_NCDM(item_data)
    train, valid, test = [transform_NCDM(data["user_id"], data["item_id"], item2knowledge,
                                         data["score"], args.batch_size, knowledge_num)
                          for data in [train_data, valid_data, test_data]]

    cdm = NCDM(knowledge_num, item_num, user_num)
    cdm.train(train, valid, epoch=args.epoch, lr=args.lr, save_dir=args.save_dir, device=device, early_stop=early_stop)
    cdm.load(os.path.join(args.save_dir, f"{args.model_name}.pth"))
    accuracy, auc, rmse, f1 = cdm.eval(valid)
    return accuracy, auc, rmse, f1


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # transform inductive data to transductive data for NCDM
    train_data = pd.read_csv(f"./data/{data_name}/InducData/train.csv")
    valid_data = pd.read_csv(f"./data/{data_name}/InducData/val/val_20_out.csv")
    valid_data2 = pd.read_csv(f"./data/{data_name}/InducData/val/val_20_in.csv")
    valid_data = pd.concat([valid_data, valid_data2], axis=0)
    valid_data.reset_index(drop=True, inplace=True)
    test_data = pd.read_csv(f"./data/{data_name}/InducData/test/test_20_in.csv")
    test_test_data = pd.read_csv(f"./data/{data_name}/InducData/test/test_20_out.csv")
    df_item = pd.read_csv(f"./data/{data_name}/InducData/Q_csv.csv")

    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))  # knowledge_code: str
        item2knowledge[item_id] = knowledge_codes  # item_id: int
        knowledge_set.update(knowledge_codes)

    item_data = pd.read_csv(args.Q_matrix)
    knowledge_num, item2knowledge = get_knowledge_nums_NCDM(item_data)

    def transform(user, item, item2knowledge, score, batch_size):
        knowledge_emb = torch.zeros((len(item), knowledge_num))
        for idx in range(len(item)):
            try:
                knowledge_emb[idx][np.array(item2knowledge[item[idx]])] = 1.0
            except:
                a = 1

        data_set = TensorDataset(
            torch.tensor(user, dtype=torch.int64),  # (1, user_n) to (0, user_n-1)
            torch.tensor(item, dtype=torch.int64),  # (1, item_n) to (0, item_n-1)
            knowledge_emb,
            torch.tensor(score, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=True)

    args.save_dir = f"./data/{data_name}/InducData"
    user_num = np.max(train_data['user_id']) + 1
    item_num = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])]) + 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train, valid= [transform_NCDM(data["user_id"], data["item_id"], item2knowledge,
                                  data["score"], args.batch_size, knowledge_num)
                   for data in [train_data, valid_data]]

    cdm = NCDM(knowledge_num, item_num, user_num)
    cdm.train(train, valid, epoch=args.epoch, lr=args.lr, save_dir=args.save_dir, device=device, early_stop=args.early_stop)
    cdm.load(os.path.join(args.save_dir, f"{args.model_name}.pth"))

    psi = pd.DataFrame(cdm.ncdm_net.k_difficulty.weight.detach().cpu().numpy())
    theta = pd.DataFrame(cdm.ncdm_net.student_emb.weight.detach().cpu().numpy())
    os.makedirs(args.save_dir, exist_ok=True)
    theta.to_csv(os.path.join(args.save_dir, "NCDM_theta.csv"), index=False)
    psi.to_csv(os.path.join(args.save_dir, "NCDM_psi.csv"), index=False)