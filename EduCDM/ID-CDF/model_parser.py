#! python3
# -*- encoding: utf-8 -*-

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file',help='the path of the train file', default='data/math1_train_0.8_0.2.csv')
    parser.add_argument('--test_file',help='the path of the test file', default='data/math1_test_0.8_0.2.csv')
    parser.add_argument('--valid_file',help='the path of the valid file', default='data/math1_valid_0.8_0.2.csv')
    parser.add_argument('--Q_matrix',help='the path of the q-matrix', default='data/math1_Q_matrix.npy')
    parser.add_argument('--save_path',help='the save path of all results',default='./result/ID-CDM-Math1')
    parser.add_argument('--n_user', help='the number of students in the entire dataset', default=4209)
    parser.add_argument('--n_item', help='the number of exercises in the entire dataset', default=20)
    parser.add_argument('--n_know', help='the number of knowledge points in the entire dataset',default=11)
    parser.add_argument('--user_dim', help='the dimension of user vector', default=64)
    parser.add_argument('--item_dim', help='the dimension of item vector', default=64)
    parser.add_argument('--batch_size', help='the batch size in the training phase', default=64)
    parser.add_argument('--lr', help='the learning rate in the training phase', default=5e-3)
    parser.add_argument('--epoch', help='the training epoch', default=10)
    parser.add_argument('--device', help='the running device. cpu or gpu', default='cuda:0')
    args = parser.parse_args()
    return args 

if __name__ == '__main__':
    args = parse_args()
    print(args.train_file)
    