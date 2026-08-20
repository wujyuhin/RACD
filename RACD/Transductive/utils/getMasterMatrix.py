import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# user_id,item_id, score ----> log_mat
def trainData2Matrix(data, n_user, n_item):
    # return: log_mat np.array
    log_mat = np.zeros((n_user, n_item))
    user_id = data['user_id'].values
    item_id = data['item_id'].values
    score = data['score'].values
    for i in range(data.shape[0]):
        log_mat[int(user_id[i]), int(item_id[i])] = (score[i] - 0.5) * 2
    return pd.DataFrame(log_mat)


# discrimination = (mean_high - mean_low)
# log_mat: DataFrame user_id*item_id  ------> discrimination: np.array item_id*1
def discrimination(log_mat, bound=0.33):
    count1 = np.array([(log_mat.iloc[i, :] == 1).sum() for i in range(log_mat.shape[0])])
    count_1 = abs(np.array([(log_mat.iloc[i, :] == -1).sum() for i in range(log_mat.shape[0])]))
    rate = count1 / (count1 + count_1)  # transform to DataFrame record the index
    # 2 class student
    bound_down, bound_up = np.quantile(rate, bound), np.quantile(rate, 1 - bound)
    low_student, high_student = log_mat.loc[np.where(rate < bound_down)], log_mat.loc[np.where(rate > bound_up)]
    # 低分组的平均得分
    count1_low = np.array([(low_student.iloc[:, j] == 1).sum() for j in range(low_student.shape[1])])
    count_1_low = np.array([(low_student.iloc[:, j] == -1).sum() for j in range(low_student.shape[1])])
    mean_low = (count1_low) / (count1_low + count_1_low + 0.000001)
    # 高分组的平均得分
    count1_high = np.array([(high_student.iloc[:, j] == 1).sum() for j in range(high_student.shape[1])])
    count_1_high = np.array([(high_student.iloc[:, j] == -1).sum() for j in range(high_student.shape[1])])
    mean_high = (count1_high) / (count1_high + count_1_high + 0.000001)
    # discrimination
    disc = mean_high - mean_low  # discrimination = (mean_high - mean_low)
    return disc


def statistic(discrimination_value,theta_mat, log_mat, Q_mat, knowledge_id=6):
    """
    theta_mat: DataFrame user_id*knowledge_id
    log_mat: DataFrame user_id*item_id
    Q_mat: nparray item_id*knowledge_id
    knowledge_id: the knowledge_id :from 1 to knowledge_n
    """
    k = 10  # 提取区分度最大的k道题目
    item = np.where(Q_mat[:, knowledge_id - 1] == 1)[0]
    item_select = np.argsort(discrimination_value[item])[-k:]  # 从小到大排序，取最后k个
    log_mat_select = log_mat.iloc[:, item_select]
    # 存在-1的行
    neg_one = np.array(np.where(log_mat_select == -1, 0, 1).sum(axis=1) < log_mat_select.shape[1])  # 存在-1的行
    zero_one = np.array(np.where(log_mat_select == 0, 0, 1).sum(axis=1)) == 0  # 存在全0的行
    pos_one = np.array(np.where(log_mat_select == 1, 0, 1).sum(axis=1) < log_mat_select.shape[1])  # 存在1的行
    log_mat_true = log_mat_select.iloc[~(neg_one + zero_one), :]
    log_mat_false = log_mat_select.iloc[~(pos_one + zero_one), :]
    # 2 class theta
    theta_mat = np.array(theta_mat)
    theta_true = theta_mat[log_mat_true.index, knowledge_id - 1]
    theta_false = theta_mat[log_mat_false.index, knowledge_id - 1]
    #print("theta_true", len(theta_true))
    #print("theta_false", len(theta_false))
    # max of all
    max_theta = np.max(theta_mat[:, knowledge_id - 1])
    min_theta = np.min(theta_mat[:, knowledge_id - 1])
    # mean of theta_true and theta_false
    mean_theta_true = np.mean(theta_true)
    mean_theta_false = np.mean(theta_false)
    # std of theta_true and theta_false

    # std_theta_true = np.std(theta_true)
    # std_theta_false = np.std(theta_false)

    std_theta_true = np.std(theta_true) + 0.001
    std_theta_false = np.std(theta_false) +0.001

    # # inf of theta_true = max( min(theta_true), max(theta_False))
    # inf_theta_true = max(min(theta_true), max(theta_false))
    # # sup of theta_false = min( max(theta_false), min(theta_true))
    # sup_theta_false = min(min(theta_true), max(theta_false))

    sup_theta_false = (mean_theta_false+mean_theta_true)/2
    inf_theta_true = sup_theta_false
    
    return (min_theta, sup_theta_false, inf_theta_true, max_theta,
            mean_theta_false, mean_theta_true, std_theta_false, std_theta_true)


def statistics(theta_mat, log_mat, Q_mat):
    """
    theta_mat: DataFrame user_id*knowledge_id
    log_mat: DataFrame user_id*item_id
    Q_mat: nparray item_id*knowledge_id
    """
    n_knowledge = Q_mat.shape[1]
    min_theta, sup_theta_f, inf_theta_t, max_theta, mean_f, mean_t, std_f, std_t = [], [], [], [], [], [], [], []
    # 计算discrimination
    discrimination_value = discrimination(log_mat)
    for knowledge_id in range(1, n_knowledge + 1):
        min_theta_i, sup_theta_f_i, inf_theta_t_i, max_theta_i, mean_f_i, mean_t_i, std_f_i, std_t_i = statistic(
            discrimination_value,theta_mat, log_mat, Q_mat, knowledge_id)
        min_theta.append(min_theta_i)
        sup_theta_f.append(sup_theta_f_i)
        inf_theta_t.append(inf_theta_t_i)
        max_theta.append(max_theta_i)
        mean_f.append(mean_f_i)
        mean_t.append(mean_t_i)
        std_f.append(std_f_i)
        std_t.append(std_t_i)
    # 合并成上述dataframe
    data = pd.DataFrame({'min': min_theta, 'sup': sup_theta_f, 'inf': inf_theta_t, 'max': max_theta,
                         'meanF': mean_f, 'meanT': mean_t, 'stdF': std_f, 'stdT': std_t})
    return data


def three_sigma(s):
    """
    3-sigma rule
    :param s: np.array
    """
    mu, std = np.mean(s), np.std(s)
    lower, upper = mu - 3 * std, mu + 3 * std
    # 删除异常值
    s = s[(s > lower) & (s < upper)]
    return s


def boxplot(s):
    """
    boxplot rule
    :param s: np.array
    """

    q1, q3 = np.quantile(s, .25), np.quantile(s, .75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    s = s[(s > lower) & (s < upper)]
    return s


# if __name__ == "__main__":
#     # params setting
#     args = parse_args()
#     # data loading
#     theta = np.array(pd.read_csv('../report/IDCD_fix_get_theta/MATH1/2/theta.csv'))
#     Q_mat = np.load('../data/math1_Q_matrix.npy')  # type is array
#     train_data = pd.read_csv('../data/math1_train_0.8_0.2.csv')
#     log_mat = trainData2Matrix(train_data, args.n_user, args.n_item)  # train_data to matrix
#     # statistics
#     data = statistics(theta, log_mat, Q_mat)
    # min_theta,sup_theta_f,inf_theta_t,max_theta,mean_f,mean_t,std_f,std_t = statistics(theta, log_mat, Q_mat)  # knowledge_id = 6
    # knowledge_id = 6
    # item = np.where(Q_mat[:, knowledge_id - 1] == 1)[0]
    # # 计算discrimination
    # discrimination_value = discrimination(log_mat)
    # # 提取区分度最大的k道题目
    # k = 5
    # item_select = np.argsort(discrimination_value[item])[-k:]  # 从小到大排序，取最后k个
    # log_mat_select = log_mat.iloc[:, item_select]
    # # 存在-1的行
    # neg_one = np.array(np.where(log_mat_select == -1, 0, 1).sum(axis=1) < log_mat_select.shape[1])  # 存在-1的行
    # zero_one = np.array(np.where(log_mat_select == 0, 0, 1).sum(axis=1)) == 0  # 存在全0的行
    # pos_one = np.array(np.where(log_mat_select == 1, 0, 1).sum(axis=1) < log_mat_select.shape[1])  # 存在1的行
    # log_mat_true = log_mat_select.iloc[~(neg_one + zero_one), :]
    # log_mat_false = log_mat_select.iloc[~(pos_one + zero_one), :]
    # # O_true的行索引
    # theta_true = theta[log_mat_true.index, knowledge_id - 1]
    # theta_false = theta[log_mat_false.index, knowledge_id - 1]
    # # 画出theta_true和theta_false的分布图
    # # 根据规则删除异常值
    # theta_true = three_sigma(theta_true)
    # theta_false = three_sigma(theta_false)
    # # np.mean(theta_false)
    # # np.std(theta_false)
    # plt.hist(theta_true, bins=100, alpha=0.5, label='True')
    # plt.hist(theta_false, bins=100, alpha=0.5, label='False')
    # plt.legend()
    # plt.show()
