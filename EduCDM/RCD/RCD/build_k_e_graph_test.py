import csv

# data_file = '../data/junyi/TransData/train.csv'
# q_file = '../data/junyi/TransData/Q_mat.npy'
# config_file = '../data/junyi/TransData/config.txt'

def build_local_map():
    # 读取配置文件
    with open('../data/junyi_more50/TransData/config.txt', 'r') as f:
        _ = f.readline()  # 跳过注释行
        config_line = f.readline().strip()
        student_n, exer_n, knowledge_n = map(int, config_line.split(','))

    # 处理Q矩阵文件
    item_to_exer = {}
    item_knowledge = {}
    with open('../data/junyi_more50/TransData/Q_csv.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            item_id = int(row[0])
            k_list = eval(row[1].strip())

            if item_id not in item_to_exer:
                item_to_exer[item_id] = len(item_to_exer)
            item_knowledge[item_to_exer[item_id]] = k_list

    exer_total = len(item_to_exer)

    # 处理用户行为数据
    edges = set()
    with open('../data/junyi_more50/TransData/train.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过标题行
        for row in reader:
            item_id = int(row[1])
            exer_id = item_to_exer.get(item_id)

            if exer_id is None:
                continue

            k_list = item_knowledge.get(exer_id, [])
            for k in k_list:
                src = exer_id
                dst = k - 1 + exer_total

                # 使用有序元组保证唯一性
                edge = tuple(sorted((src, dst)))
                if edge not in edges:
                    edges.add(edge)

    # 写入输出文件EduCDM/RCD/data/junyi/TransData/graph
    with open('../data/junyi_more50/TransData/graph/k_from_e.txt', 'w') as f:
        for s, d in edges:
            f.write(f"{s}\t{d}")

            with open('e_from_k.txt', 'w') as f:
                for s, d in edges:
                    f.write(f"{s}\t{d}")

            if __name__ == '__main__':
                build_local_map()

if __name__ == '__main__':
    build_local_map()