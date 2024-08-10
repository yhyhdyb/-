import pandas as pd
import numpy as np
# 读取CSV文件
import my_model
import  torch
import argparse
from tqdm import tqdm, trange
import data
import torch.nn as nn
#获得训练集

criterion = nn.MSELoss()  # 使用均方误差损失函数计算MSE
def get_train_data(file_path,edge_pth):
    df = pd.read_csv(file_path, encoding='utf-8') # 使用Pandas读取位于file_path的CSV文件，并将其赋值给DataFrame df
    edge_df = pd.read_csv(edge_pth, encoding='utf-8') # 读取位于edge_pth的另一个CSV文件，并将其赋值给DataFrame edge_df
    df.head()

    # %%
    geohasd_df_dict = {}
    date_df_dict = {}
    # 代码初始化了两个字典：geohasd_df_dict和date_df_dict

    number_hash = 0
    number_date = 0
    for i in df["geohash_id"]: # 用于存储DataFrame中"geohash_id"列的唯一值

        if i not in geohasd_df_dict.keys():
            geohasd_df_dict[i] = number_hash # 为字典中的每个唯一值分配一个数字索引
            number_hash += 1

    for i in df["date_id"]: # 用于存储DataFrame中"date_id"列的唯一值
        if i not in date_df_dict.keys():
            date_df_dict[i] = number_date # 为字典中的每个唯一值分配一个数字索引
            number_date += 1

    new_data = np.zeros((len(date_df_dict),len(geohasd_df_dict),38))
    # 创建了一个新的数组new_data，其维度为(len(date_df_dict), len(geohasd_df_dict))，并将其填充为零
    
    for index, row in df.iterrows():
        # print(index)
        hash_index, date_index = geohasd_df_dict[row["geohash_id"]], date_df_dict[row["date_id"]]
        # 遍历DataFrame df中的每一行，并提取"geohash_id"和"date_id"的值
        #将时间index加到里面

        new_data[date_index][hash_index] = [date_index]+list(row.iloc[2:])
        # 将行中剩余的值（从第三列开始）分配到new_data数组中的相应位置
    new_data = np.array(new_data) # 最后，将new_data转换为NumPy数组
    # x_train,y_train = new_data[:, :-2], new_data[:, -2:]
    # print(len(geohasd_df_dict))
    # exit()
    # print(x_train.shape)
    # print(y_train.shape)
    #这里构建邻接矩阵其中mask表示1为有边，0无边， value_mask表示有值
    #并且这里我考虑mask是一个无向图，如果有向删除x_mask[date_index][point2_index][point1_index],value_mask同理
    x_mask =  np.zeros((len(date_df_dict),len(geohasd_df_dict),len(geohasd_df_dict),1), dtype = float)
    x_edge_df =np.zeros((len(date_df_dict),len(geohasd_df_dict),len(geohasd_df_dict),2), dtype = float)
    # 代码初始化了两个数组：x_mask和x_edge_df，它们的维度分别为(len(date_df_dict), len(geohasd_df_dict), len(geohasd_df_dict), 1)
    # 和(len(date_df_dict), len(geohasd_df_dict), len(geohasd_df_dict), 2)
    # 并且都填充为零

    for index, row in edge_df.iterrows(): # 遍历DataFrame edge_df中的每一行
        # print(index)
        if row["geohash6_point1"] not in geohasd_df_dict.keys() or row["geohash6_point2"] not in geohasd_df_dict.keys():
            continue # 如果"geohash6_point1"和"geohash6_point2"都存在于geohasd_df_dict字典中，则继续下一步。否则，跳过当前迭代
        point1_index,point2_index,F_1,F_2,date_index= geohasd_df_dict[row["geohash6_point1"]],geohasd_df_dict[row["geohash6_point2"]]\
            ,row["F_1"],row["F_2"],date_df_dict[row["date_id"]] # 提取"geohash6_point1"、"geohash6_point2"、"F_1"、"F_2"和"date_id"的值
        x_mask[date_index][point1_index][point2_index] = 1 
        x_mask[date_index][point2_index][point1_index] = 1 # 将x_mask数组中的相应位置设置为1，表示两个点之间存在一条边
        x_edge_df[date_index][point1_index][point2_index] =  [F_1,F_2]
        x_edge_df[date_index][point2_index][point1_index] = [F_1, F_2] # 还将"F_1"和"F_2"的值分配给x_edge_df数组中的相应位置
    # print(data)

    return geohasd_df_dict, date_df_dict, new_data,x_mask, x_edge_df # 返回字典geohasd_df_dict和date_df_dict，经过处理的数据new_data，邻接矩阵x_mask和边特征x_edge_df

def eval(model, dataset, args): # 调用model.eval()将模型设置为评估模式
    model.eval()
    with torch.no_grad(): # 使用torch.no_grad()上下文管理器，确保在评估过程中不会计算梯度，从而节省内存和计算资源

        dev_loss = 0.0 # 初始化变量dev_loss为0.0，用于累计评估损失
        for j in trange(dataset.batch_count): # 使用trange迭代数据集的批次数量，其中tqdm用于显示进度条
            x_date, x_feature, x_mask_data, x_edge_data, x_tags = dataset.get_batch(j) # 从数据集中获取一个批次的输入数据，包括x_date、x_feature、x_mask_data、x_edge_data和x_tags
            act_pre, con_pre = model(x_date, x_feature, x_mask_data) # 调用模型model并传递输入数据x_date、x_feature和x_mask_data，以获取预测结果
            predict = torch.cat((act_pre, con_pre), dim=-1) # 将活动预测结果 act_pre 和内容预测结果 con_pre 沿着最后一个维度拼接起来，得到 predict
            loss = criterion(predict, x_tags) # 使用预定义的损失函数 criterion 计算预测结果 predict 和真实标签 x_tags 之间的损失
            dev_loss+= loss # 将损失累加到 dev_loss 上
        print("this epoch dev loss is {}".format(dev_loss))
        model.train() # 调用model.train()将模型设置为训练模式，以便进行后续的训练过程


def train(args):
    
    geohasd_df_dict, date_df_dict, x_train, x_mask, x_edge_df = get_train_data('./train_90.csv',"./edge_90.csv")
    # 调用函数get_train_data()从文件中加载训练数据，并将数据存储在各个变量中，如geohasd_df_dict、date_df_dict、x_train、x_mask和x_edge_df
    
    #分割各种训练集测试集
    x_train,x_dev = torch.tensor(x_train[:int(len(x_train)*args.rat)]),torch.tensor(x_train[int(len(x_train)*args.rat):])
    x_mask_train,x_mask_dev = torch.tensor(x_mask[:int(len(x_mask)*args.rat)]),torch.tensor(x_mask[int(len(x_mask)*args.rat):])
    x_edge_train, x_edge_dev = torch.tensor(x_edge_df[:int(len(x_edge_df) * args.rat)]),torch.tensor( x_edge_df[int(len(x_edge_df) * args.rat):])
    # 根据args.rat的比例，将训练数据x_train、x_mask和x_edge_df划分为训练集和验证集
    # 分别存储在x_train、x_dev、x_mask_train、x_mask_dev、x_edge_train和x_edge_dev中

    date_emb = 5
     # 这里的x包含了date_id+F35个特征+2个y值的
    # train_activate = torch.tensor(y_train[:, -2])
    # train_consume = torch.tensor(y_train[:, -1])

    # 创建GAT和BILSTM模型实例
    gat_model = my_model.GAT(date_emb=[len(date_df_dict), date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    bilstm_model = my_model.BILSTM(date_emb=[len(date_df_dict), date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)

    gat_optimizer = torch.optim.Adam(params=gat_model.parameters(),lr=args.lr)
    bilstm_optimizer = torch.optim.Adam(params=bilstm_model.parameters(),lr=args.lr)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr)
    # 创建一个Adam优化器，将模型的参数传递给它，学习率为args.lr

    gat_model.train()

    gat_trainset = data.DataIterator(x_train,x_mask_train,x_edge_train, args)
    gat_valset =data.DataIterator(x_dev,x_mask_dev,x_edge_dev, args)

    for indx in range(args.epochs):
        train_all_loss = 0.0
        for j in trange(gat_trainset.batch_count):
            x_date,x_feature,x_mask_data,x_edge_data,x_tags= gat_trainset.get_batch(j)
            act_pre, con_pre = gat_model(x_date,x_feature,x_mask_data)
            gat_predict = torch.cat((act_pre, con_pre), dim=-1)

            gat_loss = criterion(gat_predict, x_tags)
            train_all_loss += loss
            gat_optimizer.zero_grad()
            gat_loss.backward()
            gat_optimizer.step()
        print('this epoch train loss :{0}'.format(train_all_loss))
        # scheduler.step()
        eval(gat_model,gat_valset, args)
    gat_output = gat_model(data) 

    bilstm_model.train()

    bilstm_trainset = data.DataIterator(x_train,x_mask_train,x_edge_train, args)
    bilstm_valset =data.DataIterator(x_dev,x_mask_dev,x_edge_dev, args)

    for indx in range(args.epochs):
        train_all_loss = 0.0
        for j in trange(bilstm_trainset.batch_count):
            x_date,x_feature,x_mask_data,x_edge_data,x_tags= gat_trainset.get_batch(j)
            act_pre, con_pre = bilstm_model(x_date,x_feature,x_mask_data)
            bilstm_predict = torch.cat((act_pre, con_pre), dim=-1)

            bilstm_loss = criterion(bilstm_predict, x_tags)
            train_all_loss += loss
            bilstm_optimizer.zero_grad()
            bilstm_loss.backward()
            bilstm_optimizer.step()
        print('this epoch train loss :{0}'.format(train_all_loss))
        # scheduler.step()
        eval(bilstm_model,bilstm_valset, args)
    bilstm_output = bilstm_model(data)   

    features = torch.cat((gat_output, bilstm_output), dim=1)

    # rmse_loss = torch.sqrt(mse_loss)
    model = my_model.GAT(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    # 创建一个GAT模型对象，使用my_model.GAT进行初始化。模型具有一些参数，如date_emb、nfeat、nhid、dropout、alpha和nheads
    # 将模型移动到设备args.device上

    # model = my_model.BILSTM(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)
    model.train() # 调用model.train()将模型设置为训练模式
    trainset = data.DataIterator(x_train,x_mask_train,x_edge_train, args)
    valset =data.DataIterator(x_dev,x_mask_dev,x_edge_dev, args)
    # 使用data.DataIterator创建训练集和验证集的数据迭代器，传递相应的输入数据和参数args
    for indx in range(args.epochs):
        train_all_loss = 0.0
        for j in trange(trainset.batch_count):
            x_date,x_feature,x_mask_data,x_edge_data,x_tags= trainset.get_batch(j)
            act_pre, con_pre = model(features)
            predict = torch.cat((act_pre, con_pre), dim=-1)

            loss = criterion(predict, x_tags)
            train_all_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('this epoch train loss :{0}'.format(train_all_loss))
        # scheduler.step()
        eval(model,valset, args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='training epoch number')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')
    parser.add_argument('--lr', type=float, default=1e-3,
                        )
    parser.add_argument('--rat', type=float, default=0.9,)

    parser.add_argument('--decline', type=int, default=30, help="number of epochs to decline")
    train(parser.parse_args())


# %%

# gcn = GCN(n_features=51, hidden_dim=64, dropout=0.3, n_classes=2)

# %%

# date_node = np.concatenate([sample.iloc[:, 2:37].values, date_embed], axis=1)
# date_node.shape
#
# # %%
#
# adj = torch.randn(90, 90)
# x_node = torch.from_numpy(date_node).type_as(adj)
# predict = gcn(x_node, adj)
# predict.shape, predict
#
# # %%
#
# gat = GAT(nfeat=51, nhid=64, nclass=2, dropout=0.3, alpha=0.3, nheads=8)
#
# # %%
#
# adj = torch.randn(90, 90)
# x_node = torch.from_numpy(date_node).type_as(adj)
# predict = gat(x_node, adj)
# predict.shape, predict
#
# # %%
#
# actual_values = sample.iloc[:, 37:]  # active_index，consume_index
#
# # %%
#
# import torch
# import torch.nn as nn
#
# # 定义预测值和实际观测值
# actual_values = torch.from_numpy(sample.iloc[:, 37:].values)  # 实际观测值
#
# # 计算RMSE损失
# criterion = nn.MSELoss()  # 使用均方误差损失函数计算MSE
# mse_loss = criterion(predict, actual_values)
# rmse_loss = torch.sqrt(mse_loss)
#
# # 打印RMSE损失
# print("RMSE Loss:", rmse_loss.item())
#
# # %%
#
# x_node, adj
#
# # %%
#
# import pandas as pd
#
# # 读取CSV文件
# edge = pd.read_csv('/home/cike/workspace/data/datamining/edge_90.csv')
#
# # %%
#
# edge.head()
#
# # %%
#
# geohash_id = df.geohash_id
#
# # %%
#
# geohash_id.head()
#
# # %%
#
# align = edge[(edge['geohash6_point1'] == df.geohash_id[0]) & (edge['date_id'] == df.date_id[0])]
#
# # %%
#
# align
#
# # %%
#
# df.geohash_id.drop_duplicates()
