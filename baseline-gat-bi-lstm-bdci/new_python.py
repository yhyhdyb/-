# %%
import pandas as pd
import numpy as np
# 读取CSV文件
import my_model
import  torch
import argparse
from tqdm import tqdm, trange
import data
import torch.nn as nn


criterion = nn.MSELoss()  # 使用均方误差损失函数计算MSE
#%%
#获得训练集
def get_train_data(file_path,edge_pth):
    df = pd.read_csv(file_path, encoding='utf-8')
    edge_df = pd.read_csv(edge_pth, encoding='utf-8')
    df.head()

    geohasd_df_dict = {}
    date_df_dict = {}
    number_hash = 0
    number_date = 0
    for i in df["geohash_id"]:

        if i not in geohasd_df_dict.keys():
            geohasd_df_dict[i] = number_hash
            number_hash += 1

    for i in df["date_id"]:
        if i not in date_df_dict.keys():
            date_df_dict[i] = number_date
            number_date += 1

    # new_data 的二维列表。它的大小是 len(date_df_dict) 行乘以 len(geohasd_df_dict) 列。
    # 具体地，代码首先使用 geohasd_df_dict 将地理哈希编码转换为 new_data 的列索引，
    # 使用 date_df_dict 将日期转换为 new_data 的行索引，
    # 然后将新数据填入 new_data 对应位置的单元格中。
    # 这可以使用两个索引进行操作：date_index （行）和 hash_index （列），
    # 它们是通过哈希表存储的，因此读取非常快。

    new_data = [len(geohasd_df_dict) * [0]] * len(date_df_dict)
    for index, row in df.iterrows():
        # print(index)
        hash_index, date_index = geohasd_df_dict[row["geohash_id"]], date_df_dict[row["date_id"]]
        # 将时间index加到里面
        new_data[date_index][hash_index] = [date_index]+list(row.iloc[2:])
        # new_data[date_index][hash_index] = [date_index]+list(row.iloc[2:37])+[0,0]+list(row.iloc[:, -2:])
    new_data = np.array(new_data)

    # x_train,y_train = new_data[:, :-2], new_data[:, -2:]
    # print(len(geohasd_df_dict))
    # exit()
    # print(x_train.shape)
    # print(y_train.shape)
    #这里构建邻接矩阵其中mask表示1为有边，0无边， value_mask表示有值
    #并且这里我考虑mask是一个无向图，如果有向删除x_mask[date_index][point2_index][point1_index],value_mask同理
   
    x_mask =  np.zeros((len(date_df_dict),len(geohasd_df_dict),len(geohasd_df_dict),1), dtype = float)
    x_edge_df =np.zeros((len(date_df_dict),len(geohasd_df_dict),len(geohasd_df_dict),2), dtype = float)

    for index, row in edge_df.iterrows():
        # print(index)
        if row["geohash6_point1"] not in geohasd_df_dict.keys() or row["geohash6_point2"] not in geohasd_df_dict.keys():
            continue
        point1_index,point2_index,F_1,F_2,date_index= geohasd_df_dict[row["geohash6_point1"]],geohasd_df_dict[row["geohash6_point2"]]\
            ,row["F_1"],row["F_2"],date_df_dict[row["date_id"]]
        
        # 将 x_mask 中对应位置的值设为1，表示该位置存在边。
        x_mask[date_index][point1_index][point2_index] = 1
        x_mask[date_index][point2_index][point1_index] = 1

        # 同时，将 x_edge_df 中对应位置的值设为 [F_1, F_2]，即该位置的特征向量。
        x_edge_df[date_index][point1_index][point2_index] =  [F_1,F_2]
        x_edge_df[date_index][point2_index][point1_index] = [F_1, F_2]

        #将[F_1,F_2]作为两个节点新的特征值

        # new_data[date_index][point1_index]=np.concatenate((new_data[date_index][point1_index][:35], [-F_1, F_2], new_data[date_index][point1_index][35:]))
        # new_data[date_index][point1_index]=np.concatenate((new_data[date_index][point1_index][:35], [-F_1, F_2], new_data[date_index][point1_index][35:]))

    # print(data)

    return     geohasd_df_dict, date_df_dict, new_data,x_mask, x_edge_df
#%%
def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():

        dev_loss = 0.0
        # df_reg = pd.DataFrame(columns=('geohash_id','consumption_level','activity_level','date_id'))
        for j in trange(dataset.batch_count):
            x_date, x_feature, x_mask_data, x_edge_data, x_tags = dataset.get_batch(j)
            act_pre, con_pre = model(x_date, x_feature, x_mask_data)
            # torch.cat函数将两个张量act_pre和con_pre沿着最后一个维度(dim=-1)进行拼接的代码。
            # 拼接后的张量称为predict。拼接的结果是将act_pre和con_pre按照最后一个维度进行连接，形成一个新的张量。
            predict = torch.cat((act_pre, con_pre), dim=-1)
            # print('{0}-predict:{1}'.format(j,predict))
            loss = criterion(predict, x_tags)
            dev_loss+= loss
        print("this epoch dev loss is {}".format(dev_loss))
        model.train()

def test(args):

    #使用get_train_data函数获取测试数据和标签。
    geohasd_df_dict, date_df_dict, x_train, x_mask, x_edge_df = get_train_data('./node_test_4_A.csv',
                                                                                        "./edge_test_4_A.csv")
    # 转化为张量
    x_test,x_mask_test,x_edge_test=x_train,x_mask,x_edge_df
    x_test = torch.tensor(x_test)
    x_mask_test = torch.tensor(x_mask_test)
    x_edge_test = torch.tensor(x_edge_test)

    #加载模型
    date_emb = 5
    model = my_model.BILSTM(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    model.load_state_dict(torch.load("model_32_500_0.01.pth"))
    model.eval()

    #创建迭代器
    testset=data.DataIterator2(x_test,x_mask_test,x_edge_test,args)
    x_date,x_feature,x_mask_data,x_edge_data= testset.get_data()
    act_pre, con_pre = model(x_date,x_feature,x_mask_data)
    print(act_pre)
    
    # 将多维数组展平为一维数组
    con_flat=con_pre.flatten()
    act_flat=act_pre.flatten()

    # 读取模板文件
    df=pd.read_csv("submit_example.csv",encoding='utf-8',delimiter='\t')

    # 填充预测值
    df["consumption_level"]=con_flat.detach().cpu().numpy()
    df["activity_level"]=act_flat.detach().cpu().numpy()

    # 生成新文件
    df.to_csv("submit_32_500_0.01.csv", sep="\t", index=False, encoding="utf-8")

#%%
def train(args):

    #使用get_train_data函数获取训练数据和标签。
    geohasd_df_dict, date_df_dict, x_train, x_mask, x_edge_df = get_train_data('./train_90.csv',
                                                                                        "./edge_90.csv")
    #分割各种训练集测试集
    x_train,x_dev = torch.tensor(x_train[:int(len(x_train)*args.rat)]),torch.tensor(x_train[int(len(x_train)*args.rat):])
    x_mask_train,x_mask_dev = torch.tensor(x_mask[:int(len(x_mask)*args.rat)]),torch.tensor(x_mask[int(len(x_mask)*args.rat):])
    x_edge_train, x_edge_dev = torch.tensor(x_edge_df[:int(len(x_edge_df) * args.rat)]),torch.tensor( x_edge_df[int(len(x_edge_df) * args.rat):])

    # df_reg = pd.DataFrame(columns=('geohash_id','consumption_level','activity_level','date_id'))

    date_emb = 5
     # 这里的x包含了date_id+F35个特征+2个y值的
    # train_activate = torch.tensor(y_train[:, -2])
    # train_consume = torch.tensor(y_train[:, -1])


    # rmse_loss = torch.sqrt(mse_loss)

    #定义模型并将其移动到指定设备上。
    # model = my_model.GAT(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    model = my_model.BILSTM(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    #定义优化器（使用Adam算法）和损失函数（criterion）
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)
    model.train()

    #创建训练集和验证集的数据迭代器。
    trainset = data.DataIterator(x_train,x_mask_train,x_edge_train, args)
    valset =data.DataIterator(x_dev,x_mask_dev,x_edge_dev, args)
    for indx in range(args.epochs):
        train_all_loss = 0.0
        for j in trange(trainset.batch_count):
            #获取当前batch的数据（日期、特征、掩码、边数据和标签）
            x_date,x_feature,x_mask_data,x_edge_data,x_tags= trainset.get_batch(j)
            # print('tags:')
            # print(x_tags)
            act_pre, con_pre = model(x_date,x_feature,x_mask_data)
            predict = torch.cat((act_pre, con_pre), dim=-1)
            # print('predict:')
            print(predict[0][0][0])

            #计算预测结果和标签之间的损失。
            loss = criterion(predict, x_tags)
  
            train_all_loss += loss

            #清除优化器的梯度。
            optimizer.zero_grad()

            #反向传播并更新模型的参数。
            loss.backward()
        optimizer.step()
        torch.save(model.state_dict(),"model_32_500_0.01.pth") # 保存模型
        print('{0} — this epoch train loss :{1}'.format(indx,train_all_loss))
        # scheduler.step()

        #调用eval函数对验证集进行评估。
        eval(model,valset, args)

#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=500,
                        help='training epoch number')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size')
    parser.add_argument('--device', type=str, default="cpu",
                        help='gpu or cuda')
    parser.add_argument('--lr', type=float, default=1e-2,
                        )
    parser.add_argument('--rat', type=float, default=0.9,)

    parser.add_argument('--decline', type=int, default=30, help="number of epochs to decline")
    
    # test(parser.parse_args())
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
# edge = pd.read_csv('./edge_90.csv')
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
