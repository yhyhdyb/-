import pandas as pd
import numpy as np
# 读取CSV文件

import  torch
import argparse
from tqdm import tqdm, trange

import torch.nn as nn
#获得训练集

criterion = nn.MSELoss()  # 使用均方误差损失函数计算MSE
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

    new_data = [len(geohasd_df_dict) * [0]] * len(date_df_dict)
    for index, row in df.iterrows():
        # print(index)
        hash_index, date_index = geohasd_df_dict[row["geohash_id"]], date_df_dict[row["date_id"]]
        #将时间index加到里面

        new_data[date_index][hash_index] = [date_index]+list(row.iloc[2:])
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
        x_mask[date_index][point1_index][point2_index] = 1
        x_mask[date_index][point2_index][point1_index] = 1
        x_edge_df[date_index][point1_index][point2_index] =  [F_1,F_2]
        x_edge_df[date_index][point2_index][point1_index] = [F_1, F_2]
    # print(data)

    return     geohasd_df_dict, date_df_dict, new_data,x_mask, x_edge_df
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.xavier_uniform_(self.weight)  # xavier初始化，就是论文里的glorot初始化
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        support = torch.mm(self.dropout(inputs), self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, n_features, hidden_dim, dropout, n_classes):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(n_features, hidden_dim, dropout)
        self.gc2 = GraphConvolution(hidden_dim, n_classes, dropout)
        self.relu = nn.ReLU()

    def forward(self, inputs, adj):
        x = inputs
        x = self.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        self.a = nn.Parameter(torch.zeros(2 * out_features, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        '''
        h: (N, in_features)
        adj: sparse matrix with shape (N, N)
        p
        '''
        adj=torch.squeeze(adj,-1)
        # print(h.dtype)
        # print(h.shape)

        Wh = torch.matmul(h, self.W)  # (N, out_features)

        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])  # (N, 1)
        # print(Wh1.shape)
        # print(Wh2.shape)

        # Wh1 + Wh2.T 是N*N矩阵，第i行第j列是Wh1[i]+Wh2[j]
        # 那么Wh1 + Wh2.T的第i行第j列刚好就是文中的a^T*[Whi||Whj]
        # 代表着节点i对节点j的attention
        # print(torch.transpose(Wh2,2,1).shape)
        e = self.leakyrelu(Wh1 +torch.transpose(Wh2,2,1))  # (N, N)
        padding = (-2 ** 31) * torch.ones_like(e)  # (N, N)
        # print(adj.shape)
        # print(padding.shape)
        attention = torch.where(adj > 0, e, padding)  # (N, N)
        attention = F.softmax(attention, dim=1)  # (N, N)
        # attention矩阵第i行第j列代表node_i对node_j的注意力
        # 对注意力权重也做dropout（如果经过mask之后，attention矩阵也许是高度稀疏的，这样做还有必要吗？）
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GAT(nn.Module):
    def __init__(self,date_emb, nfeat, nhid, dropout, alpha, nheads):
        super(GAT, self).__init__()
        date_index_number,date_dim = date_emb[0], date_emb[1]
        self.dropout = dropout
        self.MH = nn.ModuleList([
            GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout, alpha, concat=False)
        self.date_embdding = nn.Embedding(date_index_number,date_dim)
        self.active_index = nn.Linear(nhid,1)
        self.consume_index = nn.Linear(nhid,1)
    def forward(self,x_date,x_feature,x_mask_data):


        x = x_feature
        # x = F.dropout(x_feature, self.dropout, training=self.training)  # (N, nfeat)
        x = torch.cat([head(x, x_mask_data) for head in self.MH], dim=-1)  # (N, nheads*nhid)
        x = F.dropout(x, self.dropout, training=self.training)  # (N, nfeat)


        # x = F.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        x = self.out_att(x, x_mask_data)
        # print(x.shape,x.dtype)
        act_pre= self.active_index(x)
        con_pre = self.consume_index(x)
        return  act_pre,con_pre


class BILSTM(nn.Module):
    def __init__(self,date_emb, nfeat, nhid, dropout, alpha, nheads):
        super(BILSTM, self).__init__()
        date_index_number,date_dim = date_emb[0], date_emb[1]
        self.dropout = dropout
        self.lstm = nn.LSTM(nfeat,
                nhid,
                num_layers=2,
                bias=True,
                batch_first=False,
                dropout=0,
                bidirectional=True)

        self.active_index = nn.Linear(2*nhid, 1)
        self.consume_index = nn.Linear(2*nhid, 1)
    def forward(self,x_date,x_feature,x_mask_data):
        lstm_out, (hidden, cell) = self.lstm(x_feature)
        x = lstm_out
        # print(x.shape)


        x = F.dropout(x, self.dropout, training=self.training)  # (N, nheads*nhid)
        act_pre= self.active_index(x)
        con_pre = self.consume_index(x)
        # print(act_pre.shape,con_pre.shape)
        return  act_pre,con_pre
def eval(model, dataset, args):
    model.eval()
    with torch.no_grad():

        dev_loss = 0.0
        for j in trange(dataset.batch_count):
            x_date, x_feature, x_mask_data, x_edge_data, x_tags = dataset.get_batch(j)
            act_pre, con_pre = model(x_date, x_feature, x_mask_data)
            predict = torch.cat((act_pre, con_pre), dim=-1)
            loss = criterion(predict, x_tags)
            dev_loss+= loss
        print("this epoch dev loss is {}".format(dev_loss))
        model.train()
class DataIterator(object):
    def __init__(self, x_data,x_mask_data,x_edge_data, args):
        self.x_data,self.x_mask_data,self.x_edge_data,=x_data,x_mask_data,x_edge_data,
        #date跟fearture的分开
        self.x_date,self.x_feature,self.x_tags=self.x_data[:,:,0],self.x_data[:,:,1:-2],x_data[:,:,-2:]
        # print(self.x_date.shape,self.x_feature.shape,self.x_tags.shape)
        self.args = args
        self.batch_count = math.ceil(len(x_data)/args.batch_size)

    def get_batch(self, index):
        x_date = []
        x_feature = []
        x_mask_data=[]
        x_edge_data = []
        x_tags = []


        for i in range(index * self.args.batch_size,
                       min((index + 1) * self.args.batch_size, len(self.x_data))):

            x_date.append(self.x_date[i])
            x_feature.append(self.x_feature[i].float() )

            # print(self.x_mask_data[i].shape)
            x_mask_data.append(self.x_mask_data[i])
            # print(self.x_edge_data[i].shape)
            x_edge_data.append(self.x_edge_data[i])
            x_tags.append(self.x_tags[i].float() )

        x_date = torch.stack(x_date).to(self.args.device)
        x_feature = torch.FloatTensor(torch.stack(x_feature)).to(self.args.device)
        x_mask_data = torch.stack(x_mask_data).to(self.args.device)
        x_edge_data = torch.stack(x_edge_data).to(self.args.device)
        x_tags = torch.stack(x_tags).to(self.args.device)


        return  x_date,x_feature,x_mask_data,x_edge_data,x_tags

def train(args):

    geohasd_df_dict, date_df_dict, x_train, x_mask, x_edge_df = get_train_data('./train_90.csv',
                                                                                        "./edge_90.csv")
    #分割各种训练集测试集
    x_train,x_dev = torch.tensor(x_train[:int(len(x_train)*args.rat)]),torch.tensor(x_train[int(len(x_train)*args.rat):])
    x_mask_train,x_mask_dev = torch.tensor(x_mask[:int(len(x_mask)*args.rat)]),torch.tensor(x_mask[int(len(x_mask)*args.rat):])
    x_edge_train, x_edge_dev = torch.tensor(x_edge_df[:int(len(x_edge_df) * args.rat)]),torch.tensor( x_edge_df[int(len(x_edge_df) * args.rat):])



    date_emb = 5
     # 这里的x包含了date_id+F35个特征+2个y值的
    # train_activate = torch.tensor(y_train[:, -2])
    # train_consume = torch.tensor(y_train[:, -1])


    # rmse_loss = torch.sqrt(mse_loss)
    model = GAT(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    # model = BILSTM(date_emb =[len(date_df_dict),date_emb], nfeat=35, nhid=64, dropout=0.3, alpha=0.3, nheads=8).to(args.device)
    optimizer = torch.optim.Adam(params=model.parameters(),lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decline, gamma=0.5, last_epoch=-1)
    model.train()
    trainset = DataIterator(x_train,x_mask_train,x_edge_train, args)
    valset =DataIterator(x_dev,x_mask_dev,x_edge_dev, args)
    for indx in range(args.epochs):
        train_all_loss = 0.0
        for j in trange(trainset.batch_count):
            x_date,x_feature,x_mask_data,x_edge_data,x_tags= trainset.get_batch(j)
            act_pre, con_pre = model(x_date,x_feature,x_mask_data)
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

