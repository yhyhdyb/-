import math

import torch
import numpy as np
import  pickle
import os
import json

class DataIterator(object):
    def __init__(self, x_data,x_mask_data,x_edge_data, args):

        #把日期、特征和标签分开，并在对象中添加了一些其他的成员变量，比如批次数量等等。
        self.x_data,self.x_mask_data,self.x_edge_data,=x_data,x_mask_data,x_edge_data,
        #date跟fearture的分开
        self.x_date,self.x_feature,self.x_tags=self.x_data[:,:,0],self.x_data[:,:,1:-2],x_data[:,:,-2:]
        # print(self.x_date.shape,self.x_feature.shape,self.x_tags.shape)
        self.args = args
        self.batch_count = math.ceil(len(x_data)/args.batch_size)

# 获取在特定索引处的一组数据批次。
# 通过这种方法，我们可以按照顺序批量加载数据，
# 每次获取一个小部分的数据，以避免在内存不足的情况下出现内存错误。
# 在这个方法中，先遍历一部分数据，按照指定顺序进行组合和筛选，
# 然后将它们转换为 PyTorch 张量格式，并返回它们。
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
    

    
    


class DataIterator2(object):
    def __init__(self, x_data,x_mask_data,x_edge_data, args):

        #把日期、特征和标签分开，并在对象中添加了一些其他的成员变量，比如批次数量等等。
        self.x_data,self.x_mask_data,self.x_edge_data,=x_data,x_mask_data,x_edge_data,
        #date跟fearture的分开
        self.x_date,self.x_feature=self.x_data[:,:,0],self.x_data[:,:,1:]
        # print(self.x_date.shape,self.x_feature.shape,self.x_tags.shape)
        self.args = args
        self.batch_count = math.ceil(len(x_data)/args.batch_size)

    def get_data(self):
        x_date = []
        x_feature = []
        x_mask_data=[]
        x_edge_data = []

        for i in range(len(self.x_data)):

            x_date.append(self.x_date[i])
            x_feature.append(self.x_feature[i].float() )

            # print(self.x_mask_data[i].shape)
            x_mask_data.append(self.x_mask_data[i])
            # print(self.x_edge_data[i].shape)
            x_edge_data.append(self.x_edge_data[i])

        x_date = torch.stack(x_date).to(self.args.device)
        x_feature = torch.FloatTensor(torch.stack(x_feature)).to(self.args.device)
        x_mask_data = torch.stack(x_mask_data).to(self.args.device)
        x_edge_data = torch.stack(x_edge_data).to(self.args.device)


        return  x_date,x_feature,x_mask_data,x_edge_data