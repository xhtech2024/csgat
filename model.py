import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.nn import HANConv
from buildgraph import HHG




def parse_entity(entity_str, tokenizer):
    entity = {}
    parts = entity_str.split("COL ")
    for part in parts:
        if part.strip():
            key_val = part.split("VAL ")
            if len(key_val) == 2:
                key = key_val[0].strip()
                val = key_val[1].strip()
                tokens = tokenizer.tokenize(val)
                entity[key] = tokens
    return entity


def process_data(raw_path, tokenizer,dataset):

    data_list = []
    processed_path = f'data/{dataset}/processed/' + os.path.basename(raw_path).replace(".txt", ".pt")
    if os.path.exists(processed_path):
        data_list = torch.load(processed_path)
        return data_list
    else:
        with open(raw_path, 'r') as file:
            for line in tqdm(file, desc=f'Processing {os.path.basename(raw_path)}'):
                parts = line.strip().split("\t")
                entity1_str = parts[0]
                entity2_str = parts[1]
                label = int(parts[2])

                entity1 = parse_entity(entity1_str, tokenizer)
                entity2 = parse_entity(entity2_str, tokenizer)

                hhg = HHG()  # 构造
                hhg.add_entity(entity1, "entity1")
                hhg.add_entity(entity2, "entity2")

                data = hhg.build_graph()
                data.y = torch.tensor([label])  # 添加label

                data_list.append(data)  # 将一个异构图加入到数据集中

            torch.save(data_list, processed_path)

    return data_list


class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,transformed_data):
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, metadata=transformed_data.metadata(), heads=8)
        self.conv2 = HANConv(hidden_channels, out_channels, metadata=transformed_data.metadata(), heads=4)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict
    
from torch_geometric.nn import RGCNConv


import torch
import torch.nn as nn
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
class RGCNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(RGCNModel, self).__init__()
        self.conv1 = RGCNConv(in_channels, out_channels, num_relations)
        self.conv2 = RGCNConv(out_channels, out_channels, num_relations)


    def forward(self, x_dict, edge_index_dict, edge_type_dict):
        # 初始化新的 x_dict 用于存储更新后的节点特征
        new_x_dict = {key: torch.zeros_like(x) for key, x in x_dict.items()}
        
        # 遍历 edge_index_dict 的键
        for (src_type, _, dst_type), edge_index in edge_index_dict.items():
            edge_type = edge_type_dict[(src_type, _, dst_type)]
            x_src = x_dict[src_type]
            x_dst = x_dict[dst_type]
            
            # 进行 RGCN 卷积
            x_dst = self.conv1((x_src, x_dst), edge_index, edge_type)
            x_dst = F.relu(x_dst)
            x_dst = self.conv2((x_src, x_dst), edge_index, edge_type)
            
            # 更新新的 x_dict
            new_x_dict[dst_type] += x_dst
        # 提取 entity 节点的
        
        return new_x_dict
    
from torch_geometric.nn import global_mean_pool


class MatchingModel_RGCN(nn.Module):
    def __init__(self, rgcn_model, hidden_dim):
        super(MatchingModel_RGCN, self).__init__()
        self.rgcn = rgcn_model
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x_dict = self.rgcn(data.x_dict, data.edge_index_dict, data.edge_type_dict)
        entity_embeddings = x_dict['entity']  # 抽取entity  

        # 获取批次大小
        batch_size = entity_embeddings.size(0) // 2
        
        # 初始化列表存储拼接后的嵌入
        concatenated_embeddings = []
        
        for i in range(batch_size):
            # 提取成对的实体嵌入
            entity1_embedding = entity_embeddings[2 * i].flatten()
            entity2_embedding = entity_embeddings[2 * i + 1].flatten()
            
            # 拼接实体嵌入
            concatenated_embedding = torch.cat((entity1_embedding, entity2_embedding), dim=0)
            concatenated_embeddings.append(concatenated_embedding)
        
        # 将列表转换为张量
        concatenated_embeddings = torch.stack(concatenated_embeddings)
        
        # 接入全连接层
        output = self.fc2(concatenated_embeddings)
        output = self.dropout(output)
        logits = self.fc(output)
        y_hat = logits.argmax(dim=-1)  # 预测结果
        return logits, y_hat



class MatchingModel(nn.Module):
    def __init__(self, han_model, hidden_dim):
        super(MatchingModel, self).__init__()
        self.han = han_model
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, data):
        x_dict = self.han(data.x_dict, data.edge_index_dict)
        entity_embeddings = x_dict['entity']  # 抽取entity  

        
        # 获取批次大小
        batch_size = entity_embeddings.size(0) // 2
        
        # 初始化列表存储拼接后的嵌入
        concatenated_embeddings = []
        
        for i in range(batch_size):
            # 提取成对的实体嵌入
            entity1_embedding = entity_embeddings[2 * i].flatten()
            entity2_embedding = entity_embeddings[2 * i + 1].flatten()
            
            # 拼接实体嵌入
            concatenated_embedding = torch.cat((entity1_embedding, entity2_embedding), dim=0)
            concatenated_embeddings.append(concatenated_embedding)
        
        # 将列表转换为张量
        concatenated_embeddings = torch.stack(concatenated_embeddings)
        
        # 接入全连接层
        output = self.fc2(concatenated_embeddings)
        output = self.dropout(output)
        logits = self.fc(output)
        # 使用全连接层进行分类
        #logits = torch.sigmoid(logits)
        y_hat = logits.argmax(dim=-1)  # 预测结果
        return logits, y_hat
