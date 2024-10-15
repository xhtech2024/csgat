import os
from typing import List
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData,InMemoryDataset
from transformers import BertTokenizer
from torch_geometric.nn import HANConv
import torch_geometric.transforms as T
from torch_geometric.datasets import IMDB,GNNBenchmarkDataset
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from sklearn.metrics import f1_score
import os.path as osp
from utils import visualize_hetero_graph

# class HHG:
#     def __init__(self):
#         self.entity_nodes = {}
#         self.attribute_nodes = {}
#         self.token_nodes = {}
#         self.edges = {"entity_attri": [], "attri_word": []}

#     def add_entity(self, entity, entity_id):
#         if entity_id not in self.entity_nodes:
#             self.entity_nodes[entity_id] = len(self.entity_nodes)
#         entity_node_idx = self.entity_nodes[entity_id]
        
#         for key, words in entity.items():
#             attr_node_id = f"{entity_id}_{key}"
#             if attr_node_id not in self.attribute_nodes:
#                 self.attribute_nodes[attr_node_id] = len(self.attribute_nodes)
#             attr_node_idx = self.attribute_nodes[attr_node_id]
#             self.edges["entity_attri"].append((entity_node_idx, attr_node_idx))
            
#             for word in words:
#                 if word not in self.token_nodes:
#                     self.token_nodes[word] = len(self.token_nodes)
#                 token_node_idx = self.token_nodes[word]
#                 self.edges["attri_word"].append((attr_node_idx, token_node_idx))

#     def build_graph(self):
#         num_entity_nodes = len(self.entity_nodes)
#         num_attr_nodes = len(self.attribute_nodes)
#         num_token_nodes = len(self.token_nodes)

#         data = HeteroData()
        
#         data['entity'].x = torch.randn((num_entity_nodes, 768))  # 假设嵌入维度为128
#         data['attri'].x = torch.randn((num_attr_nodes, 768))  
#         data['word'].x = torch.randn((num_token_nodes, 768))
        
#         entity_attri_edge_index = torch.tensor(self.edges["entity_attri"], dtype=torch.long).t().contiguous()
#         attri_word_edge_index = torch.tensor(self.edges["attri_word"], dtype=torch.long).t().contiguous()
        
#         # Add forward edges
#         data['entity', 'has', 'attri'].edge_index = entity_attri_edge_index
#         data['attri', 'contains', 'word'].edge_index = attri_word_edge_index

#         data['attri', 'is_contained_in', 'entity'].edge_index = entity_attri_edge_index.flip(0)
#         data['word', 'contains_attri', 'attri'].edge_index = attri_word_edge_index.flip(0)
        
#         # Add backward edges
#         # data['attri', 'has', 'entity'].edge_index = entity_attri_edge_index.t().contiguous()
#         # data['word', 'contains', 'attri'].edge_index = attri_word_edge_index.t().contiguous()

#         return data

from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import copy
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device}")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)

# 将模型设置为评估模式
bert.eval()
def generate_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()


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

class HHG:
    def __init__(self):
        self.entity_nodes = {}
        self.attribute_nodes = {}
        self.token_nodes = {}
        self.edges = {"entity_attri": [], "attri_word": []}
        self.entity_embeddings = {}
        self.attribute_embeddings = {}
        self.token_embeddings = {}

    def add_entity(self, entity, entity_id):
        if entity_id not in self.entity_nodes:
            self.entity_nodes[entity_id] = len(self.entity_nodes)
            self.entity_embeddings[entity_id] = generate_bert_embedding(entity_id)
        entity_node_idx = self.entity_nodes[entity_id]
        
        for key, words in entity.items():
            attr_node_id = f"{entity_id}_{key}"
            if attr_node_id not in self.attribute_nodes:
                self.attribute_nodes[attr_node_id] = len(self.attribute_nodes)
                self.attribute_embeddings[attr_node_id] = generate_bert_embedding(attr_node_id)
            attr_node_idx = self.attribute_nodes[attr_node_id]
            self.edges["entity_attri"].append((entity_node_idx, attr_node_idx))
            
            for word in words:
                if word not in self.token_nodes:
                    self.token_nodes[word] = len(self.token_nodes)
                    self.token_embeddings[word] = generate_bert_embedding(word)
                token_node_idx = self.token_nodes[word]
                self.edges["attri_word"].append((attr_node_idx, token_node_idx))

    def build_graph(self):
        num_entity_nodes = len(self.entity_nodes)
        num_attr_nodes = len(self.attribute_nodes)
        num_token_nodes = len(self.token_nodes)

        data = HeteroData()
        
        # 使用 BERT 嵌入
        data['entity'].x = torch.stack([self.entity_embeddings[entity_id] for entity_id in self.entity_nodes]).to(device)
        data['attri'].x = torch.stack([self.attribute_embeddings[attr_node_id] for attr_node_id in self.attribute_nodes]).to(device)
        data['word'].x = torch.stack([self.token_embeddings[word] for word in self.token_nodes]).to(device)
        
        
        entity_attri_edge_index = torch.tensor(self.edges["entity_attri"], dtype=torch.long).t().contiguous().to(device)
        attri_word_edge_index = torch.tensor(self.edges["attri_word"], dtype=torch.long).t().contiguous().to(device)
        
        edge_types = {
        ('entity', 'to', 'attri'): 0,
        ('attri', 'to', 'word'): 1,
        ('attri', 'to', 'entity'): 2,
        ('word', 'to', 'attri'): 3
        }
        # Add forward edges
        data['entity', 'to', 'attri'].edge_index = entity_attri_edge_index
        data['attri', 'to', 'word'].edge_index = attri_word_edge_index
  
        data['attri', 'to', 'entity'].edge_index = entity_attri_edge_index.flip(0).contiguous()
        data['word', 'to', 'attri'].edge_index = attri_word_edge_index.flip(0).contiguous()

            # 添加边类型
        data['entity', 'to', 'attri'].edge_type = torch.full((entity_attri_edge_index.size(1),), edge_types[('entity', 'to', 'attri')], dtype=torch.long, device=device)
        data['attri', 'to', 'word'].edge_type = torch.full((attri_word_edge_index.size(1),), edge_types[('attri', 'to', 'word')], dtype=torch.long, device=device)
        data['attri', 'to', 'entity'].edge_type = torch.full((entity_attri_edge_index.size(1),), edge_types[('attri', 'to', 'entity')], dtype=torch.long, device=device)
        data['word', 'to', 'attri'].edge_type = torch.full((attri_word_edge_index.size(1),), edge_types[('word', 'to', 'attri')], dtype=torch.long, device=device)
        

        data = data.to(device)



        return data




# 示例数据文件路径
train_data_file = './data/Beer/train.txt'

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




# for key, value in data.edge_index_dict.items():
#     if isinstance(value, torch.Tensor):
#         print(f"Key:{key},Value: {value}")

#         if not value.is_contiguous():
#             print("不连续")

#         else:
#             print("连续")





# class MyOwnDataset(InMemoryDataset):
#     """
    
#     Args:
#         root (str): Root directory where the dataset should be saved.
#         name (str): The name of the dataset.(one of:obj:'Amazon-Google', 'Beer')
#         split (str): The dataset split to return. Can be 'train', 'val', or 'test'.
    
#     transform:
#             metapaths = [
#                 [('entity', 'has', 'attri'), ('attri', 'is_contained_in', 'entity')]
#             ]
#             transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True)

#     """

#     names = ['Amazon-Google', 'Beer']

#     def __init__(self,
#                  root:str='./data/',
#                  name:str="Beer",
#                  split: str ="train",
#                  transform=None,
#                  pre_transform=None,
#                  force_reload=False):
        
        
#         self.name = name
#         self.root = root+name
#         assert self.name in self.names
#         super().__init__(self.root, transform, pre_transform,force_reload=force_reload)

#         if split == 'train':
#             path = self.processed_paths[0]
#         elif split == 'val':
#             path = self.processed_paths[1]
#         elif split == 'test':
#             path = self.processed_paths[2]
#         else:
#             raise ValueError(f"Split '{split}' found, but expected either "
#                              f"'train', 'val', or 'test'")
#         self.load(path,data_cls=HeteroData)


    
#     @property
#     def raw_file_names(self) -> List[str]:
#         return ['train.txt','valid.txt','test.txt']

#     @property
#     def processed_file_names(self):
#         return ['train_data.pt', 'val_data.pt','test_data.pt']

#     def process(self):
      
#         for i, raw_path in enumerate(self.raw_paths): # 遍历3个文件
#             data_list = []
#             with open(raw_path, 'r') as file:
        
#                 for line in tqdm(file, desc=f'Processing {os.path.basename(raw_path)}'):
#                     parts = line.strip().split("\t")
#                     entity1_str = parts[0]
#                     entity2_str = parts[1]
#                     label = int(parts[2])

#                     entity1 = parse_entity(entity1_str, tokenizer)
#                     entity2 = parse_entity(entity2_str, tokenizer)

#                     hhg = HHG()  # 构造
#                     hhg.add_entity(entity1, "entity1")
#                     hhg.add_entity(entity2, "entity2")

#                     data = hhg.build_graph()
#                     data.y = torch.tensor([label])  # 添加label
#                     # Deep copy the data list
#                     #data_copy = copy.deepcopy(data)
#                     data_list.append(data)  # 将一个异构图加入到数据集中
                    
#             self.save(data_list, path=self.processed_paths[i])
  

  



# print("数据集构建中")
# train_dataset = MyOwnDataset(name='Beer',split='test')
# num_graphs = len(train_dataset)
# first_graph = train_dataset.get(0)
# print(first_graph)

# for key, value in first_graph.edge_index_dict.items():
#     if isinstance(value, torch.Tensor):
#         print(f"Key:{key},Value: {value}")

#         if not value.is_contiguous():
#             print("不连续")
#             value = value.contiguous()
#             print(value.is_contiguous())
#             first_graph.edge_index_dict[key] = value.contiguous()
#             first_graph.edge_index_dict[key] = first_graph.edge_index_dict[key].contiguous()
        
#             print(first_graph.edge_index_dict[key].is_contiguous())
#         else:
#             print("连续")
# print("数据集构建完成")
# valid_dataset = MyOwnDataset(name='Beer',split='val')
# test_dataset = MyOwnDataset(name='Beer',split='test')

# # print(f"Train dataset len: {train_dataset.len()}")
# # print(f"Valid dataset len: {valid_dataset.len()}")
# # print(f"Test dataset len: {test_dataset.len()}")
# # t = train_dataset[0]
# # for key, value in t.edge_index_dict.items():
# #     if isinstance(value, torch.Tensor):
# #         if not value.is_contiguous():
# #             print("不连续")
# #             value = value.contiguous()
# #             t.edge_index_dict[key] = value
# #         else:
# #             print("连续")

# # print(t.y)

# # v = valid_dataset[1]
# # print(v.y)
# # tt = test_dataset[1]
# # print(tt.y)

# # visualize_hetero_graph(t)

# # print("数据集构建完成")

# # print(train_dataset[0].edge_types)


# print("元路径测试")
# metapaths = [
#                 [('attri', 'to', 'word'), ('word', 'to', 'attri')],
#                 [('entity', 'to', 'attri'), ('attri', 'to', 'word'), ('word', 'to', 'attri'),('attri', 'to', 'entity')],
#             ]
# transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False, drop_unconnected_node_types=False)

# data = train_dataset[0]
# for key, value in data.edge_index_dict.items():
#     if isinstance(value, torch.Tensor):
#         if not value.is_contiguous():
#             print("不连续")
#             value = value.contiguous()
#             data.edge_index_dict[key] = value.contiguous()
#         else:
#             print("连续")
# for key, value in data.edge_index_dict.items():
#     if isinstance(value, torch.Tensor):
#         if not value.is_contiguous():
#             print("不连续!")
#             value = value.contiguous()
            
#             data.edge_index_dict[key] = value
#         else:
#             print("连续!")


# trai = transform(train_dataset[0])
# print(trai.metadata())
# # t = transform_train_dataset[0]
# # print(t.y)
# # visualize_hetero_graph(t,output_file='metapath_transform.png')


# class MatchingModel(nn.Module):
#     def __init__(self, han_model, hidden_dim):
#         super(MatchingModel, self).__init__()
#         self.han = han_model
#         self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
#         self.fc = nn.Linear(hidden_dim, 2)

#     def forward(self, data):
#         x_dict = self.han(data.x_dict, data.edge_index_dict)
#         entity_embeddings = x_dict['entity']
        
#         # 假设我们有一对实体 (entity1, entity2) 需要判断是否匹配
#         entity1_embedding = entity_embeddings[0].flatten()
#         entity2_embedding = entity_embeddings[1].flatten()
        
#         # 拼接实体嵌入
#         concatenated_embedding = torch.cat((entity1_embedding, entity2_embedding), dim=0)
        
#         # 接入全连接层
#         output = self.fc2(concatenated_embedding)
#         output = self.fc(output)
#         # 计算相似度（例如，使用点积）
#         #similarity = torch.dot(entity1_embedding, entity2_embedding)
        
#         # 使用全连接层进行分类
#         #output = self.fc(similarity)
#         return torch.sigmoid(output)


# class HAN(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, metadata):
#         super(HAN, self).__init__()
#         self.conv1 = HANConv(in_channels, hidden_channels, metadata=metadata, heads=8)
#         self.conv2 = HANConv(hidden_channels, out_channels, metadata=metadata, heads=4)

#     def forward(self, x_dict, edge_index_dict):
#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict = {key: x.relu() for key, x in x_dict.items()}
#         x_dict = self.conv2(x_dict, edge_index_dict)
#         return x_dict
    
# # 假设我们有一个数据加载器 data_loader
# entity_feature_dim = 768  # 假设实体特征维度为128
# attri_feature_dim = 768    # 假设属性特征维度为64
# word_feature_dim = 768     # 假设词特征维度为32

# # 选择最大的特征维度作为输入特征维度
# in_channels = max(entity_feature_dim, attri_feature_dim, word_feature_dim)
# hidden_channels = 768  # 隐藏层特征维度
# out_channels = 768     # 输出特征维度
# hidden_dim = 768       # 匹配模型的隐藏层维度




# model = MatchingModel(HAN(in_channels, hidden_channels, out_channels, transform_train_dataset.metadata()), hidden_dim)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# criterion = nn.BCELoss()


# for epoch in range(100):
#     for data in transform_train_dataset:
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, data.y.float())
#         loss.backward()
#         optimizer.step()

#     # Evaluate on test data
#     y_true = []
#     y_pred = []
#     for data in test_dataset:
#         with torch.no_grad():
#             output = model(data)
#             predictions = (output > 0.5).float()
#             y_true.extend(data.y.tolist())
#             y_pred.extend(predictions.tolist())

#     f1 = f1_score(y_true, y_pred)
# #     print(f"Epoch {epoch+1}: F1 Score - {f1}")





