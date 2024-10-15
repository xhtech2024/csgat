from torch_geometric.data import HeteroData,InMemoryDataset
from tqdm import tqdm
import os
from typing import List
import torch
from transformers import BertModel, BertTokenizer
class MyOwnDataset(InMemoryDataset):
    """
    
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset.(one of:obj:'Amazon-Google', 'Beer')
        split (str): The dataset split to return. Can be 'train', 'val', or 'test'.
    
    transform:
            metapaths = [
                [('entity', 'has', 'attri'), ('attri', 'is_contained_in', 'entity')]
            ]
            transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=True, drop_unconnected_node_types=True)

    """

    names = ['Amazon-Google', 'Beer']

    def __init__(self,
                 root:str='./data/',
                 name:str="Beer",
                 split: str ="train",
                 transform=None,
                 pre_transform=None,
                 force_reload=False,
                 device='cpu'):
        self.entity_nodes = {}
        self.attribute_nodes = {}
        self.token_nodes = {}
        self.edges = {"entity_attri": [], "attri_word": []}
        self.entity_embeddings = {}
        self.attribute_embeddings = {}
        self.token_embeddings = {}
        self.device = device  
        self.name = name
        self.root = root+name
        assert self.name in self.names
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        super().__init__(self.root, transform, pre_transform,force_reload=force_reload)

        if split == 'train':
            path = self.processed_paths[0]
        elif split == 'val':
            path = self.processed_paths[1]
        elif split == 'test':
            path = self.processed_paths[2]
        else:
            raise ValueError(f"Split '{split}' found, but expected either "
                             f"'train', 'val', or 'test'")
        self.load(path,data_cls=HeteroData)

    
    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt','valid.txt','test.txt']

    @property
    def processed_file_names(self):
        return ['train_data.pt', 'val_data.pt','test_data.pt']


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


    def process(self):
      
        for i, raw_path in enumerate(self.raw_paths): # 遍历3个文件
            data_list = []
            with open(raw_path, 'r') as file:
                for line in tqdm(file, desc=f'Processing {os.path.basename(raw_path)}'):
                    data = HeteroData()
                    parts = line.strip().split("\t")
                    entity1_str = parts[0]
                    entity2_str = parts[1]
                    label = int(parts[2])

                    entity1 = self.parse_entity(entity1_str, self.tokenizer)
                    entity2 = self.parse_entity(entity2_str, self.tokenizer)
                    
                    # 使用 BERT 嵌入
                    data['entity'].x = torch.stack([self.entity_embeddings[entity_id] for entity_id in self.entity_nodes]).to(self.device)
                    data['attri'].x = torch.stack([self.attribute_embeddings[attr_node_id] for attr_node_id in self.attribute_nodes]).to(self.device)
                    data['word'].x = torch.stack([self.token_embeddings[word] for word in self.token_nodes]).to(self.device)
                    
                    entity_attri_edge_index = torch.tensor(self.edges["entity_attri"], dtype=torch.long).t().contiguous().to(self.device)
                    attri_word_edge_index = torch.tensor(self.edges["attri_word"], dtype=torch.long).t().contiguous().to(self.device)
                    

                    # Add forward edges
                    data['entity', 'to', 'attri'].edge_index = entity_attri_edge_index
                    data['attri', 'to', 'word'].edge_index = attri_word_edge_index
            
                    data['attri', 'to', 'entity'].edge_index = entity_attri_edge_index.flip(0).contiguous()
                    data['word', 'to', 'attri'].edge_index = attri_word_edge_index.flip(0).contiguous()





                    parts = line.strip().split("\t")
                    entity1_str = parts[0]
                    entity2_str = parts[1]
                    label = int(parts[2])

                    entity1 = self.parse_entity(entity1_str, self.tokenizer)
                    entity2 = self.parse_entity(entity2_str, self.tokenizer)

            
                    data.y = torch.tensor([label])  # 添加label
                    data_list.append(data)  # 将一个异构图加入到数据集中
                self.save(data_list, path=self.processed_paths[i])