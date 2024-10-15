import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from torch_geometric.data import HeteroData, Dataset, DataLoader

class HHG:
    def __init__(self):
        self.data = HeteroData()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def add_token_node(self, token):
        if 'token' not in self.data:
            self.data['token'].x = []
        self.data['token'].x.append(self.get_embedding(token))
        return len(self.data['token'].x) - 1

    def add_attribute_node(self, key, value_tokens):
        if 'attribute' not in self.data:
            self.data['attribute'].x = []
        token_indices = [self.add_token_node(token) for token in value_tokens]
        attr_embedding = self.aggregate_embeddings([self.data['token'].x[idx] for idx in token_indices])
        self.data['attribute'].x.append(attr_embedding)
        attr_index = len(self.data['attribute'].x) - 1
        for token_idx in token_indices:
            if 'contains' not in self.data['attribute', 'contains', 'token']:
                self.data['attribute', 'contains', 'token'].edge_index = []
            self.data['attribute', 'contains', 'token'].edge_index.append([attr_index, token_idx])
        return attr_index

    def add_entity_node(self, entity_id, attributes):
        if 'entity' not in self.data:
            self.data['entity'].x = []
        attr_indices = [self.add_attribute_node(key, value.split()) for key, value in attributes.items()]
        entity_embedding = self.aggregate_embeddings([self.data['attribute'].x[idx] for idx in attr_indices])
        self.data['entity'].x.append(entity_embedding)
        entity_index = len(self.data['entity'].x) - 1
        for attr_idx in attr_indices:
            if 'contains' not in self.data['entity', 'contains', 'attribute']:
                self.data['entity', 'contains', 'attribute'].edge_index = []
            self.data['entity', 'contains', 'attribute'].edge_index.append([entity_index, attr_idx])
        return entity_index

    def add_edge(self, node1, node2, relation_type):
        if relation_type not in self.data:
            self.data[relation_type].edge_index = []
        self.data[relation_type].edge_index.append([node1, node2])

    def build_from_dataset(self, line):  
        entity1, entity2, label = line.strip().split('\t')
        entity1_id, entity1_attrs = self.parse_entity(entity1)
        entity2_id, entity2_attrs = self.parse_entity(entity2)
        entity1_node = self.add_entity_node(entity1_id, entity1_attrs)
        entity2_node = self.add_entity_node(entity2_id, entity2_attrs)
        if label == '1':
            self.add_edge(entity1_node, entity2_node, 'matches')

    def parse_entity(self, entity_str):
        parts = entity_str.split('COL ')
        entity_id = parts[0].strip()
        attributes = {}
        for part in parts[1:]:
            key_val = part.split('VAL ')
            if len(key_val) == 2:
                key = key_val[0].strip()
                value = key_val[1].strip()
                attributes[key] = value
        return entity_id, attributes

    def get_embedding(self, text):
        # 使用BERT生成文本的嵌入
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding

    def aggregate_embeddings(self, embeddings):
        # 返回一个假设的聚合嵌入
        return torch.mean(torch.stack(embeddings), dim=0)
data_list = []
# 示例数据集
with open('data/Beer/train.txt', 'r') as file:
    dataset = file.readlines()
    
    for line in dataset:
        hhg = HHG()
        hhg.build_from_dataset(line)
        data_list.append(hhg.data)


class HeteroDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data_list = self.process()

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def process(self):
        data_list = []
        with open('data/Beer/train.txt', 'r') as file:
            dataset = file.readlines()
            for line in dataset:
                hhg = HHG()
                hhg.build_from_dataset(line)
                data = hhg.data
                hetero_data = HeteroData()
                # 假设data包含节点和边的信息
                hetero_data['node_type'].x = torch.tensor(data['node_features'], dtype=torch.float)
                hetero_data['node_type'].y = torch.tensor(data['node_labels'], dtype=torch.long)
                hetero_data['node_type', 'edge_type', 'node_type'].edge_index = torch.tensor(data['edge_index'], dtype=torch.long)
                data_list.append(hetero_data)
        return data_list

# 使用自定义数据集
dataset = HeteroDataset(root='data/Beer')
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 迭代数据加载器
for batch in loader:
    print(batch)