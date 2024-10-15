import os
import torch
from torch_geometric.data import HeteroData
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 将模型设置为评估模式
model.eval()

def generate_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

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
        data = HeteroData()
        
        # 使用 BERT 嵌入
        data['entity'].x = torch.stack([self.entity_embeddings[entity_id] for entity_id in self.entity_nodes])
        data['attri'].x = torch.stack([self.attribute_embeddings[attr_node_id] for attr_node_id in self.attribute_nodes])
        data['word'].x = torch.stack([self.token_embeddings[word] for word in self.token_nodes])
        
        entity_attri_edge_index = torch.tensor(self.edges["entity_attri"], dtype=torch.long).t().contiguous()
        attri_word_edge_index = torch.tensor(self.edges["attri_word"], dtype=torch.long).t().contiguous()
        
        # Add forward edges
        data['entity', 'has', 'attri'].edge_index = entity_attri_edge_index
        data['attri', 'contains', 'word'].edge_index = attri_word_edge_index

        data['attri', 'is_contained_in', 'entity'].edge_index = entity_attri_edge_index.flip(0)
        data['word', 'contains_attri', 'attri'].edge_index = attri_word_edge_index.flip(0)
        
        return data

    def save(self, filepath):
        data = self.build_graph()
        torch.save(data, filepath)

    def load(self, filepath):
        data = torch.load(filepath)
        self.entity_nodes = {i: i for i in range(data['entity'].x.size(0))}
        self.attribute_nodes = {i: i for i in range(data['attri'].x.size(0))}
        self.token_nodes = {i: i for i in range(data['word'].x.size(0))}
        self.edges = {
            "entity_attri": data['entity', 'has', 'attri'].edge_index.t().tolist(),
            "attri_word": data['attri', 'contains', 'word'].edge_index.t().tolist()
        }
        self.entity_embeddings = {i: data['entity'].x[i] for i in range(data['entity'].x.size(0))}
        self.attribute_embeddings = {i: data['attri'].x[i] for i in range(data['attri'].x.size(0))}
        self.token_embeddings = {i: data['word'].x[i] for i in range(data['word'].x.size(0))}

    def generate_datasets(self, entities, train_ratio=0.8):
        num_entities = len(entities)
        train_size = int(num_entities * train_ratio)
        train_entities = entities[:train_size]
        test_entities = entities[train_size:]

        print("Generating training dataset...")
        for i, entity in enumerate(tqdm(train_entities, desc="Training")):
            self.add_entity(entity, f"train_entity_{i}")

        print("Generating testing dataset...")
        for i, entity in enumerate(tqdm(test_entities, desc="Testing")):
            self.add_entity(entity, f"test_entity_{i}")

# 示例用法
hhg = HHG()
data_filepath = 'hhg_data.pth'

if os.path.exists(data_filepath):
    hhg.load(data_filepath)
else:
    # 假设我们有一些实体数据
    entities = [
        {"name": "Entity1", "description": "This is the first entity."},
        {"name": "Entity2", "description": "This is the second entity."}
    ]
    hhg.generate_datasets(entities)
    hhg.save(data_filepath)

data = hhg.build_graph()