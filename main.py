
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer

import torch_geometric.transforms as T

from sklearn.metrics import f1_score,accuracy_score
from utils import visualize_hetero_graph
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = BertModel.from_pretrained('bert-base-uncased').to(device)

from model import MatchingModel, HAN,process_data

data_list = []


dataset = "Walmart-Amazon"

train_datalist = process_data(f'data/{dataset}/raw/train.txt', tokenizer,dataset)
test_datalist = process_data(f'data/{dataset}/raw/test.txt', tokenizer,dataset)

# 实验2

# metapaths = [
#                 [('attri', 'to', 'word'), ('word', 'to', 'attri')],
#                 [('word', 'to', 'attri'), ('attri', 'to', 'word')],
#                 [('entity', 'to', 'attri'), ('attri', 'to', 'word'), ('word', 'to', 'attri'),('attri', 'to', 'entity')],
#             ]


# 实验1
# metapaths = [
#                  [('attri', 'to', 'word'), ('word', 'to', 'attri')],
#             ]

# 实验3
# metapaths = [
#                 [('entity', 'to', 'attri'), ('attri', 'to', 'word'), ('word', 'to', 'attri'),('attri', 'to', 'entity')]
#             ]

# 实验4 ： 无meta-path
metapaths = [
               
            ]

transform = T.AddMetaPaths(metapaths=metapaths, drop_orig_edge_types=False, drop_unconnected_node_types=False)
transformed_data = transform(train_datalist[0])


visualize_hetero_graph(transformed_data)


# 假设我们有一个数据加载器 data_loader
entity_feature_dim = 768  # 假设实体特征维度为128
attri_feature_dim = 768    # 假设属性特征维度为64
word_feature_dim = 768     # 假设词特征维度为32


# 选择最大的特征维度作为输入特征维度
in_channels = max(entity_feature_dim, attri_feature_dim, word_feature_dim)
hidden_channels = 384  # 隐藏层特征维度
out_channels = 128    # 输出特征维度
hidden_dim = 128       # 匹配模型的隐藏层维度


from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import  DataLoader
from sklearn.metrics import confusion_matrix,recall_score
import time

model = MatchingModel(HAN(in_channels, hidden_channels, out_channels,transformed_data), hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion =  nn.CrossEntropyLoss()


batch_size = 16
dataloader = DataLoader(train_datalist, batch_size=batch_size, shuffle=True)


writer = SummaryWriter(log_dir=f'runs/{dataset}_experiment')

best_f1 = 0.0
best_epoch = 0

for epoch in range(60):
    start_time = time.time()
    model.train()
    train_total_loss = 0
    for data in dataloader:
        data = transform(data)
        optimizer.zero_grad()
        logits, y_hat = model(data)
        y = data.y
        y = y.to(device)
        logits = logits.view(-1, logits.shape[-1])
        y = y.view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
    
    train_avg_loss = train_total_loss / len(dataloader)
    
    # 记录训练损失
    writer.add_scalar('Loss/train', train_avg_loss, epoch)
    
    # 评估逻辑
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for data in test_datalist:
            data = transform(data)
            logits, y_hat = model(data)
            y = data.y
            y = y.to(device)
            logits = logits.view(-1, logits.shape[-1])
            y = y.view(-1)
            preds = y_hat.flatten()
            all_preds.extend(preds.cpu().numpy())


            all_labels.extend(y.cpu().numpy())
            loss = criterion(logits, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_datalist)
    
    # 记录验证损失
    writer.add_scalar('Loss/val', avg_loss, epoch)
    # 计算准确率、召回率和F1分数
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # 计算混淆矩阵
    confusion_mat = confusion_matrix(all_labels, all_preds)
    tp = confusion_mat[1][1]
    fp = confusion_mat[0][1]
    tn = confusion_mat[0][0]
    fn = confusion_mat[1][0]
    print("True Positives (TP):", tp)
    print("False Positives (FP):", fp)
    print("True Negatives (TN):", tn)
    print("False Negatives (FN):", fn)

    
    # 记录准确率和F1分数
    writer.add_scalar('F1_Score/val', f1, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)
    writer.add_scalar('Recall/val', recall, epoch)
    
    
    
    print(f'Epoch: {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Recall:{recall:.4f},F1 Score: {f1:.4f}')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time elapsed for epoch {epoch+1}: {elapsed_time:.2f} seconds")
    # 更新最高F1分数和对应的epoch
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        # 保存模型
        torch.save(model.state_dict(), f'./checkpoints/{dataset}/best_model.pt')

print(f'Best F1 Score: {best_f1:.4f} at Epoch: {best_epoch+1}')

# 关闭 TensorBoard 记录器
writer.close()