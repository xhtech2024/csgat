import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import AttentionLayer as AL, GlobalAttentionLayer as GoAL, StructAttentionLayer as SAL
from .dataset import get_lm_path
from torch_geometric.nn import HANConv

class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HAN, self).__init__()
        # H, D = self.heads, self.out_channels // self.heads
        self.conv1 = HANConv(in_channels, hidden_channels, graph.metadata(), heads=8)
        self.conv2 = HANConv(hidden_channels, out_channels, graph.metadata(), heads=4)

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        x = x['author']
        return x

class TranHGAT(nn.Module):
    def __init__(self, attr_num, device='cpu', finetuning=True, lm='bert', lm_path=None):
        super().__init__()

        # load the model or model checkpoint
        path = get_lm_path(lm, lm_path)
        self.lm = lm
        if lm == 'bert':
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaModel
            self.bert = RobertaModel.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetModel
            self.bert = XLNetModel.from_pretrained(path)

        self.device = device
        self.finetuning = finetuning

        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        self.inits = nn.ModuleList([
            GoAL(hidden_size, 0.2)
            for _ in range(attr_num)])  # 每个属性都有一个初始化层， 好在属性的数量不太多
        self.conts = nn.ModuleList([
            AL(hidden_size + hidden_size, 0.2, device)
            for _ in range(attr_num)])
        self.out = SAL(hidden_size * (attr_num + 1), 0.2)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, xs, y, masks): # 分析
        xs = xs.to(self.device)
        y = y.to(self.device)
        masks = masks.to(self.device)
 
        xs = xs.permute(1, 0, 2) #[Attributes, Batch, Tokens] 他是处理好的Bert可以接受的输入

        masks = masks.permute(0, 2, 1) # [Batch, All Tokens, Attributes]

        attr_outputs = []
        pooled_outputs = []
        attns = []
        if self.training and self.finetuning:
            self.bert.train()
            for x, init, cont in zip(xs, self.inits, self.conts):  # 模型计算 四个属性 每个x是一行数据集的一对要比较的属性。
                attr_embeddings = init(self.bert.get_input_embeddings()(x)) # [Batch, Hidden] 获得一行内容的bert嵌入, bert是
                # attr_embeddings  
                # 预训练好的，后续随着训练会进行微调 
                attr_outputs.append(attr_embeddings)  # 根据token属性，计算

                attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings) # [Batch, All Tokens]
                attns.append(attn)

            attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks # [Batch, All Tokens, Attributes]
            attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2) # [Batch, Attributes, Hidden]

            for x in xs:
                if self.lm == 'distilbert':
                    words_emb = self.bert.embeddings(x)
                else:
                    words_emb = self.bert.get_input_embeddings()(x)  # 获取bert的嵌入层，然后将x交给嵌入层进行嵌入处理

                for i in range(words_emb.size()[0]): # i is index of batch
                    words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])  # 生成WpC  

                    # 这步的矩阵乘法是为了将属性嵌入和词嵌入进行融合，融合的方式是通过属性的权重来融合的


                # 以下对应论文第5节，分层聚合的内容
                output = self.bert(inputs_embeds=words_emb)  # 再次使用bert
                pooled_output = output[0][:, 0, :] # 对应属性摘要层 获取CLS 表示
                pooled_output = self.dropout(pooled_output)
                pooled_outputs.append(pooled_output)  # 三个 1x768

            attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
            entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1) # attr拼接形成entity emb  四个属性
            entity_output = self.out(attr_outputs, entity_outputs)  #
            pass
        else:
            self.bert.eval()
            with torch.no_grad():
                for x, init, cont in zip(xs, self.inits, self.conts):
                    attr_embeddings = init(self.bert.get_input_embeddings()(x))
                    attr_outputs.append(attr_embeddings)

                    # 64 * 768
                    attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings)
                    attns.append(attn)

                attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks
                attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)
                for x in xs:
                    if self.lm == 'distilbert':
                        words_emb = self.bert.embeddings(x)
                    else:
                        words_emb = self.bert.get_input_embeddings()(x)

                    for i in range(words_emb.size()[0]):
                        words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                    output = self.bert(inputs_embeds=words_emb)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    pooled_outputs.append(pooled_output)

                attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
                entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
                entity_output = self.out(attr_outputs, entity_outputs)

        logits = self.fc(entity_output)  # 接入一个全连接层输出分类
        y_hat = logits.argmax(-1)  # 预测结果
        return logits, y, y_hat



class HANGAT(nn.Module):
    def __init__(self, attr_num, device='cpu', finetuning=True, lm='bert', lm_path=None):
        super().__init__()

        # load the model or model checkpoint
        path = get_lm_path(lm, lm_path)
        self.lm = lm
        if lm == 'bert':
            from transformers import BertModel
            self.bert = BertModel.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertModel
            self.bert = DistilBertModel.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaModel
            self.bert = RobertaModel.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetModel
            self.bert = XLNetModel.from_pretrained(path)

        self.device = device
        self.finetuning = finetuning
        self.HAN = HAN()  # 定义一个HAN模型
        # hard corded for now
        hidden_size = 768
        hidden_dropout_prob = 0.1

        self.inits = nn.ModuleList([
            GoAL(hidden_size, 0.2)
            for _ in range(attr_num)])  # 每个属性都有一个初始化层， 好在属性的数量不太多
        self.conts = nn.ModuleList([
            AL(hidden_size + hidden_size, 0.2, device)
            for _ in range(attr_num)])
        self.out = SAL(hidden_size * (attr_num + 1), 0.2)

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.fc = nn.Linear(hidden_size, 2)

    def _generate_graph(xs):
        graph = None  #[Attributes, Batch, Tokens]



        pass





        return graph

    def forward(self, xs, y, masks): # 分析
        xs = xs.to(self.device)
        y = y.to(self.device)
        masks = masks.to(self.device)
        graph  = self._generate_graph(xs)
        xs = xs.permute(1, 0, 2) #[Attributes, Batch, Tokens] 他是处理好的Bert可以接受的输入

        masks = masks.permute(0, 2, 1) # [Batch, All Tokens, Attributes]
        
    
        output = self.HAN(xs)  # 通过HAN模型进行处理
        attr_outputs = []
        pooled_outputs = []
        attns = []
        if self.training and self.finetuning:
            self.bert.train()
            for x, init, cont in zip(xs, self.inits, self.conts):  # 模型计算
                attr_embeddings = init(self.bert.get_input_embeddings()(x)) # [Batch, Hidden] 获得一行内容的bert嵌入, bert是
                # 预训练好的，后续随着训练会进行微调 
                attr_outputs.append(attr_embeddings)  # 根据token属性，计算

                attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings) # [Batch, All Tokens]
                attns.append(attn)

            attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks # [Batch, All Tokens, Attributes]
            attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2) # [Batch, Attributes, Hidden]

            for x in xs:
                if self.lm == 'distilbert':
                    words_emb = self.bert.embeddings(x)
                else:
                    words_emb = self.bert.get_input_embeddings()(x)  # 获取bert的嵌入层，然后将x交给嵌入层进行嵌入处理

                for i in range(words_emb.size()[0]): # i is index of batch
                    words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])  # 生成WpC  

                    # 这步的矩阵乘法是为了将属性嵌入和词嵌入进行融合，融合的方式是通过属性的权重来融合的


                # 以下对应论文第5节，分层聚合的内容
                output = self.bert(inputs_embeds=words_emb)  # 再次使用bert
                pooled_output = output[0][:, 0, :] # 对应属性摘要层 获取CLS 表示
                pooled_output = self.dropout(pooled_output)
                pooled_outputs.append(pooled_output)  # 三个 1x768

            attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
            entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1) # attr拼接形成entity emb
            entity_output = self.out(attr_outputs, entity_outputs)  #
        else:
            self.bert.eval()
            with torch.no_grad():
                for x, init, cont in zip(xs, self.inits, self.conts):
                    attr_embeddings = init(self.bert.get_input_embeddings()(x))
                    attr_outputs.append(attr_embeddings)

                    # 64 * 768
                    attn = cont(x, self.bert.get_input_embeddings(), attr_embeddings)
                    attns.append(attn)

                attns = self.softmax(torch.stack(attns).permute(1, 2, 0)) * masks
                attr_outputs = torch.stack(attr_outputs).permute(1, 0, 2)
                for x in xs:
                    if self.lm == 'distilbert':
                        words_emb = self.bert.embeddings(x)
                    else:
                        words_emb = self.bert.get_input_embeddings()(x)

                    for i in range(words_emb.size()[0]):
                        words_emb[i] += torch.matmul(attns[i][x[i]], attr_outputs[i])

                    output = self.bert(inputs_embeds=words_emb)
                    pooled_output = output[0][:, 0, :]
                    pooled_output = self.dropout(pooled_output)
                    pooled_outputs.append(pooled_output)

                attr_outputs = torch.stack(pooled_outputs).permute(1, 0, 2)
                entity_outputs = attr_outputs.reshape(attr_outputs.size()[0], -1)
                entity_output = self.out(attr_outputs, entity_outputs)

        logits = self.fc(entity_output)  # 接入一个全连接层输出分类
        y_hat = logits.argmax(-1)  # 预测结果
        return logits, y, y_hat