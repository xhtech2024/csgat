import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    注意力层（Attention Layer）模块用于计算注意力权重。

    Args:
        hidden_size (int): 隐藏层的大小。
        alpha (float): LeakyReLU激活函数的负斜率。
        device (torch.device): 模型所在的设备。

    Attributes:
        a (torch.Tensor): 注意力权重参数。
        leakyrelu (nn.LeakyReLU): LeakyReLU激活函数。
        device (torch.device): 模型所在的设备。

    """

    def __init__(self, hidden_size, alpha, device):
        super(AttentionLayer, self).__init__()

        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

        self.device = device

    def forward(self, words, word_emb, attr_emb):
        """
        前向传播函数，计算注意力权重。

        Args:
            words (torch.Tensor): 输入的单词序列。
            word_emb (nn.Embedding): 单词嵌入层。
            attr_emb (torch.Tensor): 属性嵌入。

        Returns:
            torch.Tensor: 注意力权重。

        """
        words_emb = word_emb(words)
        attr_emb = attr_emb.unsqueeze(1)
        attrs_emb = attr_emb.repeat(1, words_emb.size()[1], 1)
        combina = torch.cat([words_emb, attrs_emb], dim=2)

        e = self.leakyrelu(torch.matmul(combina, self.a)).squeeze(-1)  # (batch size, seq length)
        attn = torch.zeros(words_emb.size()[0], word_emb.num_embeddings)  # (batch size, vocab length)
        attn = attn.to(self.device)
        for i in range(words_emb.size()[0]):
            attn[i][words[i]] = e[i]

        return attn


class ContAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha):
        super(ContAttentionLayer, self).__init__()

        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, attrs, all):
        alls = all.repeat(attrs.size()[0], 1)
        combina = torch.cat([attrs, alls], dim=1)

        e = self.leakyrelu(torch.matmul(combina, self.a))
        attention = F.softmax(e, dim=0)

        return attrs - attention * alls



class GlobalAttentionLayer(nn.Module):
    """
    全局注意力层，计算注意力权重并将其应用于输入的嵌入向量。

    参数:
        hidden_size (int): 隐藏层的大小。
        alpha (float): LeakyReLU激活函数的负斜率。

    属性:
        linear (nn.Linear): 线性变换层。
        a (nn.Parameter): 注意力权重的可学习参数。
        leakyrelu (nn.LeakyReLU): LeakyReLU激活函数。

    """

    def __init__(self, hidden_size, alpha):
        super(GlobalAttentionLayer, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, words_emb):
        """计算全局注意力层的前向传播。

        参数:
            words_emb (torch.Tensor): 输入的词嵌入向量。

        返回:
            torch.Tensor: 应用注意力后的输出属性嵌入向量。

        """
        words_emb = self.linear(words_emb)

        e = self.leakyrelu(torch.matmul(words_emb, self.a)).squeeze(-1)
        attention = F.softmax(e, dim=1).unsqueeze(1)

        attributes_emb = torch.matmul(attention, words_emb).squeeze(1)
        return F.relu(attributes_emb)



class StructAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha):
        super(StructAttentionLayer, self).__init__()

        self.a = nn.Parameter(torch.zeros(size=(hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, attrs_emb, entity_emb):
        attr_num = attrs_emb.size()[1]

        entity_emb = entity_emb.unsqueeze(1)
        entitys_emb = entity_emb.repeat(1, attr_num, 1)
        combina = torch.cat([attrs_emb, entitys_emb], dim=2)

        e = self.leakyrelu(torch.matmul(combina, self.a)).squeeze(-1)
        attention = F.softmax(e, dim=1).unsqueeze(1) * attr_num

        entitys_emb = torch.matmul(attention, attrs_emb).squeeze(1)  # 计算实体表示
        return entitys_emb


class ResAttentionLayer(nn.Module):

    def __init__(self, hidden_size, alpha, thr=0.5):
        super(ResAttentionLayer, self).__init__()

        self.thr = thr

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.zeros(size=(2 * hidden_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, entity_embs):
        Wh = self.linear(entity_embs)

        a_input = self._prepare_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)
        attention = F.softmax(e, dim=1)

        # We apply the pooling operation
        attention = (attention < self.thr).type(attention.dtype) * attention
        h_prime = torch.matmul(attention, Wh)

        return F.elu(entity_embs - h_prime)

    def _prepare_input(self, Wh):
        N = Wh.size()[0]
        d = Wh.size()[1]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2 * d)
