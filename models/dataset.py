import random
import re

import numpy as np
import torch

from torch.utils import data
from torch_geometric.data import HeteroData
# from .argument import Augmenter

tokenizer = None


def get_lm_path(lm, lm_path):
    if lm_path != None:
        return lm_path

    if lm == 'bert':
        return 'bert-base-uncased'
    elif lm == 'distilbert':
        return 'distilbert-base-uncased'
    elif lm == 'roberta':
        return 'roberta-base'
    elif lm == 'xlnet':
        return 'xlnet-base-cased'


def get_tokenizer(lm, lm_path):
    global tokenizer

    path = get_lm_path(lm, lm_path)
    if tokenizer is None:
        if lm == 'bert':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained(path)
        elif lm == 'distilbert':
            from transformers import DistilBertTokenizer
            tokenizer = DistilBertTokenizer.from_pretrained(path)
        elif lm == 'roberta':
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained(path)
        elif lm == 'xlnet':
            from transformers import XLNetTokenizer
            tokenizer = XLNetTokenizer.from_pretrained(path)

    return tokenizer


class Dataset(data.Dataset):
    def __init__(self, source, category, lm='bert', lm_path=None, max_len=512, split=True, augment_op=None):
        self.tokenizer = get_tokenizer(lm, lm_path)

        # tokens and tags
        self.max_len = max_len
        self.augment_op = augment_op # Augmentation is not used in the currently.
        # if augment_op != None:
        #     self.augmenter = Augmenter()
        # else:
        self.augmenter = None

        sents, tags_li, attributes = self.read_classification_file(source, split)

        # assign class variables
        self.sents, self.tags_li, self.attributes = sents, tags_li, attributes
        self.category = category

        self.attr_num = len(self.attributes[0][0])

        # index for tags/labels
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.category)}
        self.idx2tag = {idx: tag for idx, tag in enumerate(self.category)}

    def read_classification_file(self, path, split):
        sents, labels, attributes = [], [], []
        for line in open(path):
            items = line.strip().split('\t')

            if self.augmenter != None:
                aug_items = [self.augmenter.augment_sent(item, self.augment_op)
                             for item in items[0:-1]]  # 获得两个实体
                items[0:-1] = aug_items

            attrs = []
            if split:
                attr_items = [item + ' COL' for item in items[0:-1]]
                for attr_item in attr_items:
                    attrs.append([f"COL {attr_str}" for attr_str
                                  in re.findall(r"(?<=COL ).*?(?= COL)", attr_item)])
                assert len(attrs[0]) == len(attrs[1])
            else:
                attrs = [[item] for item in items[0:-1]]

            sents.append(items[0] + ' [SEP] ' + items[1])
            labels.append(items[2])
            attributes.append(attrs)
        return sents, labels, attributes

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        """Embedding 不再是使用提前处理好的，而是用BERT进行成生成

        Args:
            idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        words, tags, attributes = self.sents[idx], self.tags_li[idx], self.attributes[idx]
        HHG  =  HeteroData()
        HHG['entity'].x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        HHG['attr'].x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        HHG['token'].x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        HHG['entity'].num_nodes = 2

        xs = [self.tokenizer.encode(text=attributes[0][i], text_pair=attributes[1][i],
                                    add_special_tokens=True, truncation="longest_first", max_length=self.max_len)
              for i in range(self.attr_num)]
        left_zs = [self.tokenizer.encode(text=attributes[0][i], add_special_tokens=True,
                                         truncation="longest_first", max_length=self.max_len)
                   for i in range(self.attr_num)]
        right_zs = [self.tokenizer.encode(text=attributes[1][i], add_special_tokens=True,
                                          truncation="longest_first", max_length=self.max_len)
                    for i in range(self.attr_num)]
        



        masks = [torch.zeros(self.tokenizer.vocab_size, dtype=torch.int)
                 for _ in range(self.attr_num)]
        for i in range(self.attr_num):
            masks[i][xs[i]] = 1
        masks = torch.stack(masks)

        y = self.tag2idx[tags]  # label

        seqlens = [len(x) for x in xs]
        left_zslens = [len(left_z) for left_z in left_zs]
        right_zslens = [len(right_z) for right_z in right_zs]

        return words, xs, y, seqlens, masks, left_zs, right_zs, left_zslens, right_zslens, attributes

    def get_attr_num(self):
        return self.attr_num

    @staticmethod
    def pad(batch):
        f = lambda x: [sample[x] for sample in batch]
        g = lambda x, seqlen, val: \
            [[sample + [val] * (seqlen - len(sample)) \
              for sample in samples[x]]
             for samples in batch]  # 0: <pad>

        # get maximal sequence length
        seqlens = f(3)
        maxlen = np.array(seqlens).max()

        words = f(0)
        xs = torch.LongTensor(g(1, maxlen, 0))
        y = f(2)
        masks = torch.stack(f(4))

        if isinstance(y[0], float):
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)
        return words, xs, y, seqlens, masks

    @staticmethod
    def padJoin(batch):
        f = lambda x: [sample[x] for sample in batch]
        g = lambda x, seqlen, val: \
            [[sample + [val] * (seqlen - len(sample)) \
              for sample in samples[x]]
             for samples in batch]  # 0: <pad>

        # get maximal sequence length
        seqlens = f(3)
        maxlen = np.array(seqlens).max()

        words = f(0)
        xs = torch.LongTensor(g(1, maxlen, 0))
        y = f(2)
        masks = torch.stack(f(4))

        attributes = f(9)
        attr_num = xs.size()[1]

        right_attributes = []
        for i in range(attr_num):
            right_attribute = []
            for attribute in attributes:
                right_attribute.append(attribute[1][i])
            right_attributes.append(right_attribute)

        zs = [tokenizer.encode(text=' '.join(right_attributes[i]),
                    add_special_tokens=False, truncation="longest_first", max_length=512)
              for i in range(attr_num)]
        maxlen = np.array([len(z) for z in zs]).max()
        zs = [z + [0] * (maxlen-len(z)) for z in zs]
        zs = torch.LongTensor(zs).unsqueeze(0).permute(1, 0, 2)


        if isinstance(y[0], float):
            y = torch.Tensor(y)
        else:
            y = torch.LongTensor(y)

        return words, xs, zs, y, seqlens, masks



import os.path as osp

import torch
from torch_geometric.data import InMemoryDataset, download_url,HeteroData

import torch
from torch_geometric.data import HeteroData

class HHGDataset:
    def __init__(self, sents, tags_li, attributes, tokenizer, max_len, attr_num, tag2idx):
        self.sents = sents
        self.tags_li = tags_li
        self.attributes = attributes
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.attr_num = attr_num
        self.tag2idx = tag2idx

    def build_hhg(self, idx):
        words, tags, attributes = self.sents[idx], self.tags_li[idx], self.attributes[idx]
        HHG = HeteroData()

        # 初始化节点特征
        HHG['entity'].x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        HHG['attr'].x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        HHG['token'].x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
        HHG['entity'].num_nodes = 2

        # 编码属性
        xs = [self.tokenizer.encode(text=attributes[0][i], text_pair=attributes[1][i],
                                    add_special_tokens=True, truncation="longest_first", max_length=self.max_len)
              for i in range(self.attr_num)]
        left_zs = [self.tokenizer.encode(text=attributes[0][i], add_special_tokens=True,
                                         truncation="longest_first", max_length=self.max_len)
                   for i in range(self.attr_num)]
        right_zs = [self.tokenizer.encode(text=attributes[1][i], add_special_tokens=True,
                                          truncation="longest_first", max_length=self.max_len)
                    for i in range(self.attr_num)]

        # 构建 Token-Attribute 边
        for i in range(self.attr_num):
            for token in xs[i]:
                HHG['token', 'to', 'attr'].edge_index = torch.tensor([[token], [i]], dtype=torch.long)

        # 构建 Attribute-Entity 边
        for i in range(self.attr_num):
            HHG['attr', 'to', 'entity'].edge_index = torch.tensor([[i], [0]], dtype=torch.long)

        # 构建 Entity-Entity 边
        HHG['entity', 'to', 'entity'].edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # 标签
        y = self.tag2idx[tags]  # label

        return HHG, y

# 示例使用
# sents = [["example", "sentence"]]
# tags_li = [["tag1", "tag2"]]
# attributes = [[["key1", "key2"], ["val1", "val2"]]]
# tokenizer = ...  # 初始化你的tokenizer
# max_len = 128
# attr_num = 2
# tag2idx = {"tag1": 0, "tag2": 1}

# dataset = HHGDataset(sents, tags_li, attributes, tokenizer, max_len, attr_num, tag2idx)
# HHG, y = dataset.build_hhg(0)





class BeerDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return 'data_1.pt'

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        import scipy.sparse as sp
        data = HeteroData()  # 
        node_types = ['entity', 'attr', 'token']
        for i, node_type in enumerate(node_types[:2]):
            x = sp.load_npz(osp.join(self.raw_dir, f'features_{i}.npz'))
            data[node_type].x = torch.from_numpy(x.todense()).to(torch.float)
        x = np.load(osp.join(self.raw_dir, 'features_2.npy')) 
        data['entity'].x = torch.from_numpy(x).to(torch.float)  # 节点初始特征
        node_type_idx = np.load(osp.join(self.raw_dir, 'node_types.npy'))
        node_type_idx = torch.from_numpy(node_type_idx).to(torch.long)
        data['attr'].num_nodes = int((node_type_idx == 3).sum())  # 属性初始特征

        y = np.load(osp.join(self.raw_dir, 'labels.npy'))
        data['token'].y = torch.from_numpy(y).to(torch.long)  # token初始特征

        split = np.load(osp.join(self.raw_dir, 'train_val_test_idx.npz'))

        A = sp.load_npz(osp.join(self.raw_dir, 'adjM.npz'))  # 邻接矩阵
        for src, dst in product(node_types, node_types):
            A_sub = A[s[src][0]:s[src][1], s[dst][0]:s[dst][1]].tocoo()
            if A_sub.nnz > 0:
                row = torch.from_numpy(A_sub.row).to(torch.long)
                col = torch.from_numpy(A_sub.col).to(torch.long)
                data[src, dst].edge_index = torch.stack([row, col], dim=0)





        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.

        
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data