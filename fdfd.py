import torch

# 创建一个张量
data = torch.tensor([[ 0,  0,  0,  0,  0,  0,  1,  1,  2,  2,  2,  2,  2,  3,  3,  3,  3,  4,
                      4,  4,  4,  4,  5,  5,  5,  6,  6,  6,  7,  7,  7,  7],
                     [ 0,  1,  2,  3,  2,  4,  5,  6,  7,  8,  9,  2,  4, 10, 11, 12, 13, 14,
                     15,  3,  2,  4, 14, 16, 17,  7, 18,  4, 19, 11, 20, 13]],
                    device='cuda:0')

# 检查张量是否连续
print("Is data contiguous?", data.is_contiguous())

# 如果张量不是连续的，则使其连续
if not data.is_contiguous():
    data = data.contiguous()

print("Is data contiguous after calling contiguous()?", data.is_contiguous())