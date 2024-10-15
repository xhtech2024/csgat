import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置中文字体
zh_font = font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

datasets = ['Amazon-Google', 'BeerAdvo-RateBeer',"Walmart-Amazon"]

# 使用A模块的F1分数
f1_with_A = [0.87, 0.80,0.901]

# 不使用A模块的F1分数
f1_without_A = [0.80, 0.775,0.75]

# 设置柱状图的宽度
bar_width = 0.25

# 设置柱状图的位置
index = np.arange(len(datasets))

# 创建图形和子图，调整图形大小
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制柱状图
bars1 = ax.bar(index, f1_with_A, bar_width, label='分层注意力')
bars2 = ax.bar(index + bar_width, f1_without_A, bar_width, label='无分层注意力')

# 添加标签、标题和图例
ax.set_xlabel('数据集', fontproperties=zh_font)
ax.set_ylabel('F1分数', fontproperties=zh_font)
ax.set_title('使用和不使用分层注意力在不同数据集上的F1分数对比', fontproperties=zh_font)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(datasets, fontproperties=zh_font)
ax.legend(prop=zh_font)

# 保存图形到本地文件
plt.savefig('f1_score_comparison.png')

# 显示图形
plt.show()