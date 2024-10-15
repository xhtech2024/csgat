import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx


def visualize_hetero_graph(data, output_file="graph.png"):
    G = to_networkx(data)

    # 将图转换为无向图
    G = G.to_undirected()

    # 为不同类型的节点分配颜色
    color_map = []
    for node in G:
        if G.nodes[node]['type'] == 'entity':
            color_map.append('red')
        elif G.nodes[node]['type'] == 'attri':
            color_map.append('blue')
        elif G.nodes[node]['type'] == 'word':
            color_map.append('green')
        else:
            color_map.append('gray')  # 默认颜色

    plt.figure(figsize=(8, 8))
    nx.draw(G, with_labels=True, node_color=color_map, node_size=700, edge_color='gray')

    # 保存图形为文件
    plt.savefig(output_file)
    print(f"图形已保存为 {output_file}")