import matplotlib.pyplot as plt
import networkx as nx

def draw_computingGraph(nodes_op, nodes_tensor, edges):
    # 创建一个图形对象
    G = nx.DiGraph()

    # 添加节点
    G.add_nodes_from(nodes_op)
    G.add_nodes_from(nodes_tensor)
    
    # 添加边
    G.add_edges_from(edges)

    # 使用不同的节点布局算法
    # pos = nx.circular_layout(G)  # 环形布局
    # pos = nx.random_layout(G)  # 随机布局
    # pos = nx.spring_layout(G)  # 弹簧布局
    # pos = nx.shell_layout(G)  # 壳布局
    pos = nx.spectral_layout(G)  # 谱布局
    
    # 绘制图形
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes_op, node_color='lightgreen', node_shape='s')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=nodes_tensor, node_color='lightblue', node_shape='o')
    nx.draw_networkx_edges(G, pos=pos, edge_color='gray')  # 绘制边
    
    # 绘制节点标签
    labels = {node: node for node in G.nodes}  # 使用节点名称作为标签
    nx.draw_networkx_labels(G, pos=pos, labels=labels)
    
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('computingGraph')