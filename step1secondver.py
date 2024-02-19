import matplotlib.pyplot as plt
import numpy as np
import random
import re

def ReadData(file_path : str) -> list:
    """read dot file and save as a list"""

    with open(file_path, 'r') as file:
        dot_content = file.read()
    lines = dot_content.split(';\n')
    lines[0] = '1'
    return lines

def AdjacencyMatrix(num_node : int, lines : list) -> np.array:
    """trans the dot list into a adjacenct matrix"""

    matrix = np.zeros((num_node,num_node))
    sum_edge = 0
    for line in lines:
        if '--' in line:
            sum_edge += 1
            edge = re.split('--|\[weight=|\]',line)
            node1  = int(edge[0])
            node2  = int(edge[1])
            weight = int(edge[2])
            matrix[node1-1, node2-1] = weight
            matrix[node2-1, node1-1] = weight
    print(sum_edge)
    return matrix

def  GetLocation(adjmatrix : np.array) -> np.array:
    """思路:分layer画图,按edge数确定node的高度, edge越多node位置越高"""
    #求每个node的edge数量
    nonzero_counts = np.count_nonzero(adj_matrix, axis = 1)
    #按照edge数量排序node
    edge_numidx_sorted = np.argsort(nonzero_counts)
    #根据edge数量生成node y坐标
    edge_counts = np.unique(nonzero_counts,return_counts = True) #[0]是edge数量唯一值数组，[1]是值与唯一值相同元素数量数组
    layer_inter = 10
    layer_y = [y * layer_inter for y in range(len(edge_counts[0]))] #按照edge唯一值数量生成y坐标
    node_y = np.repeat([i for i in layer_y], [j for j in edge_counts[1]]) #计算节点y坐标
    #根据有相同edge的node数目生成node x坐标 中心不固定
    node_x = []
    for i, j in zip(edge_counts[1],edge_counts[0]):
        idx = int(np.argwhere(edge_counts[0] == j))
        x_layer_center = (max(edge_counts[0]) - j) // 4 
        if i % 2 == 0:
            for k in range(- ( i // 2 ), i // 2):
                node_x.append(k + ( (-1) ** idx) * x_layer_center)
        else: 
            for k in range(- ( i // 2 ), i // 2 + 1):
                node_x.append(k + ( (-1) ** idx) * x_layer_center)
  
    # for i,j in zip(edge_counts[1],edge_counts[0]):
    #     x_highlayer = [0, 2, -2, 4, -4]
    #     inter_x_ra = (1-(-1)) * np.random.random() + (-1)
    #     # inter_x_ra *= -1
    #     if j>15:
    #         node_x.append(x_highlayer[len(edge_counts[0]) - int(np.argwhere(edge_counts[0] == j))-1])
    #     elif i%2 == 0:
    #         for j in range(- ( i // 2 ), i // 2):
    #             node_x.append(j + inter_x_ra)
    #     else: 
    #         for j in range(- ( i // 2 ), i // 2 + 1):
    #             node_x.append(j + inter_x_ra)

    #拼接坐标与索引，根据索引进行排序
    x_y_idx = np.vstack((np.array(node_x), node_y, edge_numidx_sorted))
    x_y_idx_sorted = x_y_idx[ :, np.argsort(x_y_idx[2])]
    return x_y_idx_sorted


def SimpleGraphViz(x_y_idx_sorted : np.array, adjmatrix : np.array) -> None:
    fig, ax = plt.subplots()
    edge_idx1, edge_idx2 = np.nonzero(adjmatrix)
    #画线
    for i,j in zip(edge_idx1, edge_idx2):
        if i < j:
            weight = adjmatrix[i,j]
            ax.plot([x_y_idx_sorted[0,i],x_y_idx_sorted[0,j]],[x_y_idx_sorted[1,i],x_y_idx_sorted[1,j]],
                    color = (0, 0 , 1 , 0.3 + weight * (0.7 / np.max(adj_matrix))), 
                    linewidth = 0.15 * weight, 
                    zorder = 1)
    #画点
    ax.scatter(x_y_idx_sorted[0], x_y_idx_sorted[1], color = 'red', s = 10, zorder = 2) 
    for i in range(77):
        ax.text(x_y_idx_sorted[0, i], x_y_idx_sorted[1, i], s = i, fontsize = 5)
    fig.show()
    a=0


if __name__ == "__main__":
    dot_file_path = r"data/LesMiserables.dot" 
    num_node = 77
    dot_lines = ReadData(dot_file_path)
    adj_matrix = AdjacencyMatrix(num_node, dot_lines)
    x_y_idx_sorted = GetLocation(adj_matrix)
    SimpleGraphViz(x_y_idx_sorted, adj_matrix)
    #0.422142 s/run
    a=0