import matplotlib.pyplot as plt
import numpy as np
import random
import re
import math
import time

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

    return matrix
import math

def GetSubNode(x0, y0, r, n):
    """围绕父节点画圆生成子节点"""
    points = []
    theta_step = 2 * math.pi / n
    for i in range(n):
        theta = i * theta_step
        xi = x0 + r * math.sin(theta)
        yi = y0 + r * math.cos(theta)
        points.append((xi, yi))
    return points

def GetSubNodeSub(x0, y0, r, n):
    """围绕子节点画圆的一部分生成下级子节点"""
    points = []
    theta_step = 2 * math.pi / 40
    for i in range(n):
        theta = i * theta_step
        if x0 == 0:
            xi = x0 + r * math.sin(theta)
            yi = y0 + r * math.cos(theta) * (y0 / (abs(y0)))
        elif y0 == 0:
            xi = x0 + r * math.cos(theta) * (x0 / (abs(x0)))
            yi = y0 + r * math.sin(theta)
        else:    
            xi = x0 + r * math.sin(theta) * (x0 / (abs(x0)))
            yi = y0 + r * math.cos(theta) * (y0 / (abs(y0)))
        points.append((xi, yi))
    return points

def  GetLocationParentParent(adjmatrix : np.array) -> np.array:
    """先画最多edge的node, 逐步扩展"""
    x_y_idx = np.zeros((3 , len(adjmatrix)))
    #求每个node的edge数量
    nonzero_counts = np.count_nonzero(adj_matrix, axis = 1)
    #按照edge数量排序node,ATTENTION手动改成了从大到小
    edge_num_sorted     = np.sort(nonzero_counts)[::-1]
    edge_num_idx_sorted = np.argsort(nonzero_counts)[::-1]
    num_parent = 0
    parent_interval = 10
    previous_parent_x = 0
    previous_parent_y = 0
    r_parent = 4
    sign1 = [1,-1,1,-1]
    sign2 = [1,1,-1,-1]
    for i, j in zip(edge_num_idx_sorted, edge_num_sorted):
        if x_y_idx[2, i] == 0:
            # x_y_idx[0 , i] = previous_parent_x + (-1) ** num_parent * num_parent * parent_interval
            # x_y_idx[1 , i] = previous_parent_y + (-1) ** num_parent * num_parent * parent_interval
            x_y_idx[0 , i] = previous_parent_x + (num_parent %2) * ((-1) ** int(num_parent / 2)) * (num_parent + 1) / 2 * parent_interval
            
            x_y_idx[1 , i] = x_y_idx[0 , i] * sign2[num_parent%4]

            x_y_idx[2 , i] = i

            previous_parent_x = x_y_idx[0 , i]
            previous_parent_y = x_y_idx[1 , i]
            # print(num_parent,previous_parent_x,previous_parent_y)
            num_parent += 1
            num_edge = j
            idx_subnodes = np.nonzero(adjmatrix[i])[0]
            subnodes = GetSubNode(x_y_idx[0 , i], x_y_idx[1 , i], r , num_edge)
            for k, idx_subnode in enumerate(idx_subnodes):
                if x_y_idx[2, idx_subnode] == 0:
                    x_y_idx[0, idx_subnode] = subnodes[k][0]
                    x_y_idx[1, idx_subnode] = subnodes[k][1]
                    x_y_idx[2, idx_subnode] = idx_subnode
    a=0
    return x_y_idx

def  GetLocationParentSub(adjmatrix : np.array) -> np.array:
    """先画最多edge的node, 逐步扩展"""
    x_y_idx = np.zeros((3 , len(adjmatrix)))
    #求每个node的edge数量
    nonzero_counts = np.count_nonzero(adj_matrix, axis = 1)
    #按照edge数量排序node,ATTENTION手动改成了从大到小
    edge_num_sorted     = np.sort(nonzero_counts)[::-1]
    edge_num_idx_sorted = np.argsort(nonzero_counts)[::-1]
    num_parent = 0
    parent_interval = 9
    previous_parent_x = 0
    previous_parent_y = 0
    r = 4
    sign1 = [1,1,-1,-1]
    for i, j in zip(edge_num_idx_sorted, edge_num_sorted):
        num_edge = j
        idx_subnodes = np.nonzero(adjmatrix[i])[0]
        if x_y_idx[2, i] == 0 and (i != 0):
            # x_y_idx[0 , i] = previous_parent_x + (-1) ** num_parent * num_parent * parent_interval
            # x_y_idx[1 , i] = previous_parent_y + (-1) ** num_parent * num_parent * parent_interval
            x_y_idx[0 , i] = previous_parent_x + (num_parent % 2) * ((-1) ** int(num_parent / 2)) * (num_parent + 1) / 2 * parent_interval
            x_y_idx[1 , i] = x_y_idx[0 , i] * sign1[num_parent % 4]
            x_y_idx[2 , i] = i

            previous_parent_x = x_y_idx[0 , i]
            # print(num_parent,previous_parent_x,previous_parent_y)
            num_parent += 1
            
            subnodes = GetSubNode(x_y_idx[0 , i], x_y_idx[1 , i], r, num_edge)
        else: subnodes = GetSubNodeSub(x_y_idx[0 , i], x_y_idx[1 , i], r, num_edge)
        k = 0
        for idx_subnode in idx_subnodes:
            if x_y_idx[2, idx_subnode] == 0:
                x_y_idx[0, idx_subnode] = subnodes[k][0]
                x_y_idx[1, idx_subnode] = subnodes[k][1]
                x_y_idx[2, idx_subnode] = idx_subnode
                k += 1
        # else:
        #     subnodes = GetSubNode(x_y_idx[0, i], x_y_idx[1, i], r, num_edge)
        #     for k, idx_subnode in enumerate(idx_subnodes):
        #         if x_y_idx[2, idx_subnode] == 0:
        #             x_y_idx[0, idx_subnode] = subnodes[k][0]
        #             x_y_idx[1, idx_subnode] = subnodes[k][1]
        #             x_y_idx[2, idx_subnode] = idx_subnode
    a=0
    return x_y_idx

def SimpleGraphViz(x_y_idx_sorted : np.array, adjmatrix : np.array) -> None:
    fig, ax = plt.subplots()
    edge_idx1, edge_idx2 = np.nonzero(adjmatrix)
    #画线
    for i,j in zip(edge_idx1, edge_idx2):
        if i < j:
            weight = adjmatrix[i,j]
            ax.plot([x_y_idx_sorted[0,i],x_y_idx_sorted[0,j]],[x_y_idx_sorted[1,i],x_y_idx_sorted[1,j]],
                    color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adj_matrix))), 
                    linewidth = 0.15 * weight, 
                    zorder = 1)
    #画点
    ax.scatter(x_y_idx_sorted[0], x_y_idx_sorted[1], color = 'red', s = 10, zorder = 2) 
    for i in range(77):
        ax.text(x_y_idx_sorted[0, i], x_y_idx_sorted[1, i], s = i, fontsize = 5)
    # plt.close(fig)
    fig.show()


if __name__ == "__main__":

    dot_file_path = r"data/LesMiserables.dot" 
    num_node = 77
    dot_lines = ReadData(dot_file_path)
    adj_matrix = AdjacencyMatrix(num_node, dot_lines)
    x_y_idx_sorted = GetLocationParentSub(adj_matrix)
    SimpleGraphViz(x_y_idx_sorted, adj_matrix)

    #0.47129 s / run
    a=0