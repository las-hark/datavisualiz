import matplotlib.pyplot as plt
from matplotlib import patches
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
            weight = 1#int(edge[2])
            matrix[node1-1, node2-1] = weight
            matrix[node2-1, node1-1] = weight

    return matrix

def SortNodesByEdgeNum(unsorted: np.array) -> np.array:

    nonzero_counts = np.count_nonzero(unsorted, axis = -1)
    edge_num_sorted     = np.sort(nonzero_counts)[::-1]
    edge_num_idx_sorted = np.argsort(nonzero_counts)[::-1]

    return np.stack((edge_num_idx_sorted, edge_num_sorted))

def BFS(adjmatrix : np.array) -> np.array:
    """从最多edge的node, 逐步扩展
    首先访问起始顶点v,接着由v出发,
    依次访问v的各个未访问过的邻接顶点w1,w2,…,wi,
    然后再依次访问w1,w2,…,wi的所有未被访问过的邻接顶点,
    再从这些访问过的顶点出发，再访问它们所有未被访问过的邻接顶点……
    依次类推，直到图中所有顶点都被访问过为止。"""
   
    num_node = len(adjmatrix)
    x_y_ord_parent = np.zeros((4 , num_node))
    level_node = np.zeros(num_node)
    node_sorted = SortNodesByEdgeNum(adj_matrix)
    #initialize
    loop = sub = 0
    start_node = node_sorted[0, 0]
    x_y_ord_parent[3, start_node] = 0.1
    #search
    while loop < num_node:
        idx_subnodes = np.nonzero(adjmatrix[start_node])[0]
        subnode_sorted = SortNodesByEdgeNum(adjmatrix[idx_subnodes])
        for k in idx_subnodes[subnode_sorted[0]]:
            if (x_y_ord_parent[2, k] == 0) and (k != node_sorted[0, 0]):
                sub += 1
                x_y_ord_parent[2, k] = sub #标注访问序号
                x_y_ord_parent[3, k] = start_node #标注父节点
        start_node = np.where(x_y_ord_parent[2] == (loop + 1))[0][0]
        loop += 1
        if (num_node-1) in x_y_ord_parent[2]: break
        
    return x_y_ord_parent
        

def DFS(adjmatrix : np.array) -> np.array:
    """首先访问图中某一起始顶点v,然后由v出发,访问与v邻接且未被访问的任一顶点w1,
    再访问与w1邻接且未被访问的任一顶点w2,……重复上述过程。
    当不能再继续向下访问时，依次退回到最近被访问的顶点，
    若它还有邻接顶点未被访问过，则从该点开始
    依次类推，直到图中所有顶点均被访问过为止"""
    
    num_node = len(adjmatrix)
    x_y_ord_parent = np.zeros((4 , num_node))
    nodes_sorted = SortNodesByEdgeNum(adj_matrix)
    #initialize
    loop = order = idx = level = 0
    startnode = nodes_sorted[0, 0]
    x_y_ord_parent[3, startnode] = 0.1
    #search
    while loop < num_node:
        idx_subnodes = np.nonzero(adjmatrix[startnode])[0]
        subnodes_sorted = SortNodesByEdgeNum(adjmatrix[idx_subnodes])
        #找到第一个没访问过的subnode
        for i in idx_subnodes[subnodes_sorted[0]]:
            flag = 0
            #找到第一个没遍历过的点
            if (x_y_ord_parent[2, i] == 0) and (i != nodes_sorted[0, 0]):
                subnode = i #下一个startnode即该未被遍历的点
                flag = 1             
                order += 1 #更新访问序号
                idx  = order #idx也代表访问序号，但idx会减小（回退）来指导倒回过程，order不会
                x_y_ord_parent[2, i] = order
                x_y_ord_parent[3, i] = startnode
                startnode = subnode #startnode更新为当前node的subnode
                break
        #第二重判断：是否还有可访问节点？ 没有了往回退，loop不增加
        if flag == 0:
            idx -= 1 
            if idx != 0:
                startnode = np.where(x_y_ord_parent[2] == (idx))[0][0]
            else: startnode = nodes_sorted[0, 0]
            #bug:当退回root的时候出问题 update:de了
            continue
        loop += 1
        if (num_node-1) in x_y_ord_parent[2]: break

    return x_y_ord_parent


def angle_between_points(x1, y1, x2, y2, x0, y0):
    """
    计算p1和p2形成的向量与p0形成的向量之间的夹角
    p1, p2, p0是表示点的元组 (x, y)
    返回夹角（以弧度表示）
    """
    #计算向量p1p0和p2p0
    v1 = (x1 - x0, y1 - y0)
    v2 = (x2 - x0, y2 - y0)
    #v1旋转到v2，逆时针为正，顺时针为负
    #向量模的乘积
    TheNorm = np.linalg.norm(v1) * np.linalg.norm(v2)
    #叉乘
    rho = np.rad2deg(np.arcsin(np.cross(v1, v2) / TheNorm))
    #点乘
    theta = np.rad2deg(np.arccos(np.dot(v1, v2) / TheNorm))
    if rho < 0:
        return - theta
    else:
        return theta


def GetLocation_Bubble(x_y_ord_parent) -> [np.array, np.array]:
    """
    根据访问序号与父节点生成坐标
    """
    a_r_d = np.zeros((3, len(x_y_ord_parent[0])))
    a_r_d[1, np.argsort(x_y_ord_parent[2])[0]] = 8 #待改,这个是自己设的，应该是参数之一，root也是影响因素
    x_center_node = y_center_node = 0
    x_bends = []
    y_bends = []
    ancester_bends = []
    idx_bends = []
    # 待优化,感觉可以更简洁一些
    # 按照访问顺序将索引排序，i是节点索引
    for i in np.argsort(x_y_ord_parent[2]):
        #若节点i是父节点之一，算其子节点排布，否则跳过
        if i not in x_y_ord_parent[3]: 
            continue
        else:
            #节点i作为父节点的相关参数
            parent_radiu  = a_r_d[1, i]
            parent_weight = 1 + len(x_y_ord_parent[3, x_y_ord_parent[3] == i])
            x_center_node = x_y_ord_parent[0, i]
            y_center_node = x_y_ord_parent[1, i]
            ancester = x_y_ord_parent[3, i]
            idx_center_node = i            
            #将节点i的子节点按照访问顺序排序
            order_subnodes = np.sort(x_y_ord_parent[2, x_y_ord_parent[3] == idx_center_node])
            #保存节点i的所有子节点索引与权重（子节点自己的子节点数量）
            idx_weight_subnodes = np.zeros((2, len(order_subnodes) + 1)) #将bend考虑进来因此+1
            for index, order in enumerate(order_subnodes):
                idx_weight_subnodes[0, index] = np.where(x_y_ord_parent[2] == order)[0][0]
                idx_weight_subnodes[1, index] = 1 + len(x_y_ord_parent[2, x_y_ord_parent[3] == idx_weight_subnodes[0, index]])
            idx_weight_subnodes[0, -1] = 0.1 #bend标记为0.1
            idx_weight_subnodes[1, -1] = 1 #bend权重为1
            #子节点enclosing circle的半径radius：基于节点i的半径按权重确定
            radius = parent_radiu * idx_weight_subnodes[1] / parent_weight
            #子节点所在sector的角度angles
            largest_weight = np.max(idx_weight_subnodes[1])
            sum_weights    = np.sum(idx_weight_subnodes[1])
            if (largest_weight / sum_weights) > 0.5 and (largest_weight / sum_weights) < 1:
                #若某一子节点权重过大，控制其angle不超过pi
                angles = math.pi * idx_weight_subnodes[1] / np.sum(idx_weight_subnodes[1, idx_weight_subnodes[1] != largest_weight])
                angles[idx_weight_subnodes[1] == largest_weight] = math.pi
            else: 
                #无过大子节点的普通情况：按权重分配
                angles = 2 * math.pi * idx_weight_subnodes[1]/ np.sum(idx_weight_subnodes[1])
            #bug: parent_radiu必须迭代！ update:已经de了
            #求子节点圆心到节点i距离
            distances = np.maximum(radius + parent_radiu, radius / (np.sin(angles/2) + 0.0001))
            #求坐标
            x_y_subnodes = np.zeros((2,len(angles)))
            sum_angles = subnode_idx = 0
            for angle, distance in zip(angles, distances):
                sum_angles += angle                
                x_y_subnodes[0, subnode_idx] = x_center_node + distance * math.cos(sum_angles - angle / 2)
                x_y_subnodes[1, subnode_idx] = y_center_node + distance * math.sin(sum_angles - angle / 2)
                subnode_idx += 1
            x_bend_ori = x_y_subnodes[0, -1]
            y_bend_ori = x_y_subnodes[1, -1]
            #加bend，直接将dummycircle作为另一个子节点
            #根据bend旋转, 有ancestor才转，旋转角由节点i坐标与bend坐标决定，bend要转到i的父节点与i连线上
            if ancester != 0.1:
                x_ancester = x_y_ord_parent[0, int(ancester)]
                y_ancester = x_y_ord_parent[1, int(ancester)]
                rad = math.radians(angle_between_points(x_bend_ori, y_bend_ori, x_ancester, y_ancester, x_center_node, y_center_node))
                x_subnodes_ori = np.copy(x_y_subnodes[0])
                y_subnodes_ori = np.copy(x_y_subnodes[1])
                x_y_subnodes[0] = x_center_node + (x_subnodes_ori- x_center_node) * math.cos(rad) - (y_subnodes_ori - y_center_node) * math.sin(rad)
                x_y_subnodes[1] = y_center_node + (x_subnodes_ori - x_center_node) * math.sin(rad) + (y_subnodes_ori - y_center_node) * math.cos(rad)
                #bug 转反了 update：de了
                x_bends.append(x_y_subnodes[0, -1])
                y_bends.append(x_y_subnodes[1, -1])
                ancester_bends.append(int(ancester))
                idx_bends.append(idx_center_node)

            idx_subnodes = idx_weight_subnodes[0, :-1].astype(int)
            x_y_ord_parent[0, idx_subnodes] = x_y_subnodes[0, :-1]
            x_y_ord_parent[1, idx_subnodes] = x_y_subnodes[1, :-1]
            a_r_d[0, idx_subnodes] = angles[:-1]
            a_r_d[1, idx_subnodes] = radius[:-1]
            a_r_d[2, idx_subnodes] = distances[:-1]

    x_y_idx_ancester_bends = np.array((x_bends, y_bends, idx_bends, ancester_bends))
    return x_y_ord_parent, x_y_idx_ancester_bends, a_r_d[1]

def SimpleGraphViz(x_y_idx_sorted : np.array, adjmatrix : np.array) -> None:
    fig, ax = plt.subplots()
    edge_idx1, edge_idx2 = np.nonzero(adjmatrix)
    #画线
    for i,j in zip(edge_idx1, edge_idx2):
        if i < j:
            weight = adjmatrix[i,j]
            ax.plot([x_y_idx_sorted[0, i],x_y_idx_sorted[0, j]],
                    [x_y_idx_sorted[1, i],x_y_idx_sorted[1, j]],
                    color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adj_matrix))), 
                    linewidth = 0.15 * weight, 
                    zorder = 1)
    #画点
    ax.scatter(x_y_idx_sorted[0], x_y_idx_sorted[1], color = 'red', s = 10, zorder = 2) 
    for i in range(77):
        ax.text(x_y_idx_sorted[0, i], x_y_idx_sorted[1, i], s = i, fontsize = 10)
    # plt.close(fig)
    fig.show()
    a = 0
    return fig

def BendGraphViz(
    x_y_ord_parent: np.array, 
    x_y_idx_ancester_bends: np.array, 
    radius: np.array, 
    adjmatrix: np.array) -> None:

    fig, ax = plt.subplots()
    edge_idx1, edge_idx2 = np.nonzero(adjmatrix)
    real = edge_idx1 < edge_idx2
    edge_idx1 = edge_idx1[real]
    edge_idx2 = edge_idx2[real]
    #思路：先确定哪些点之间是有bend的，将这些点有bend的edge先画了，再画剩下没有bend的edge
    #bug：看图发现bend算法有问题，现在的方法实际上是让ancester和父圆圆心连线上不要出现点，而不是加拐点
    ax.scatter(x_y_ord_parent[0], x_y_ord_parent[1], color = 'red', s = 10, zorder = 3) 
    for i, j in enumerate(x_y_ord_parent[3]):
        i = int(i)
        j = int(j)
        weight = adjmatrix[i,j]
        ax.plot([x_y_ord_parent[0, i], x_y_ord_parent[0, j]],
                [x_y_ord_parent[1, i], x_y_ord_parent[1, j]],
                        color = (138/355, 43/255,226/255, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
                        linewidth = 0.15 * weight, 
                        zorder = 2)
             
    # for i, j in zip(edge_idx1, edge_idx2):
    #     weight = adjmatrix[i, j]
    #     if (j != x_y_ord_parent[3, i]) and (i != x_y_ord_parent[3, j]):
    #         ax.plot([x_y_ord_parent[0, i], x_y_ord_parent[0, j]],
    #         [x_y_ord_parent[1, i], x_y_ord_parent[1, j]],
    #                 color = (0, 150/255, 1, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
    #                 linewidth = 0.15 * weight, 
    #                 zorder = 1)
    # for i,j in zip(edge_idx1, edge_idx2):
    #         weight = adjmatrix[i,j]
    #         if (i in x_y_idx_ancester_bends[2] and j == x_y_idx_ancester_bends[3, np.where(x_y_idx_ancester_bends[2] == i)[0][0]]):
    #             ind = np.where(x_y_idx_ancester_bends[2] == i)[0][0]
    #             x_bend = x_y_idx_ancester_bends[0, ind]
    #             y_bend = x_y_idx_ancester_bends[1, ind]
    #             ax.plot([x_bend, x_y_ord_parent[0, i]],
    #                 [y_bend, x_y_ord_parent[1, i]],
    #                 color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
    #                 linewidth = 0.15 * weight, 
    #                 zorder = 1)
    #             ax.plot([x_bend, x_y_ord_parent[0, j]],
    #                 [y_bend, x_y_ord_parent[1, j]],
    #                 color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
    #                 linewidth = 0.15 * weight, 
    #                 zorder = 1)
    #         elif (j in x_y_idx_ancester_bends[2] and i == x_y_idx_ancester_bends[3, np.where(x_y_idx_ancester_bends[2] == j)[0][0]]):
    #             ind = np.where(x_y_idx_ancester_bends[2] == j)[0][0]
    #             x_bend = x_y_idx_ancester_bends[0, ind]
    #             y_bend = x_y_idx_ancester_bends[1, ind]
    #             ax.plot([x_bend, x_y_ord_parent[0, i]],
    #                 [y_bend, x_y_ord_parent[1, i]],
    #                 color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
    #                 linewidth = 0.15 * weight, 
    #                 zorder = 1)
    #             ax.plot([x_bend, x_y_ord_parent[0, j]],
    #                 [y_bend, x_y_ord_parent[1, j]],
    #                 color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
    #                 linewidth = 0.15 * weight, 
    #                 zorder = 1)
    #         else:
    #             ax.plot([x_y_ord_parent[0, i], x_y_ord_parent[0, j]],
    #                     [x_y_ord_parent[1, i], x_y_ord_parent[1, j]],
    #                     color = (0, 0, 1, 0.5 + weight * (0.5/ np.max(adjmatrix))), 
    #                     linewidth = 0.15 * weight, 
    #                     zorder = 1)
    #画点

    for i in range(len(adjmatrix)):
        ax.text(x_y_ord_parent[0, i], x_y_ord_parent[1, i], s = i, fontsize = 6)
        ax.add_patch(patches.Circle(
                    ( x_y_ord_parent[0, i], x_y_ord_parent[1, i]),
                    radius[i],  fill = False,
                    color = (219/255, 112/255, 147/255, 0.3 + weight * (0.7/ np.max(adjmatrix)))
                    ))
    # plt.close(fig)
    ax.axis ('off')
    fig.show()
    a = 0
    return fig, ax



if __name__ == "__main__":

    dot_file_path = r"JazzNetwork.dot" #
    num_node = 198
    dot_lines = ReadData(dot_file_path)
    adj_matrix = AdjacencyMatrix(num_node, dot_lines)
    x_y_ord_parent = DFS(adj_matrix)
    x_y_ord_parent, x_y_idx_ancester_bends, radius = GetLocation_Bubble(x_y_ord_parent)
    # fig = SimpleGraphViz(x_y_idx_sorted, adj_matrix)
    fig, ax = BendGraphViz(x_y_ord_parent, x_y_idx_ancester_bends, radius , adj_matrix)
    a=0