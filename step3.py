import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial.distance import cdist
import numpy as np
import random
import re
import math
import time


class ReadFun:

    def ReadData(file_path : str) -> list:
        """read dot file and save as a list"""

        with open(file_path, 'r') as file:
            dot_content = file.read()
        lines = dot_content.split(';\n')
        lines[0] = '1'
        return lines

    def AdjacencyMatrix(file_path: str, num_node : int, with_weight: bool) -> np.array:
        """trans the dot list into a adjacenct matrix"""

        lines = ReadFun.ReadData(file_path)
        matrix = np.zeros((num_node,num_node))
        sum_edge = 0
        for line in lines:
            if '--' in line:
                sum_edge += 1
                edge = re.split('--|\[weight=|\]',line)
                node1  = int(edge[0])
                node2  = int(edge[1])
                if with_weight == True:
                    weight = int(edge[2])
                else: weight = 1
                #weight = 1#int(edge[2])
                matrix[node1-1, node2-1] = weight
                matrix[node2-1, node1-1] = weight

        return matrix

class layout:

    def __init__(
        self, 
        adjmatrix: np.array,
        C: float = 2.0,
        length: float = 8,
        width: float = 8):

        """
        adjmatrix: 领接矩阵
        C: ideal length系数参数(调参)
        length/width: 画布尺寸
        """

        fig, ax = plt.subplots(figsize = (length, width))
        ax.axis ('off')
        self.fig    = fig
        self.ax     = ax
        self.width  = width
        self.length = length
        self.num_nodes = len(adjmatrix)
        self.matrix    = adjmatrix
        self.C = C
        self.L = (length * width / self.num_nodes) ** 0.5 * self.C    



    def ForceModelInit(self):
        #position init(shape: 2 x numnodes)
        np.random.seed(42)
        self.positions_nodes = 10 * np.random.rand(2, self.num_nodes)
        
    
    def GetPara(self):
        #|Pv - Pu| distances init(shape: numnodes x numnodes)
        self.distance_nodes = cdist(self.positions_nodes.T, self.positions_nodes.T)
        #PuPv vectors init(shape: 2 x numnodes x numnodes)
        pos_u_v_vectors = np.zeros((2, self.num_nodes, self.num_nodes))
        # for i in range(self.num_nodes):
        #     #Puv = Pv - Pu
        #     pos_u_v_vectors[0, i, :] = self.positions_nodes[0, :] - self.positions_nodes[0, i]
        #     pos_u_v_vectors[1, i, :] = self.positions_nodes[1, :] - self.positions_nodes[1, i]
        pos_u_v_vectors =  np.expand_dims(self.positions_nodes, axis=1) - np.expand_dims(self.positions_nodes, axis=2) 
        self.pos_u_v_vectors = pos_u_v_vectors
        #replusive force(shape: 2 x numnodes x numnodes)
        #第一维x方向受力与y方向受力，第二维u，第三维v，[i,j,k]表示Node k对Node j在i方向上的力
        self.f_rep = np.nan_to_num(np.expand_dims((self.L ** 2 / self.distance_nodes), axis = 0) * self.pos_u_v_vectors)
        #attractive force only for adjecent nodes(shape:2 x numnodes x numnodes)
        #维度定义同上
        adj_bool = self.matrix > 0 #adjecent nodes mask
        self.f_att = np.nan_to_num(np.expand_dims((self.distance_nodes ** 2   * adj_bool / self.L), axis = 0) * (-1) * self.pos_u_v_vectors)
        #spring force(shape: 2 x numnodes x numnodes)
        #维度定义同上
        self.f_spr = self.f_att + self.f_rep
        #以上全验证了，都没算错

    def ForceModel(
        self, 
        times : int = 500000,
        upadate_rate : float = 0.00001):

        """
        times:迭代次数(调参)
        update_rate:更新率(调参)
        """

        self.ForceModelInit()
        self.GetPara()
        for i in range(times):
            self.positions_nodes += upadate_rate * np.sum(self.f_spr, axis = 1)
            # for j in range(self.num_nodes):
            #     self.positions_nodes[0, j] = min(self.width, max(0, self.positions_nodes[0, j]))
            #     self.positions_nodes[1, j] = min(self.length, max(0, self.positions_nodes[1, j]))
            self.GetPara()

    def ForceGraphViz(self):
        plt.cla()
        edge_idx1, edge_idx2 = np.nonzero(self.matrix)
        real = edge_idx1 < edge_idx2
        edge_idx1 = edge_idx1[real]
        edge_idx2 = edge_idx2[real]
        self.ax.scatter(self.positions_nodes[0,:], self.positions_nodes[1,:], color = 'red', s = 10, zorder = 3)
        for i in range(self.num_nodes):
            self.ax.text(self.positions_nodes[0, i], self.positions_nodes[1, i], s = i, fontsize = 6)
        for i, j in zip(edge_idx1, edge_idx2):
            weight = self.matrix[i, j]
            self.ax.plot([self.positions_nodes[0, i], self.positions_nodes[0, j]],
                         [self.positions_nodes[1, i], self.positions_nodes[1, j]],
                         color = (138/355, 43/255,226/255, 0.5 + weight * (0.5/ np.max(self.matrix))), 
                         linewidth = 0.15 * weight, 
                         zorder = 2)
        self.fig.show()


if __name__ == "__main__":

    dot_file_path = r"LesMiserables.dot" #"JazzNetwork.dot""LesMiserables.dot"
    num_node = 77 #198 77
    adj_matrix = ReadFun.AdjacencyMatrix(dot_file_path, num_node, with_weight = True)
    force_directed = layout(adjmatrix = adj_matrix)
    # force_directed.ForceModelInit()
    # force_directed.ForceGraphViz()
    force_directed.ForceModel()
    force_directed.ForceGraphViz()
    force_directed.fig.show()
    a=0