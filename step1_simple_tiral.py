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
    return matrix

def  GetLocation(adjmatrix : np.array) -> np.array:
    x_y_idx = np.zeros((3,77))
    x_y_idx[0] = [random.randint(0, 100) for _ in range(77)]
    x_y_idx[1] = [random.randint(0, 100) for _ in range(77)]
    x_y_idx[2] = list(range(77))
    return x_y_idx


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
    
    plt.close(fig)#fig.show()
    a=0


if __name__ == "__main__":
    import time
    sumtime = 0
    for i in range(100):
        a = time.time()
        dot_file_path = r"data/LesMiserables.dot" 
        num_node = 77
        dot_lines = ReadData(dot_file_path)
        adj_matrix = AdjacencyMatrix(num_node, dot_lines)
        x_y_idx_sorted = GetLocation(adj_matrix)
        SimpleGraphViz(x_y_idx_sorted, adj_matrix)
        b = time.time()
        sumtime += (b- a)
    print(sumtime/100)
    #0.47276 s/run
    a=0