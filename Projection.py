import matplotlib.pyplot as plt
import numpy as np
import random
import re
import math
import time
import networkx as nx
from scipy.optimize import minimize
from scipy.special import comb
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from matplotlib import ticker


#---------------------------------------------------相似度矩阵（距离矩阵）--------------------------------------------------------------------------------



class layout:
	def __init__(
		self, 
		file_name: str,
		length: float = 14,
		width: float = 9):

		self.width  = width
		self.length = length
		
		self.file_name = file_name
		dot_file_path = self.file_name + ".dot"
		self.G = nx.DiGraph(nx.nx_pydot.read_dot(dot_file_path))
		distancemartix = self.FloydWarshallShortest(self.G.copy())
		embeddingIso = Isomap(n_components=2)
		X_Iso = embeddingIso.fit_transform(distancemartix)
		self.drawGraph(X_Iso, "Iso Projection")
		
	def FloydWarshallShortest(self,g) -> np.array:
		"""此算法见tutorial slides,目前未考虑权重因为大悲网络没有"""
		#init
		self.nodes = [int(i) for i in list(g.nodes)]
		self.edges = [[int(i),int(j)] for (i,j) in list(g.edges)]
		num_nodes = len(self.nodes)
		distancemartix = np.ones((num_nodes, num_nodes)) * np.inf
		#直接相连的边最短距离为1,同点最短距离为0
		for edge in self.edges:
			distancemartix[edge[0] - 1, edge[1] - 1] = 1
		for node in self.nodes:
			distancemartix[node - 1, node - 1] = 0
		#迭代
		for k in range(num_nodes):
			for i in range(num_nodes):
				for j in range(num_nodes):
					if distancemartix[i,j] > distancemartix[i,k] + distancemartix[k,j]:
						distancemartix[i,j] = distancemartix[i,k] + distancemartix[k,j]
		#异常情况处理
		if np.min(distancemartix) < 0:
			raise Exception("negative cycle in the graph!")
		if np.max(distancemartix) == np.inf:
			distancemartix[distancemartix == np.inf] = 10*np.max(distancemartix[distancemartix < np.inf])
		return distancemartix

	def projection(self, distancematrix, method):
		#input Isomap, MDS, TSNE
		X_proj=0
		if method== Isomap:
			embeddingIso = Isomap(n_components=2)
			X_proj = embeddingIso.fit_transform(distancemartix)

		if method == TSNE:
			emb_TSNE = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
			X_proj = emb_TSNE.fit_transform(distancemartix)

		if method == MDS:
			emb_MDS = MDS(n_components=2, normalized_stress='auto')
			X_proj = emb_MDS.fit_transform(distancemartix)

		return X_proj

	def drawGraph(self, X_proj, title):
		self.fig, self.ax = plt.subplots(figsize=(self.length, self.width), facecolor="white", constrained_layout=True)
		x, y = X_proj.T
		self.ax.scatter(x, y ,s=70, alpha=0.8)

		edge_coordinates = [(X_proj[edge[0]-1], X_proj[edge[1]-1]) for edge in self.edges]
		for i in range(len(X_proj)):
			self.ax.text(x[i],y[i], s = i+1, fontsize = 10)
		# draw lines
		for edge in edge_coordinates:
			self.ax.plot([edge[0][0], edge[1][0]], [edge[0][1], edge[1][1]], color ='orange', alpha=0.2)

		self.ax.set_title(title)
		self.ax.xaxis.set_major_formatter(ticker.NullFormatter())
		self.ax.yaxis.set_major_formatter(ticker.NullFormatter())
		
#----------------------------------------------------------------------------主函数--------------------------------------------------------------------------------

if __name__ == "__main__":
	dot_file_path = r"LesMisérablesNetwork"
	pro = layout(dot_file_path)
	a=0

