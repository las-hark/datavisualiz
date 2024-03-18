import matplotlib.pyplot as plt
import numpy as np
import random
import re
import math
import time
import networkx as nx
from matplotlib import patches
from scipy.spatial.distance import cdist
#
def ReadMultiGraph(file_path: str) -> dict:
	"""
	返回
	nodes:dict key是node序号,value是node的类型与所属subgraph
	edges:dict key是edge(起点，终点),value是类型、起点所属subgraph、终点所属subgraph
	alltypes:dict 存储node的类型、edge的类型、subgraph的类型
	"""
	with open(file_path, 'r') as file:
		dot_content = file.read()
	#init
	nodes = {}
	edges = {}
	node_types = []
	edge_types = []
	subgraph_types = []
	current_subgraph = None
	#save
	for line in dot_content.splitlines():
		line = line.strip()
		if line.startswith("subgraph"):
			current_subgraph = line.split(' ')[1]
			current_subgraph = current_subgraph.strip('"')
			if current_subgraph not in subgraph_types: subgraph_types.append(current_subgraph)

		elif "->" in line:
			parts = line.split("[")
			edge_nodes = parts[0].split("->")
			source = edge_nodes[0].strip()
			target = edge_nodes[1].strip()
			source = int(source[1:])
			target = int(target[1:])
			attributes = parts[1].rstrip("];")
			edge_type = attributes.split("=")[1].strip('"')
			if edge_type not in edge_types: edge_types.append(edge_type)
			# if nodes[source]['subgraph'] != nodes[target]['subgraph']:
			# 	a=0
			edges[(source,target)] = {"edge_type": edge_type,
									 "source_graph": nodes[source]['subgraph'],
									 "subnode_graph": nodes[target]['subgraph']}

		elif line.startswith("n"):
			parts = line.split("[")
			node_id = parts[0].strip()
			attributes = parts[1].rstrip("];")
			node_type = attributes.split('="')[-1][:-1]
			if node_type not in node_types: node_types.append(node_type)
			nodes[int(node_id[1:])] = {"node_type": node_type, "subgraph": current_subgraph}


	alltypes = {"node": node_types, "edge": edge_types, "subgraph": subgraph_types}

	return nodes, edges, alltypes

def IntraADJ(nodes: dict, edges: dict, subgraph: str):
	"""
	edge会跨subgraph相连,本函数返回一个subgraph内部的adj
	注意:此adj仍然囊括所有节点,只是不统计跨subgraph的edge,可能会影响结果,待调试
	"""
	#init
	num_node = len(nodes)
	adjmatrix = np.zeros((num_node, num_node))
	#节点映射列表
	node_map = dict(zip(list(nodes), list(range(num_node))))
	#save
	for edge, attri in list(edges.items()):
		if attri['source_graph'] == subgraph and attri['subnode_graph'] == subgraph:
			adjmatrix[node_map[edge[0]], node_map[edge[1]]] = 1
			adjmatrix[node_map[edge[1]], node_map[edge[0]]] = 1
	return adjmatrix, node_map

def InterADJ(nodes: dict, edges: dict):
	"""
	edge会跨subgraph相连,本函数返回跨subgraph的adj
	"""
	#init
	num_node = len(nodes)
	adjmatrix = np.zeros((num_node, num_node))
	#节点映射列表
	node_map = dict(zip(list(nodes), list(range(num_node))))
	#save
	for edge, attri in list(edges.items()):
		if attri['source_graph'] != attri['subnode_graph']:
			adjmatrix[node_map[edge[0]], node_map[edge[1]]] = 1
			adjmatrix[node_map[edge[1]], node_map[edge[0]]] = 1
	return adjmatrix, node_map


class FD:

	def __init__(
		self,
		adjmatrix: np.array,
		node_map: dict,
		nodes: dict,
		subgraph: str = 'intersubgraphs',
		C: float = 2.0,
		length: float = 8,
		width: float = 8,
		distance: float = 10):

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
		self.node_map  = node_map
		self.subgraph = subgraph
		self.nodes = nodes
		self.distance = distance


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
		times : int = 200000,
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
		#另一subgraph节点全为l
		for node in self.nodes:
			if self.nodes[node]['subgraph'] != self.subgraph:
				self.positions_nodes[:,self.node_map[node]] = self.distance



	def ForceGraphViz(self):
		plt.cla()
		nodes_real = list(self.node_map.keys())
		nodes_idx = list(self.node_map.values())
		reverse_fun = lambda x, lista, listb: lista[listb.index(x)]
		nodes_colormap = plt.cm.tab20b(np.linspace(0, 1, len(alltypes['node'])))
		edges_colormap = plt.cm.Set2_r(np.linspace(0, 1, len(alltypes['edge'])))
		for source_real,subnode_real in list(edges):
			source_graph = edges[(source_real, subnode_real)]['source_graph']
			subnode_graph = edges[(source_real, subnode_real)]['subnode_graph']
			# if self.subgraph != 'intersubgraphs':
			#     if source_graph != subnode_graph or source_graph != self.subgraph or subnode_graph != self.subgraph:
			#         continue
			source_type  = nodes[source_real]['node_type']
			subnode_type = nodes[subnode_real]['node_type']
			edge_type = edges[(source_real, subnode_real)]['edge_type']
			i = node_map[source_real]
			j = node_map[subnode_real]
			weight = self.matrix[i,j]
			self.ax.scatter(
							self.positions_nodes[0,i],
							self.positions_nodes[1,i],
							color = reverse_fun(source_type, nodes_colormap, alltypes['node']),
							s = 10, zorder = 3
			)
			self.ax.text(
						self.positions_nodes[0,i],
						self.positions_nodes[1,i],
						s =  source_real,
						fontsize = 6, zorder = 4
			)
			self.ax.scatter(
							self.positions_nodes[0,j],
							self.positions_nodes[1,j],
							color = reverse_fun(subnode_type, nodes_colormap, alltypes['node']),
							s = 10, zorder = 3
			)
			self.ax.text(
						self.positions_nodes[0,j],
						self.positions_nodes[1,j],
						s =  subnode_real,
						fontsize = 6, zorder = 4
			)
			self.ax.plot(
						[self.positions_nodes[0, i], self.positions_nodes[0, j]],
						[self.positions_nodes[1, i], self.positions_nodes[1, j]],
						color = reverse_fun(edge_type, edges_colormap, alltypes['edge']),
						linewidth = 0.5 * weight,
						zorder = 2
			)



		# self.ax.scatter(self.positions_nodes[0,:], self.positions_nodes[1,:], color = 'red', s = 10, zorder = 3)
		# for node_real,i in list(self.node_map.items()):
		# 	# for i in range(self.num_nodes):
		#     self.ax.text(self.positions_nodes[0, i], self.positions_nodes[1, i], s = node_real, fontsize = 6)
		# for i, j in zip(edge_idx1, edge_idx2):
		#     weight = self.matrix[i, j]

		#     edge_type = edges[(real_source,real_sub)]
		#     self.ax.plot([self.positions_nodes[0, i], self.positions_nodes[0, j]],
		#                  [self.positions_nodes[1, i], self.positions_nodes[1, j]],
		#                  color = (138/355, 43/255,226/255, 0.5 + weight * (0.5/ np.max(self.matrix))),
		#                  linewidth = 0.15 * weight,
		#                  zorder = 2)
		self.fig.show()
		a= 0

def NonBunLayout(nodes, edges, alltypes, positions_nodes, node_map):

	plt.cla()
	fig, ax = plt.subplots(figsize = (8,8))
	ax.axis ('off')

	nodes_real = list(node_map.keys())
	nodes_idx = list(node_map.values())
	reverse_fun = lambda x, lista, listb: lista[listb.index(x)]
	nodes_colormap = plt.cm.tab20b(np.linspace(0, 1, len(alltypes['node'])))
	edges_colormap = plt.cm.Set2_r(np.linspace(0, 1, len(alltypes['edge'])))
	for source_real,subnode_real in list(edges):
		source_graph = edges[(source_real, subnode_real)]['source_graph']
		subnode_graph = edges[(source_real, subnode_real)]['subnode_graph']
		# if self.subgraph != 'intersubgraphs':
		#     if source_graph != subnode_graph or source_graph != self.subgraph or subnode_graph != self.subgraph:
		#         continue
		source_type  = nodes[source_real]['node_type']
		subnode_type = nodes[subnode_real]['node_type']
		edge_type = edges[(source_real, subnode_real)]['edge_type']
		i = node_map[source_real]
		j = node_map[subnode_real]
		weight = 1
		ax.scatter(
						positions_nodes[0,i],
						positions_nodes[1,i],
						color = reverse_fun(source_type, nodes_colormap, alltypes['node']),
						s = 10, zorder = 3
		)
		ax.text(
					positions_nodes[0,i],
					positions_nodes[1,i],
					s =  source_real,
					fontsize = 6, zorder = 4
		)
		ax.scatter(
						positions_nodes[0,j],
						positions_nodes[1,j],
						color = reverse_fun(subnode_type, nodes_colormap, alltypes['node']),
						s = 10, zorder = 3
		)
		ax.text(
					positions_nodes[0,j],
					positions_nodes[1,j],
					s =  subnode_real,
					fontsize = 6, zorder = 4
		)
		ax.plot(
					[positions_nodes[0, i], positions_nodes[0, j]],
					[positions_nodes[1, i], positions_nodes[1, j]],
					color = reverse_fun(edge_type, edges_colormap, alltypes['edge']),
					linewidth = 0.5 * weight,
					zorder = 2
		)
	ax.plot([-45,6,6,-45,-45],[0,0,-40,-40,0],
			linewidth = 0.5,
			zorder = 1,
			color = (195/255,145/255,255/255)
	)
	ax.plot([8,35,35,8,8],[50,50,13,13,50],
			linewidth = 0.5,
			zorder = 1,
			color = (195/255,225/255,145/255)
	)
	fig.show()


def BunLayout(bunded_edges, nodes, edges, alltypes, positions_nodes, node_map):
	plt.cla()
	fig, ax = plt.subplots(figsize = (8,8))
	ax.axis ('off')
	
	nodes_real = list(node_map.keys())
	nodes_idx = list(node_map.values())
	reverse_fun = lambda x, lista, listb: lista[listb.index(x)]
	nodes_colormap = plt.cm.tab20b(np.linspace(0, 1, len(alltypes['node'])))
	edges_colormap = plt.cm.Set2_r(np.linspace(0, 1, len(alltypes['edge'])))
	for source_real,subnode_real in list(edges):
		source_graph = edges[(source_real, subnode_real)]['source_graph']
		subnode_graph = edges[(source_real, subnode_real)]['subnode_graph']
		if source_graph != subnode_graph:

			continue
		source_type  = nodes[source_real]['node_type']
		subnode_type = nodes[subnode_real]['node_type']
		edge_type = edges[(source_real, subnode_real)]['edge_type']
		i = node_map[source_real]
		j = node_map[subnode_real]
		weight = 1
		ax.scatter(
						positions_nodes[0,i],
						positions_nodes[1,i],
						color = reverse_fun(source_type, nodes_colormap, alltypes['node']),
						s = 10, zorder = 3
		)
		ax.text(
					positions_nodes[0,i],
					positions_nodes[1,i],
					s =  source_real,
					fontsize = 6, zorder = 4
		)
		ax.scatter(
						positions_nodes[0,j],
						positions_nodes[1,j],
						color = reverse_fun(subnode_type, nodes_colormap, alltypes['node']),
						s = 10, zorder = 3
		)
		ax.text(
					positions_nodes[0,j],
					positions_nodes[1,j],
					s =  subnode_real,
					fontsize = 6, zorder = 4
		)
		ax.plot(
					[positions_nodes[0, i], positions_nodes[0, j]],
					[positions_nodes[1, i], positions_nodes[1, j]],
					color = reverse_fun(edge_type, edges_colormap, alltypes['edge']),
					linewidth = 0.8 * weight,
					zorder = 2
		)
	for edge in bunded_edges:
		ax.plot( edge[:,0], edge[:,1], color = (132/255,132/255,235/255,0.3), linewidth = 0.5, zorder = 2)
	ax.plot([-45,6,6,-45,-45],[0,0,-40,-40,0],
			linewidth = 0.5,
			zorder = 1,
			color = (195/255,145/255,255/255)
	)
	ax.plot([8,35,35,8,8],[50,50,13,13,50],
			linewidth = 0.5,
			zorder = 1,
			color = (195/255,225/255,145/255)
	)
	fig.show()

def get_interEdges (wholegraph_pos, inter_adj):
	"""
	wholegraph_pos -- nodes_position for whole graph
	inter_adj -- adjmatrix record inter subgraph edges
	有且只有连接subgraph edge的信息

	return the edges in [[[node1]， [node2]], ... ]
	"""
	inter_adj_matrix = np.where(np.array(inter_adj) ==1)
	nodeP = inter_adj_matrix[0] #是index
	nodeQ = inter_adj_matrix[1]

	posX = wholegraph_pos[0,:]
	posY = wholegraph_pos[1,:]

	interEdge = []
	for i in range(len(nodeP)):
		edge = [[posX[nodeP[i]],posY[nodeP[i]]], [posX[nodeQ[i]],posY[nodeQ[i]]]]
		interEdge.append(edge)

	return np.array(interEdge)

def subdivide_edge(edges: np.ndarray, num_points: int) -> np.ndarray:
	"""Subdivide edges into `num_points` segments. 分割edges
	Parameters
	----------
	edges : array-like, shape (n_edges, current_points, 2)
		The edges to subdivide.
	num_points : int
		The number of points to generate along each edge.
	Returns
	-------
	new_points : array-like, shape (n_edges, num_points, 2)
	"""
	segment_vecs = edges[:, 1:] - edges[:, :-1]
	segment_lens = np.linalg.norm(segment_vecs, axis=-1)
	cum_segment_lens = np.cumsum(segment_lens, axis=1)
	cum_segment_lens = np.hstack(
		[np.zeros((cum_segment_lens.shape[0], 1)), cum_segment_lens]
	)

	total_lens = cum_segment_lens[:, -1]

	# At which lengths do we want to generate new points
	t = np.linspace(0, 1, num=num_points, endpoint=True)
	desired_lens = t * total_lens[:, None]
	# Which segment should the new point be interpolated on
	i = np.argmax(desired_lens[:, None] < cum_segment_lens[..., None], axis=1)
	# At what percentage of the segment does this new point actually appear
	pct = (desired_lens - np.take_along_axis(cum_segment_lens, i - 1, axis=-1)) / (
		np.take_along_axis(segment_lens, i - 1, axis=-1) + 1e-8
	)

	row_indices = np.arange(edges.shape[0])[:, None]
	new_points = (
		(1 - pct[..., None]) * edges[row_indices, i - 1]
		+ pct[..., None] * edges[row_indices, i]
	)

	return new_points


def compute_edge_compatibility(edges: np.ndarray) -> np.ndarray:
	"""Compute pairwise-edge compatibility scores.
	Parameters
	----------
	edges : array-like, shape (n_edges, n_points, 2)
		The edges to compute compatibility scores for.

	Returns
	-------
	compat : array-like, shape (n_edges, n_edges)
		The pairwise edge compatibility scores.
	"""
	vec = edges[:, -1] - edges[:, 0]
	vec_norm = np.linalg.norm(vec, axis=1, keepdims=True)

	# Angle comptability
	compat_angle = np.abs((vec @ vec.T) / (vec_norm @ vec_norm.T + 1e-8))

	# Length compatibility
	l_avg = (vec_norm + vec_norm.T) / 2
	compat_length = 2 / (
		l_avg / (np.minimum(vec_norm, vec_norm.T) + 1e-8)
		+ np.maximum(vec_norm, vec_norm.T) / (l_avg + 1e-8)
		+ 1e-8
	)

	# Distance compatibility
	midpoint = (edges[:, 0] + edges[:, -1]) / 2
	midpoint_dist = np.linalg.norm(midpoint[None, :] - midpoint[:, None], axis=-1)
	compat_dist = l_avg / (l_avg + midpoint_dist + 1e-8)

	# Visibility compatibility
	ap = edges[None, ...] - edges[:, None, None, 0]
	t = np.sum(ap * vec[:, None, None, :], axis=-1) / (
		np.sum(vec**2, axis=-1)[:, None, None] + 1e-8
	)
	I = edges[:, None, 0, None] + t[..., None] * vec[:, None, None, :]

	i0, i1 = I[..., 0, :], I[..., 1, :]
	Im = (i0 + i1) / 2

	denom = np.sqrt(np.sum((i0 - i1) ** 2, axis=-1))
	num = 2 * np.linalg.norm(midpoint[:, None, ...] - Im, axis=-1)

	compat_visibility = np.maximum(0, 1 - num / (denom + 1e-8))
	compat_visibility = np.minimum(compat_visibility, compat_visibility.T)

	# Combine compatibility scores
	return compat_angle * compat_length * compat_dist * compat_visibility


def compute_forces(e: np.ndarray, e_compat: np.ndarray, kp: np.ndarray) -> np.ndarray:
	"""Compute forces on each edge point.

	Parameters
	----------
	e : array-like, shape (n_edges, n_points, 2)，The edge points.
	e_c，ompat : array-like, shape (n_edges, n_edges) The pairwise edge compatibility scores.
	kp : array-like, shape (n_edges, 1, 1)， The spring constant for each edge.
	Returns
	-------
	F : array-like, shape (n_edges, n_points, 2)
		The forces on each edge point.
	"""
	# Left-mid spring direction
	v_spring_l = e[:, :-1] - e[:, 1:]
	v_spring_l = np.concatenate(
		[np.zeros((v_spring_l.shape[0], 1, v_spring_l.shape[-1])), v_spring_l],
		axis=1,
	)

	# Right-mid spring direction
	v_spring_r = e[:, 1:] - e[:, :-1]
	v_spring_r = np.concatenate(
		[v_spring_r, np.zeros((v_spring_l.shape[0], 1, v_spring_l.shape[-1]))],
		axis=1,
	)

	f_spring_l = np.sum(v_spring_l**2, axis=-1, keepdims=True)
	f_spring_r = np.sum(v_spring_r**2, axis=-1, keepdims=True)

	F_spring = kp * (f_spring_l * v_spring_l + f_spring_r * v_spring_r)

	# Electrostatic force
	v_electro = e[:, None, ...] - e[None, ...]
	f_electro = e_compat[..., None] / (np.linalg.norm(v_electro, axis=-1) + 1e-8)

	F_electro = np.sum(f_electro[..., None] * v_electro, axis=0)

	F = F_spring + F_electro
	# The first and last points are fixed
	F[:, 0, :] = F[:, -1, :] = 0

	return F


def fdeb(
	edges: np.ndarray,
	K: float = 0.1,
	n_iter: int = 60,
	n_iter_reduction: float = 2 / 3,
	lr: float = 0.04,
	lr_reduction: float = 0.5,
	n_cycles: int = 8,
	initial_segpoints: int = 1,
	segpoint_increase: float = 2,
	compat_threshold: float = 0.5,
) -> np.ndarray:

	"""Run the Force-Directed Edge Bundling algorithm.
	返回的是edge画线position
	edges的input格式为 [[[node1],[node2]], ... ]

	Parameters
	----------
	edges: array-like, shape (n_edges, 2, 2)
		The edge points.
	K: float
		The spring constant.
	n_iter: int
		The number of iterations to run in the first cycle.
	n_iter_reduction: float
		The factor by which to reduce the number of iterations in each cycle.
	lr: float
		The learning rate.
	lr_reduction: float
		The factor by which to reduce the learning rate in each cycle.
	n_cycles: float
		The number of cycles to run the algorithm for. In each cycle, the number
		of segments is increased by a factor `segpoint_increase
	initial_segpoints: int
		The initial number of segments to start with, e.g., 1 corresponds to a
		single midpoint.
	segpoint_increase: float
		The factor by which to increase the number of segments in each cycle.
	compat_threshold: float
		Edge interactions with compatibility lower than a specified threshold
		are ignored.

	Returns
	-------
	画edge的数据
	edges: array-like, shape (n_edges, n_segments + 1, 2)

	"""
	initial_edge_vecs = edges[:, 0] - edges[:, -1]

	initial_edge_lengths = np.linalg.norm(initial_edge_vecs, axis=-1, keepdims=True)
	edge_compatibilities = compute_edge_compatibility(edges)
	edge_compatibilities = (edge_compatibilities > compat_threshold).astype(np.float32)

	num_segments = initial_segpoints

	for cycle in range(n_cycles):
		edges = subdivide_edge(edges, num_segments + 2)  # Add 2 for endpoints
		num_segments = int(np.ceil(num_segments * segpoint_increase))

		kp = K / (initial_edge_lengths * num_segments + 1e-8)
		kp = kp[..., None]

		for epoch in range(n_iter):
			F = compute_forces(edges, edge_compatibilities, kp)
			edges += F * lr

		n_iter = int(np.ceil(n_iter * n_iter_reduction))
		lr = lr * lr_reduction

	return edges

#----------------------------------------------------------------------------主函数--------------------------------------------------------------------------------

if __name__ == "__main__":
	dot_file_path = r"devonshiredebate_twoclusters.dot"
	nodes, edges, alltypes = ReadMultiGraph(dot_file_path)
	subgraph1 = alltypes['subgraph'][0]
	subgraph2 = alltypes['subgraph'][1]
	intraadj_sub1, node_map = IntraADJ(nodes, edges, subgraph1)
	fd_sub1 = FD(adjmatrix = intraadj_sub1, node_map = node_map, subgraph = subgraph1, nodes = nodes, distance = 10)
	fd_sub1.ForceModel()
	fd_sub1.ForceGraphViz()
	fd_sub1.fig.show()

	intraadj_sub2, node_map = IntraADJ(nodes, edges, subgraph2)
	fd_sub2 = FD(adjmatrix = intraadj_sub2, node_map = node_map, subgraph = subgraph2, nodes = nodes, distance = -10)
	fd_sub2.ForceModel()
	fd_sub2.ForceGraphViz()
	fd_sub2.fig.show()

	pos_whole = fd_sub1.positions_nodes + fd_sub2.positions_nodes
	NonBunLayout(nodes,edges,alltypes,pos_whole,node_map)
	interadj, node_map = InterADJ(nodes, edges)
	inter_edges = get_interEdges(pos_whole, interadj)
	bunded_edges = fdeb(inter_edges) 
	BunLayout(bunded_edges, nodes, edges, alltypes, pos_whole, node_map)
	a=0






