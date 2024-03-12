import matplotlib.pyplot as plt
import numpy as np
import random
import re
import math
import time
import networkx as nx
from scipy.optimize import minimize
from scipy.special import comb


#remove cycle , contract DAG
#based on Feedback Arc Set 
"""
A feedback arc set of a directed graph is a subset of the edges 
such that if those edges are removed from the graph, the resulting graph becomes acyclic.
"""

def get_nodes_degree_dict(g,nodes):
	"""
	返回字典， node 和 edge的箭头方向
	{'5': (0, 'out'),
     '13': (1.0, 'in'),」 
	"""
	# get nodes degree dict: key = node, value = (max(d(in)/d(out),d(out)/d(in),"in" or "out")
	in_degrees = g.in_degree(nodes)
	out_degrees = g.out_degree(nodes)
	degree_dict = {}
	for node in nodes:
		in_d = in_degrees[node]
		out_d = out_degrees[node]
		if in_d >= out_d:
			try:
				value = in_d * 1.0 / out_d
			except Exception as e:
				value = 0
			f = "in"
		else:
			try:
				value = out_d * 1.0 / in_d
			except Exception as e:
				value = 0
			f = "out"
		degree_dict[node] = (value,f)
		#print("node: %d: %s" % (node,degree_dict[node]))
	return degree_dict


def greedy_local_heuristic(sccs,degree_dict,edges_to_be_removed):
	while True:
		graph = sccs.pop()
		temp_nodes_degree_dict = {}
		for node in graph.nodes():
			temp_nodes_degree_dict[node] = degree_dict[node][0]

		#from helper_funs import pick_from_dict
		#max_node,_ = pick_from_dict(temp_nodes_degree_dict)
		#max_value = degree_dict[max_node]
			
		degrees = [(node,degree_dict[node]) for node in list(graph.nodes())]
		max_node,max_value = max(degrees,key = lambda x: x[1][0])
		
		if max_value[1] == "in":
			# indegree > outdegree, remove out-edges
			edges = [(max_node,o) for o in graph.neighbors(max_node)]
		else:
			# outdegree > indegree, remove in-edges
			edges = [(i,max_node) for i in graph.predecessors(max_node)]
		edges_to_be_removed += edges 

		sub_graphs = filter_big_scc(graph,edges_to_be_removed)
		
		if sub_graphs:
			for index,sub in enumerate(sub_graphs):
				sccs.append(sub)
		if not sccs:
			return


#strongly connnected component is gauranteed to be DAG

def get_big_sccs(g):
    self_loop_edges = nx.selfloop_edges(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    num_big_sccs = 0
    edges_to_be_removed = []
    big_sccs = []
    for sub in (g.subgraph(c).copy() for c in nx.strongly_connected_components(g)):
        number_of_nodes = sub.number_of_nodes()
        if number_of_nodes >= 2:
            # strongly connected components
            num_big_sccs += 1
            big_sccs.append(sub)
    #print(" # big sccs: %d" % (num_big_sccs))
    return big_sccs


def filter_big_scc(g,edges_to_be_removed):
	#Given a graph g and edges to be removed
	#Return a list of big scc subgraphs (# of nodes >= 2)
	g.remove_edges_from(edges_to_be_removed)
	sub_graphs = filter(lambda scc: scc.number_of_nodes() >= 2, [g.subgraph(c).copy() for c in nx.strongly_connected_components(g)])
	return sub_graphs


def remove_self_loops_from_graph(g):
	
	self_loops = list(nx.selfloop_edges(g))
	g.remove_edges_from(self_loops)
	return self_loops


def scc_nodes_edges(g):
	"""
	返回strongly connnected component的node 和 edges
	"""
	scc_nodes = set()
	scc_edges = set()
	num_big_sccs = 0
	num_nodes_biggest_scc = 0
	biggest_scc = None
	for sub in (g.subgraph(c).copy() for c in nx.strongly_connected_components(g)):
		number_nodes = sub.number_of_nodes()
		if number_nodes >= 2:
			scc_nodes.update(sub.nodes())
			scc_edges.update(sub.edges())
			num_big_sccs += 1
			if num_nodes_biggest_scc < number_nodes:
				num_nodes_biggest_scc = number_nodes
				biggest_scc = sub
	nonscc_nodes = set(g.nodes()) - scc_nodes
	nonscc_edges = set(g.edges()) - scc_edges
	print(num_nodes_biggest_scc)
	print("num of big sccs: %d" % num_big_sccs)
	if biggest_scc == None:
		return scc_nodes,scc_nodes,nonscc_nodes,nonscc_edges
	print("# nodes in biggest scc: %d, # edges in biggest scc: %d" % (biggest_scc.number_of_nodes(),biggest_scc.number_of_edges()))
	print("# nodes,edges in scc: (%d,%d), # nodes, edges in non-scc: (%d,%d) " % (len(scc_nodes),len(scc_edges),len(nonscc_nodes),len(nonscc_edges)))
	num_of_nodes = g.number_of_nodes()
	num_of_edges = g.number_of_edges()
	print("# nodes in graph: %d, # of edges in graph: %d, percentage nodes, edges in scc: (%0.4f,%0.4f), percentage nodes, edges in non-scc: (%0.4f,%0.4f)" % (num_of_nodes,num_of_edges,len(scc_nodes)*1.0/num_of_nodes,len(scc_edges)*1.0/num_of_edges,len(nonscc_nodes)*1.0/num_of_nodes,len(nonscc_edges)*1.0/num_of_edges))
	return scc_nodes,scc_edges,nonscc_nodes,nonscc_edges



def remove_cycle_edges_by_mfas(graph_file) -> list:
	"""
	返回造成cycle的edges
	"""

	g = nx.DiGraph(nx.nx_pydot.read_dot(graph_file))
	self_loops = remove_self_loops_from_graph(g)

	scc_nodes,_,_,_ = scc_nodes_edges(g)
	degree_dict = get_nodes_degree_dict(g,scc_nodes)

	sccs = get_big_sccs(g)
	if len(sccs) == 0:
		print("After removal of self loop edgs: %s" % nx.is_directed_acyclic_graph(g))
		return self_loops
	
	edges_to_be_removed = []
	
	#import timeit
	#t1 = timeit.default_timer()
	greedy_local_heuristic(sccs,degree_dict,edges_to_be_removed)
	#t2 = timeit.default_timer()
	#print("mfas time usage: %0.4f s" % (t2 - t1))
	

	edges_to_be_removed = list(set(edges_to_be_removed))
	g.remove_edges_from(edges_to_be_removed)
	edges_to_be_removed += self_loops
	#update:返回已进行反向的DAG
	edges_reversed = [(i[1], i[0]) for i in edges_to_be_removed]
	g.add_edges_from(edges_reversed)
	return edges_to_be_removed, g

#-----------------------------------------------------------------------以下是layer assignment------------------------------------------------------------------------------------

def LayerAssign(DAG):
	"""
	assign layers by height optimization
	"""

	DAG_in_degree  = list(DAG.in_degree)
	num_nodes = len(DAG.nodes)
	#以没有in的节点为初始source
	sources = [node[0] for node in DAG_in_degree if node[1] == 0]
	layer_idx = 1
	nodes_layers = []
	for source in sources:
		nodes_layers.append((source, layer_idx))
	while sources:
		layer_idx += 1
		subndoes_nested = [list(DAG.succ[source]) for source in sources]
		subnodes = list(set(item for sublist in subndoes_nested for item in sublist))
		#判断subnode是否有除了source以外的父节点，如果有，则其层数还要更大，此时要将其跳过
		subnodes_clean = subnodes.copy()
		for subnode in subnodes:
			if set(list(DAG.pred[subnode])) - set(sources):
				subnodes_clean.remove(subnode)
			else:
				nodes_layers.append((subnode, layer_idx))
		#从DAG中删除sources
		DAG.remove_nodes_from(sources)
		sources = subnodes_clean
	return nodes_layers, layer_idx - 1

def Add_dummy(DAG, nodes_layers: list):
	"""
	add dummy nodes between edges crossing more than 1 layer
	output: DAG with dummy nodes and dummy edges, dict with nodes and their layer
	"""
	DAG_edges = list(dag.edges)
	dict_nodes_layer = dict(nodes_layers)
	dict_edge_dummies = {}
	for source, subnode in DAG_edges:
		layer_diff = dict_nodes_layer[subnode] - dict_nodes_layer[source]
		startnode = source
		if layer_diff > 1:
			dict_edge_dummies[(source, subnode)] = []
			for dummy_layer in range(dict_nodes_layer[source] + 1, dict_nodes_layer[subnode]):
				dummy_node = f'd_{source}to{subnode}_{dummy_layer}'
				dict_nodes_layer[dummy_node] = dummy_layer
				DAG.add_node(dummy_node)
				DAG.add_edge(startnode, dummy_node)
				dict_edge_dummies[(source, subnode)].append(dummy_node)
				startnode = dummy_node
			DAG.add_edge(dummy_node, subnode)
			DAG.remove_edge(source, subnode)
			
	return DAG, dict_nodes_layer, dict_edge_dummies

#---------------------------------------------------------------------以下是crossings minimization------------------------------------------------------------------------------------

def InvertKeyValueList(input_dict):
    #反转字典，将值作为键，相关的键作为值（存储在列表中）
	inverted_dict = {}
	for key, value in input_dict.items():
		for v in value:
			if v in inverted_dict:
				inverted_dict[v].append(key)
			else:
				inverted_dict[v] = [key]
	return inverted_dict

def InvertKeyValue(input_dict):
	#反转字典，将值作为键，相关的键作为值（存储在列表中）
	inverted_dict = {}
	for key, value in input_dict.items():
		if value in inverted_dict:
			inverted_dict[value].append(key)
		else:
			inverted_dict[value] = [key]
	return inverted_dict

def FindBaryCenter(dag_with_dummy, fixed_layer_idx: int, dict_layer_nodes: dict):
	"""注:此算法未引入weight"""

	#第一步:找l2各点与l1相连的边
	fixed_layer = dict_layer_nodes[fixed_layer_idx]
	next_layer =  dict_layer_nodes[fixed_layer_idx + 1]
	#注意,直接指向原字典,直接改变字典中的顺序
	dict_l2_adjl1 = {}
	for node in fixed_layer:
		dict_l2_adjl1[node] = list(set(list(dag_with_dummy.adj[node])) & set(next_layer))
	dict_l2_adjl1 = InvertKeyValueList(dict_l2_adjl1)
	#第二步，找barycenter
	barycenter = []
	for l2node, adj_l1nodes in dict_l2_adjl1.items():
		#bary center = sum(order of adj l1 nodes) / num(adj l1 nodes) 若考虑权重分母应该是权重和
		barycenter.append(sum((fixed_layer.index(l1node) + 1) for l1node in adj_l1nodes) / len(adj_l1nodes))
	#第三步，根据barycenter排序l2
	dict_layer_nodes[fixed_layer_idx + 1] = [x for _,x in sorted(zip(barycenter, dict_l2_adjl1))]
	return dict_layer_nodes

def FindOrder_bc(dag_with_dummy, dict_node_layer: dict):
	dict_layer_nodes = InvertKeyValue(dict_node_layer)
	#固定一层找下一层,第一层随机
	for layer in range(1, len(dict_layer_nodes)):
		# print(f'{layer}:{dict_layer_nodes[layer]}')
		# print(f'{layer + 1}:{dict_layer_nodes[layer + 1]}')
		dict_layer_nodes = FindBaryCenter(dag_with_dummy.copy(), layer, dict_layer_nodes)
		# print(f'{layer}:{dict_layer_nodes[layer]}')
		# print(f'{layer + 1}:{dict_layer_nodes[layer + 1]}')
	#节点顺序已经存储在字典里，字典里的节点是已经排了序的
	return dict_layer_nodes

def SortByMedian(order: list, nodes: dict, degrees: list) -> dict:
	combined = list(zip(nodes, degrees, order))

	# Define a custom sorting key
	def sorting_key(item):
		_, degree, ord = item
		return ord, degree % 2 == 0, degree

	sorted_combined = sorted(combined, key = sorting_key)
	return [node for node, _, _ in sorted_combined]


def FindMedianCenter(dag_with_dummy, fixed_layer_idx: int, dict_layer_nodes: dict):
	"""注:此算法未引入weight"""

	#第一步:找l2各点与l1相连的边
	fixed_layer = dict_layer_nodes[fixed_layer_idx]
	next_layer =  dict_layer_nodes[fixed_layer_idx + 1]
	#注意,直接指向原字典,直接改变字典中的顺序
	dict_l2_adjl1 = {}
	for node in fixed_layer:
		dict_l2_adjl1[node] = list(set(list(dag_with_dummy.adj[node])) & set(next_layer))
	dict_l2_adjl1 = InvertKeyValueList(dict_l2_adjl1)
	#第二步，找median center
	mediancenter = []
	degree = []
	for l2node, adj_l1nodes in dict_l2_adjl1.items():
		#找中位数 若考虑权重degree应该是权重和
		mediancenter.append(np.median([fixed_layer.index(l1node) + 1 for l1node in adj_l1nodes]))
		degree.append(len(adj_l1nodes))
	#第三步，根据mediancenter排序l2
	dict_layer_nodes[fixed_layer_idx + 1] = SortByMedian(mediancenter, list(dict_l2_adjl1),degree)
	return dict_layer_nodes

def FindOrder_mc(dag_with_dummy, dict_node_layer: dict):
	dict_layer_nodes = InvertKeyValue(dict_node_layer)
	#固定一层找下一层,第一层随机
	for layer in range(1, len(dict_layer_nodes)):
		# print(f'{layer}:{dict_layer_nodes[layer]}')
		# print(f'{layer + 1}:{dict_layer_nodes[layer + 1]}')
		dict_layer_nodes = FindMedianCenter(dag_with_dummy.copy(), layer, dict_layer_nodes)
		# print(f'{layer}:{dict_layer_nodes[layer]}')
		# print(f'{layer + 1}:{dict_layer_nodes[layer + 1]}')
	#节点顺序已经存储在字典里，字典里的节点是已经排了序的
	return dict_layer_nodes

#-------------------------------------------------------------------------以下是coordinate assignment----------------------------------------------------------------

def AssignCoordinate(dag_with_dummy, dict_layer_nodes: dict) -> dict:
	#init
	dict_node_xy = {node: (nodes.index(node)*2, layer) for layer, nodes in dict_layer_nodes.items() for node in nodes}
	nx.set_node_attributes(dag_with_dummy, dict_node_xy, 'pos')
	node_pos = nx.get_node_attributes(dag_with_dummy, 'pos')
	nx.draw(dag_with_dummy, 
			pos = node_pos,
			with_labels=True,
			font_size = 8)
	plt.show()
	return dict_node_xy



def SteighteningEdges(dag_with_dummy, dict_node_xy: dict, dict_edge_dummies: dict, delta : float = 5.0) -> dict:
	
	def g(x_puv, x_u, x_v):
		k = len(x_puv) - 2
		a = [x_u + (i + 1) / (k + 1) * (x_v - x_u) for i in range(k)]
		return sum((x_puv[i + 1] - a[i]) ** 2 for i in range(k))
	#设置优化目标
	def objective_function(x, dict_edge_dummies, node_idx):
		total_cost = 0
		for (u, v), dummy_nodes in dict_edge_dummies.items():
			u_idx = node_idx.index(u)
			v_idx = node_idx.index(v)
			x_u = x[u_idx]
			x_v = x[v_idx] 
			x_puv = [x_u] + [x[node_idx.index(dummy_node)] for dummy_node in dummy_nodes] + [x_v]
			total_cost += g(x_puv, x_u, x_v)
		return total_cost

	initial_guess = [x for _,(x,_) in dict_node_xy.items()]
	node_idx      = list(dict_node_xy)
	#设置约束条件
	cons = []
	for z, (x_z, y_z) in dict_node_xy.items():
		for w, (x_w, y_w) in dict_node_xy.items():
			if y_w == y_z and x_w > x_z:
				cons.append({'type': 'ineq', 
				'fun': lambda x, w=w, z=z, delta=delta: x[node_idx.index(w)] - x[node_idx.index(z)] - delta})
	
	result = minimize(objective_function, initial_guess, args=(dict_edge_dummies, node_idx), constraints= cons)

	opt_x = result.x
	dict_node_xy_opt = {node: (x, dict_node_xy[node][1]) for node, x in zip(dict_node_xy, opt_x)}

	nx.set_node_attributes(dag_with_dummy, dict_node_xy_opt, 'pos')
	node_pos = nx.get_node_attributes(dag_with_dummy, 'pos')
	nx.draw(dag_with_dummy, 
			pos = node_pos,
			font_size = 8,
			with_labels = True)

	plt.show()

	return dict_node_xy_opt

#----------------------------------------------------------------------------曲线拟合--------------------------------------------------------------------------------
def bernstein_poly(i, n, t):
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(control_points, n_points = 1000):
    """
    计算贝塞尔曲线上的点
    """
    n = len(control_points) - 1
    curve = []
    for t in np.linspace(0, 1, n_points):
        x = 0
        y = 0
        for i, point in enumerate(control_points):
            x += point[0] * bernstein_poly(i, n, t)
            y += point[1] * bernstein_poly(i, n, t)
        curve.append([x, y])
    return np.array(curve)

def EdgeDrawing(G, dict_node_xy, dict_edge_dummies):
	"""用反向前的网络画图"""
	fig, ax = plt.subplots()
	ax.axis ('off')
	edges = list(G.edges)
	nodes = list(G.nodes)
	# plot dummy edges
	for (u,v) in dict_edge_dummies:
		points = []
		points.append([dict_node_xy[u][0],dict_node_xy[u][1]])
		for dummy in dict_edge_dummies[(u,v)]:
			points.append([dict_node_xy[dummy][0], dict_node_xy[dummy][1]])
		points.append([dict_node_xy[v][0],dict_node_xy[v][1]])
		curve_points = bezier_curve(points)
		ax.plot(curve_points[:,0],curve_points[:,1], '-', color = 'b', zorder = 3)
		ax.arrow(curve_points[1,0],curve_points[1,1],
				curve_points[0,0] - curve_points[1,0], 
				curve_points[0,1] - curve_points[1,1],
				color = 'b',
				head_width = 0.08,
				zorder = 3)
	#plot straight edge
	for (u,v) in edges:
		if (u,v) not in dict_edge_dummies and (v,u) not in dict_edge_dummies:
			# ax.plot(
			# 	[dict_node_xy[u][0], dict_node_xy[u][1]],
			# 	[dict_node_xy[v][0], dict_node_xy[v][1]],
			# 	color = 'black',
			# 	zorder = 2
			# )
			ax.arrow(
				dict_node_xy[u][0], dict_node_xy[u][1],
				dict_node_xy[v][0] - dict_node_xy[u][0],
				dict_node_xy[v][1] - dict_node_xy[u][1],
				color = 'b',
				head_width = 0.08,
				zorder = 3
			)
	#plot nodes and labels
	for node in nodes:
		ax.scatter(
			dict_node_xy[node][0], dict_node_xy[node][1],
			color = 'red',
			s = 250,
			zorder = 2
		)
		ax.text(
			dict_node_xy[node][0] - 0.1,
			dict_node_xy[node][1] + 0.1,
			s = node,
			fontsize = 9,
			zorder = 5,
			weight='bold')
	a = 0


#----------------------------------------------------------------------------主函数--------------------------------------------------------------------------------

if __name__ == "__main__":
	dot_file_path = r"noname.dot" 
	G = nx.DiGraph(nx.nx_pydot.read_dot(dot_file_path))
	G_before_reverse = G.copy()
	# nx.draw(G, with_labels = True)
	# plt.show()
	nodes = list(G.nodes())
	degreeRecord = get_nodes_degree_dict(G, nodes)
	#reslove cycles
	cycle_edges, dag = remove_cycle_edges_by_mfas(dot_file_path) # update:remove函数回传增加dag
	#layer assignment
	nodes_layers, num_layer = LayerAssign(dag.copy())
	dag_with_dummy, dict_node_layer, dict_edge_dummies = Add_dummy(dag.copy(), nodes_layers)
	dict_a = AssignCoordinate(dag_with_dummy.copy(),InvertKeyValue(dict_node_layer))
	# dict_layer_nodes_sortedbybc = FindOrder_bc(dag_with_dummy.copy(), dict_node_layer)
	# dict_node_xy = AssignCoordinate(dag_with_dummy.copy(), dict_layer_nodes_sortedbybc)
	dict_layer_nodes_sortedbymc = FindOrder_mc(dag_with_dummy.copy(), dict_node_layer)
	dict_node_xy = AssignCoordinate(dag_with_dummy.copy(), dict_layer_nodes_sortedbymc)
	# dict_node_xy_opt = SteighteningEdges(dag_with_dummy.copy(), dict_node_xy, dict_edge_dummies)
	EdgeDrawing(G_before_reverse.copy(), dict_node_xy, dict_edge_dummies)
	a=0






