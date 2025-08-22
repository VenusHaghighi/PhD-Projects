from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from collections import defaultdict
import random



def generate_edge_labels(graph, idx_train, idx_test, idx_valid):
    
    labels = graph.ndata['label']
    
    edge_type_list = ["p", "s", "v", "homo"]
    
    for e in edge_type_list:
    
    
          src, dst = graph.edges(etype = e)
          edge_labels = []
          edge_train_mask = []
          edge_valid_mask = []
          edge_test_mask = []
          
          
          for i, j in zip(src, dst):
              i = i.item()
              j = j.item()
              if labels[i] == labels[j]:
                  edge_labels.append(1)
              else:
                  edge_labels.append(-1)
              
              if i in idx_train and j in idx_train:
                  edge_train_mask.append(1)
              else:
                   edge_train_mask.append(0)
                   
              if i in idx_valid and j in idx_valid:
                  edge_valid_mask.append(1)
              else:
                   edge_valid_mask.append(0)
                   
              if i in idx_test and j in idx_test:
                  edge_test_mask.append(1)
              else:
                   edge_test_mask.append(0)
      
          
          edge_labels = torch.Tensor(edge_labels).long()
          edge_train_mask = torch.Tensor(edge_train_mask).bool()
          edge_valid_mask = torch.Tensor(edge_valid_mask).bool()
          edge_test_mask = torch.Tensor(edge_test_mask).bool()
          
          graph.edges[e].data['edge_labels'] = edge_labels
          graph.edges[e].data['edge_train_mask'] = edge_train_mask
          graph.edges[e].data['edge_test_mask'] = edge_test_mask
          graph.edges[e].data['edge_valid_mask'] = edge_valid_mask
    
    return graph


            
#dataset = FraudAmazonDataset()
#graph = dataset[0]
#            
#print(graph)
#
#feature = graph.ndata['feature']
#labels = graph.ndata['label']
#train_mask = graph.ndata['train_mask']
#test_mask = graph.ndata['test_mask']
#valid_mask = graph.ndata['val_mask']
#                
#src_rsr, dst_rsr = graph.edge_type_subgraph(['net_upu']).edges()
#src_rtr, dst_rtr = graph.edge_type_subgraph(['net_usu']).edges()
#src_rur, dst_rur = graph.edge_type_subgraph(['net_uvu']).edges()
#src_homo = torch.cat([src_rsr, src_rtr, src_rur])
#dst_homo = torch.cat([dst_rsr, dst_rtr, dst_rur])
#                
#amazon_graph = dgl.heterograph({('u', 'homo', 'u'): (src_homo, dst_homo), ('u', 'p', 'u'): (src_rsr, dst_rsr), ('u', 's', 'u'): (src_rtr, dst_rtr), ('u', 'v', 'u'): (src_rur, dst_rur)})
#print("amazon_graph:", amazon_graph)
#                
#amazon_graph.ndata['feature'] = feature
#amazon_graph.ndata['label'] = labels
#amazon_graph.ndata['train_mask'] = train_mask
#amazon_graph.ndata['test_mask'] = test_mask
#amazon_graph.ndata['valid_mask'] = valid_mask
#                
#print("amazon_graph ndata:", amazon_graph.ndata)
#print("amazon_graph edata:", amazon_graph.edata)               
#index = list(range(len(labels)))
#
#idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
#                                                                        train_size=0.4,
#                                                                        random_state=2, shuffle=True)
#idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
#                                                                        test_size=0.67,
#                                                                        random_state=2, shuffle=True)
#train_mask = torch.zeros([len(labels)]).bool()
#valid_mask = torch.zeros([len(labels)]).bool()
#test_mask = torch.zeros([len(labels)]).bool()
#
#train_mask[idx_train] = 1
#valid_mask[idx_valid] = 1
#test_mask[idx_test] = 1
#            
#print("start generating edge labels")
#graph = generate_edge_labels(amazon_graph, idx_train, idx_test, idx_valid) 
#print("finish edge labels!!!")                
#dgl.save_graphs('amazon.dgl', graph)
#                
#graph = dgl.load_graphs("amazon.dgl")[0][0]   


        
def get_adj_list(graph):

    adjacency_list = {}
    
    for node_id in range(graph.number_of_nodes()):
  
        neighbors = graph.successors(node_id)  
    
        neighbor_list = neighbors.tolist()
    
        adjacency_list[node_id] = neighbor_list
    
    return adjacency_list
    
    
    

def sparse_to_adjlist(sp_matrix):

	"""Transfer sparse matrix to adjacency list"""

	#add self loop
	homo_adj = sp_matrix + sp.eye(sp_matrix.shape[0])
	#creat adj_list
	adj_lists = defaultdict(set)
	edges = homo_adj.nonzero()
	
	for index, node in enumerate(edges[0]):
		adj_lists[node].add(edges[1][index])
		adj_lists[edges[1][index]].add(node)
	adj_lists = {keya:random.sample(adj_lists[keya],10) if len(adj_lists[keya])>=10 else adj_lists[keya] for i, keya in enumerate(adj_lists)}

	return adj_lists




def connection_domination(graph):
    
    """ determine which types of connections are dominated in a given neighborhood """
    src, dst = graph.edges()
    edgeType = graph.edata['edgeType']
    k = graph.number_of_nodes()
    h2_rate = []
    for i in range(0, k):
        temp  = torch.cat([src.view(-1,1), dst.view(-1,1), edgeType.view(-1,1)], dim = -1)
        temp = temp.type(torch.int64)
        #print("i:", i)
        mask = temp[:, 0] == i
        #print("mask:", mask)
        temp = temp[mask]
        #print("temp:", temp)
        result = temp[:, -1]
        #print("result:", result)
        homo_rate = np.count_nonzero(result == 1)
        hetero_rate = np.count_nonzero(result == -1)
        if (homo_rate > hetero_rate):
            rate = 1
        elif(hetero_rate >= homo_rate):
            rate = -1
        h2_rate.append(rate)
        
    #print("h2_rate:", h2_rate, len(h2_rate))
    h2_rate = torch.tensor(h2_rate)
    #print("h2_rate:", h2_rate, h2_rate.size())
    graph.ndata['h2_rate'] = h2_rate
    #dgl.save_graphs('amazon_completed.dgl', graph)
    return h2_rate, graph

