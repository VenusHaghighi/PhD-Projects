import matplotlib.pyplot as plt
import hypernetx as hnx
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from collections import defaultdict
import random
import networkx as nx
import csv as csv
from collections import namedtuple



class Generate_Different_Groups_of_Hyperedges:
    
    def __init__(self, graph_data: namedtuple):
        
        
        default_feat = torch.zeros(graph_data.feat_dim)
        
        self.default_feat = default_feat
        
        self.relations = list(graph_data.graph.etypes)
        self.features = graph_data.features
        self.labels = graph_data.labels
        self.n_classes = graph_data.n_classes

        
        self.graph = graph_data.graph
        
        
        self.train_mask = graph_data.train_mask
        self.val_mask = graph_data.val_mask
        self.test_mask = graph_data.test_mask
        
        self.train_nid = graph_data.train_nid
        self.val_nid = graph_data.val_nid
        self.test_nid = graph_data.test_nid
        
        
        self.train_nid_set = set(self.train_nid.tolist())
        self.val_nid_set = set(self.val_nid.tolist())
        self.test_nid_set = set(self.test_nid.tolist())
        
     
    def load_generated_hyperedges(self, graph):
         
         print("Start generating hyperedges based on khop (k=4) ...") 
        
         khop_neigh_dict_rel_1, khop_neigh_dict_rel_2, khop_neigh_dict_rel_3  = self.generate_hyperedges_with_Khop(graph, k=4)
         
         # save Khop neighbors hyperedges dictionaries for different relations
         torch.save(khop_neigh_dict_rel_1, "4_hop_hyperedges_UPU_Amazon.pt")
         torch.save(khop_neigh_dict_rel_2, "4_hop_hyperedges_USU_Amazon.pt")
         torch.save(khop_neigh_dict_rel_3, "4_hop_hyperedges_UVU_Amazon.pt")
         
         print("Start generating hyperedges based on kNN (k=5) ...") 
         
         KNN_dict = self.generate_hyperedges_with_KNN(graph)
         
         # save hyperdeges based on KNN
         torch.save(KNN_dict, "kNN_5_hyperedges_Amazon.pt")
         
         print("Start generating hyperedges based on one_hop ...") 
         
         one_hop_dict_rel_1, one_hop_dict_rel_2, one_hop_dict_rel_3 = self.generate_hyperedges_with_Khop(graph, k=1)
         torch.save(one_hop_dict_rel_1, "1_hop_hyperedges_UPU_Amazon.pt")
         torch.save(one_hop_dict_rel_2, "1_hop_hyperedges_USU_Amazon.pt")
         torch.save(one_hop_dict_rel_3, "1_hop_hyperedges_UVU_Amazon.pt")
         
         
         return khop_neigh_dict_rel_1, khop_neigh_dict_rel_2, khop_neigh_dict_rel_3, KNN_dict, one_hop_dict_rel_1, one_hop_dict_rel_2, one_hop_dict_rel_3
         
        
    def  calculateLength(self, a, b, spl):
     
        try:
            return spl[a][b]
        except KeyError:
            return 0 
        
            
    def calculate_khop_neighbors_matrix(self, graph, k):
    
        
            nx_graph = graph.to_networkx()
        
            # shortest path lenght for each pairs of nodes
        
            spl = dict(nx.all_pairs_shortest_path_length(nx_graph))
            
            
            unique_nodes = list(np.unique(graph.nodes()))
            
            num_nodes = len(unique_nodes)
            unique_nodes.sort()
            
            
            k_matrix = np.zeros((num_nodes, num_nodes))
                
            for i, row in enumerate(unique_nodes):
                 for j, col in enumerate(unique_nodes):
                        
                       length = self.calculateLength(row, col, spl)
                        
                       if length <= k and length != 0:
                                k_matrix[i, j] = 1
                       else:
                                k_matrix[i, j] = 0
            
            
            self_loop_matrix = np.eye(num_nodes)
            k_matrix = k_matrix + self_loop_matrix
            k_matrix = torch.tensor(k_matrix)
           
            
            return k_matrix   
    
    
    
    def calculate_khop_neighbors_matrix_BFS(self, graph, k):
          nx_graph = graph.to_networkx()
          
          num_nodes = len(nx_graph.nodes())
          khop_matrices = {}
          
          for i in range(1, k + 1):
              khop_matrix = np.zeros((num_nodes, num_nodes))
              
              for node in nx_graph.nodes():
                  neighbors = set(nx.bfs_tree(nx_graph, node, depth_limit=i).nodes()) - {node}
                  for neighbor in neighbors:
                      khop_matrix[node, neighbor] = 1
              
              khop_matrices[i] = torch.tensor(khop_matrix)
              
              self_loop_matrix = np.eye(num_nodes)
              khop_matrices[i] = khop_matrices[i]+ self_loop_matrix
              khop_matrices[i]= torch.tensor(khop_matrices[i])
          
          return khop_matrices[k]
          
    
    
    
    
    

    def generate_hyperedges_with_Khop(self, graph, k):
    
            relations = graph.etypes
            
            print("relations:", relations, relations[0], relations[1], relations[2])
            
            print("generate hyperedges with khop neighors for relation:"+ relations[0])
            
            sub_graph_1 = graph.edge_type_subgraph([relations[0]])
    
            k_matrix_relation_1 = self.calculate_khop_neighbors_matrix(sub_graph_1, k)
            
            nodes = sub_graph_1.nodes().tolist()
            
            k_hop_neighbors_relation_1 = {}
            
            
            for i, node_id in enumerate(nodes):
                    
                    # Get the row corresponding to the current node
                    row = k_matrix_relation_1[i]
                    
                    # Find the indices where the row has non-zero values (indicating neighbors)
                    
                    neighbor_indices = torch.nonzero(row, as_tuple=False).flatten()
            
                    # Convert the neighbor indices to neighbor node IDs
                    neighbor_node_ids = [nodes[idx.item()] for idx in neighbor_indices]
                   
                    # Store the neighbors in the dictionary
                    k_hop_neighbors_relation_1[node_id] = neighbor_node_ids
            
            
                 
            
            print("generate hyperedges with khop neighors for relation:"+ relations[1])
            sub_graph_2 = graph.edge_type_subgraph([relations[1]])
    
            k_matrix_relation_2 = self.calculate_khop_neighbors_matrix(sub_graph_2, k)
            
            nodes = sub_graph_2.nodes().tolist()
            
            k_hop_neighbors_relation_2 = {}
            
            for i, node_id in enumerate(nodes):
                    
                    # Get the row corresponding to the current node
                    row = k_matrix_relation_2[i]
                    
                    # Find the indices where the row has non-zero values (indicating neighbors)
                    
                    neighbor_indices = torch.nonzero(row, as_tuple=False).flatten()
            
                    # Convert the neighbor indices to neighbor node IDs
                    neighbor_node_ids = [nodes[idx.item()] for idx in neighbor_indices]
                   
                    # Store the neighbors in the dictionary
                    k_hop_neighbors_relation_2[node_id] = neighbor_node_ids
            
            
            
            print("generate hyperedges with khop neighors for relation:"+ relations[2])
            sub_graph_3 = graph.edge_type_subgraph([relations[2]])
    
            k_matrix_relation_3 = self.calculate_khop_neighbors_matrix(sub_graph_3, k)
            
            nodes = sub_graph_3.nodes().tolist()
            
            k_hop_neighbors_relation_3 = {}
            
            for i, node_id in enumerate(nodes):
                    
                    # Get the row corresponding to the current node
                    row = k_matrix_relation_3[i]
                    
                    # Find the indices where the row has non-zero values (indicating neighbors)
                    
                    neighbor_indices = torch.nonzero(row, as_tuple=False).flatten()
            
                    # Convert the neighbor indices to neighbor node IDs
                    neighbor_node_ids = [nodes[idx.item()] for idx in neighbor_indices]
                   
                    # Store the neighbors in the dictionary
                    k_hop_neighbors_relation_3[node_id] = neighbor_node_ids
            
            
            
            
            return k_hop_neighbors_relation_1, k_hop_neighbors_relation_2, k_hop_neighbors_relation_3  
        
    
    
    def Eu_dis(self, x):
        
            """
            Calculate the distance among each raw of x
            :param x: N X D
                        N: the object number
                        D: Dimension of the feature
            :return: N X N distance matrix
            """
            x = np.mat(x)
            aa = np.sum(np.multiply(x, x), 1)
            ab = x * x.T
            dist_mat = aa + aa.T - 2 * ab
            dist_mat[dist_mat < 0] = 0
            dist_mat = np.sqrt(dist_mat)
            dist_mat = np.maximum(dist_mat, dist_mat.T)
            return dist_mat
        
        
    def construct_hyperedges_with_KNN_from_distance(self, dis_mat, k):
            """
            construct hypregraph dictionary from hypergraph node distance matrix
            :param dis_mat: node distance matrix
            :param k: K nearest neighbor parameter
            :return: a dictionary which includes hyperedges
            """
            num_nodes= dis_mat.shape[0]
            dic ={}
            for idx in range(num_nodes):
                
                dis_vec = dis_mat[idx]
                nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
                
                # select K nearest neighbors for each individual node based on the pair-wise calculated Euclidian distance
                nearest_idx = nearest_idx.tolist()[0:k+1]
                # create hyperedges dictionary
                dic[idx]=nearest_idx
              
            return dic



    def generate_hyperedges_with_KNN(self, graph):
            """
            init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
            :param X: N_object x feature_number
            :param K_neigs: the number of neighbor expansion
            :param split_diff_scale: whether split hyperedge group at different neighbor scale
            :param is_probH: prob Vertex-Edge matrix or binary
            :param m_prob: prob
            :return: N_object x N_hyperedge
            """
            X = graph.ndata['feature']
            dis_mat = self.Eu_dis(X)
            
            dict_2 = self.construct_hyperedges_with_KNN_from_distance(dis_mat, k = 5)
        
            return dict_2
        
        
        
    def generate_hypergraph_pairwise(self, graph):
        
        relations = graph.etypes
        sub_graph_1 = graph.edge_type_subgraph([relations[0]])
        sub_graph_2 = graph.edge_type_subgraph([relations[1]])
        sub_graph_3 = graph.edge_type_subgraph([relations[2]])
         
        src_1, dst_1 = sub_graph_1.edges()
        pairwise_relations_1 = list(zip(src_1.numpy(), dst_1.numpy()))
        
        pairwise_dict_1 = {index: value for index, value in enumerate(pairwise_relations_1)}
        
        src_2, dst_2 = sub_graph_2.edges()
        pairwise_relations_2 = list(zip(src_2.numpy(), dst_2.numpy()))
        
        pairwise_dict_2 = {index: value for index, value in enumerate(pairwise_relations_2)}
        
        
        src_3, dst_3 = sub_graph_3.edges()
        pairwise_relations_3 = list(zip(src_3.numpy(), dst_3.numpy()))
        
        pairwise_dict_3 = {index: value for index, value in enumerate(pairwise_relations_3)}
        
        return pairwise_dict_1, pairwise_dict_2, pairwise_dict_3 