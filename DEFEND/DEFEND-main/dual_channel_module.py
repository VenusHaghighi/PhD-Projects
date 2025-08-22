from torch.nn import Sequential, Linear, ReLU
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import sympy
import scipy
from dgl.nn import EdgeWeightNorm
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


EOS = 1e-10



def MR_Combination_Module(data, embedding_dict, linear_c):
   
    etypes = data.graph.etypes
    
    # get embeddings of different relations

    emb_R1 = embedding_dict[etypes[0]]
    emb_R2 = embedding_dict[etypes[1]]
    emb_R3 = embedding_dict[etypes[2]]
    
    combine_emb = torch.cat([emb_R1, emb_R2, emb_R3], dim=-1)
    
    part_1 = combine_emb[:, [0,2,4]]
    part_2 = combine_emb[:, [1,3,5]]
    
    part_1_proj = linear_c(part_1)
    part_2_proj = linear_c(part_2)
    
    final_emb = torch.cat([part_1_proj, part_2_proj], dim=-1)
    
    
    return final_emb
    
    

def get_adj_from_edges(edges, weights, nnodes):
    adj = torch.zeros(nnodes, nnodes)
    adj[edges[0], edges[1]] = weights
    return adj



class Dual_Channel_Module(nn.Module):
    def __init__(self, data, n_layers, in_dim, hid_dim, out_dim, proj_dim, dropout, coeff):
        super(Dual_Channel_Module, self).__init__()
        
        self.data = data
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.proj_dim = proj_dim
        self.dropout = dropout
        self.coeff = coeff
        
        self.linear = nn.Linear(self.in_dim, self.proj_dim)
        self.encoder_hm = DC_layer(self.n_layers, self.proj_dim, self.hid_dim, self.out_dim, self.dropout)
        self.encoder_ht = DC_layer(self.n_layers, self.proj_dim, self.hid_dim, self.out_dim, self.dropout)
        
        self.linear_c = nn.Linear(3,1, dtype=torch.float32)
        

       



    def forward(self, data, weights_dict, coeff):
        features = data.features
        x = self.linear(features)
        x = torch.relu(x)
        
        etypes = data.graph.etypes
        emb = {}
        
        for e in etypes:
            adj_lp, adj_hp, weights_lp, weights_hp = weights_dict[e]
            emb1 = self.encoder_hm(data, e, x, weights_lp, coeff, flag = 1)
            emb2 = self.encoder_ht(data, e, x, weights_hp, coeff, flag = 0)
            emb[e] = torch.cat((emb1, emb2), dim=1)
        
        
        final_emb = MR_Combination_Module(data, emb, self.linear_c)
       
            
        return final_emb
        

        
        
        


# dual channel aggregation layer

class DC_layer(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, out_dim, dropout):
        super(DC_layer, self).__init__()
        
        self.dropout = dropout
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.act = nn.ReLU()
        
        

    def forward(self, data, e, feat, weighted_adj, coeff_degree, flag):
    
    
       def Laplacian(data, e, feat, weighted_adj):
       
           nnodes = data.graph.nodes().size()[0]
           edges = data.graph.edges(etype = e)
           
           adj = get_adj_from_edges(edges, weighted_adj, nnodes)
           adj += torch.eye(nnodes)
          
           inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
           
           norm_adj = inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
           
           Laplacian = torch.eye(nnodes) - norm_adj
           
           return Laplacian
           
           
           
           
       def compute_coeff(coeff_degree):
        
           thetas = []
           x = sympy.symbols('x')
           d = coeff_degree
           index = int((d+1)/2)
               
           for i in range(d+1):
               
               f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
               coeff = f.all_coeffs()
               inv_coeff = []
               for i in range(d+1):
                   inv_coeff.append(float(coeff[d-i]))
               thetas.append(inv_coeff)
                   
               
           return thetas
               
               
             
       
       
       def generate_beta_filters(data, Laplacian, thetas, coeff_degree):
        
           filters = []
            
           nnodes = data.graph.nodes().size()[0]
            
           for i in range(len(thetas)):
            
                k = len(thetas[i])
                temp = torch.zeros(nnodes, nnodes)
                
                for j in range(1, k):
                
                    temp += thetas[i][j] * Laplacian
                    
                filters.append(temp)    
    
           return filters
        
        
        
       def separate_filters(filters):
        
           d = len(filters)
        
           index = int((d+1)/2)
            
           LP_filters = filters[0:index]
           HP_filters = filters[index:d+1]
        
           return LP_filters, HP_filters
        
    

            
            
        
            
            
       with data.graph.local_scope(): 
       
            Laplacian = Laplacian(data, e, feat, weighted_adj)
            thetas = compute_coeff(coeff_degree)
            filters = generate_beta_filters(data, Laplacian, thetas, coeff_degree)  
            LP_filters, HP_filters = separate_filters(filters)
       
       
            for _ in range(self.n_layers):
                 
                 if flag == 1:
                 
                    k = len(LP_filters)
                    
                    for i in range(1, k):
                 
                        x = torch.matmul(LP_filters[i], feat)
                        x = self.act(x)
                     
                 else:
                 
                     k = len(HP_filters)
                    
                     for i in range(1, k):
                 
                         x = torch.matmul(HP_filters[i], feat)
                         x = self.act(x)
             
                 
            return x
       
       
       
       
       
       
       
       
       
       