from torch.nn import Sequential, Linear, ReLU
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import numpy as np
from dgl.nn import EdgeWeightNorm
import random
import dgl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


EOS = 1e-10
norm = EdgeWeightNorm(norm='both')

def get_adj_from_edges(edges, weights, nnodes):
    adj = torch.zeros(nnodes, nnodes)
    adj[edges[0], edges[1]] = weights
    return adj



def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges.cuda()



def normalize_adj(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())





class Edge_Discriminator(nn.Module):
    def __init__(self, nnodes, input_dim, alpha, sparse, hidden_dim=128, temperature=1.0, bias=0.0 + 0.0001):
        super(Edge_Discriminator, self).__init__()

        self.embedding_layers = nn.ModuleList()
        self.embedding_layers.append(nn.Linear(input_dim, hidden_dim))
        self.edge_mlp = nn.Linear(hidden_dim * 2, 1)

        self.temperature = temperature
        self.bias = bias
        self.nnodes = nnodes
        self.sparse = sparse
        self.alpha = alpha


    def get_node_embedding(self, h):
        for layer in self.embedding_layers:
            h = layer(h)
            h = F.relu(h)
        return h


    def get_edge_weight(self, embeddings, edges):
        s1 = self.edge_mlp(torch.cat((embeddings[edges[0]], embeddings[edges[1]]), dim=1)).flatten()
        s2 = self.edge_mlp(torch.cat((embeddings[edges[1]], embeddings[edges[0]]), dim=1)).flatten()
        return (s1 + s2) / 2


    def gumbel_sampling(self, edges_weights_raw):
        eps = (self.bias - (1 - self.bias)) * torch.rand(edges_weights_raw.size()) + (1 - self.bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs
        gate_inputs = (gate_inputs + edges_weights_raw) / self.temperature
        return torch.sigmoid(gate_inputs).squeeze()


    def weight_forward(self, features, edges):
        embeddings = self.get_node_embedding(features)
        edges_weights_raw = self.get_edge_weight(embeddings, edges)
        weights_lp = self.gumbel_sampling(edges_weights_raw)
        weights_hp = 1 - weights_lp
        return weights_lp, weights_hp


    def weight_to_adj(self, edges, weights_lp, weights_hp):
        if not self.sparse:
            adj_lp = get_adj_from_edges(edges, weights_lp, self.nnodes)
            adj_lp += torch.eye(self.nnodes)
            adj_lp = normalize_adj(adj_lp, 'sym', self.sparse)

            adj_hp = get_adj_from_edges(edges, weights_hp, self.nnodes)
            adj_hp += torch.eye(self.nnodes)
            adj_hp = normalize_adj(adj_hp, 'sym', self.sparse)

            mask = torch.zeros(adj_lp.shape)
            mask[edges[0], edges[1]] = 1.
            mask.requires_grad = False
            adj_hp = torch.eye(self.nnodes) - adj_hp * mask * self.alpha
        else:
            adj_lp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes)
            adj_lp = dgl.add_self_loop(adj_lp)
            weights_lp = torch.cat((weights_lp, torch.ones(self.nnodes))) + EOS
            weights_lp = norm(adj_lp, weights_lp)
            adj_lp.edata['w'] = weights_lp

            adj_hp = dgl.graph((edges[0], edges[1]), num_nodes=self.nnodes)
            adj_hp = dgl.add_self_loop(adj_hp)
            weights_hp = torch.cat((weights_hp, torch.ones(self.nnodes))) + EOS
            weights_hp = norm(adj_hp, weights_hp)
            weights_hp *= - self.alpha
            weights_hp[edges.shape[1]:] = 1
            adj_hp.edata['w'] = weights_hp
        return adj_lp, adj_hp


    def forward(self, data, features, struc_encoding):
        
        etypes = data.graph.etypes
        weight_dict = {}
        for e in etypes:
            edges = data.graph.edges(etype = e)
            featse = torch.cat((features, struc_encoding[e]), 1)
            weights_lp, weights_hp = self.weight_forward(featse, edges)
            adj_lp, adj_hp = self.weight_to_adj(edges, weights_lp, weights_hp)
            weight_dict[e]= adj_lp, adj_hp, weights_lp, weights_hp
        return weight_dict


