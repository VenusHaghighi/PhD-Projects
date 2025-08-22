import torch
import scipy.sparse as sp
import numpy as np
import os



def compute_structural_encoding(data, str_enc_dim):
    
    print("Structiral Encoding Dimension is:", str_enc_dim)
    nnodes = data.features.size()[0]
    edge_types = data.graph.etypes
    
    SE_dict = {}
    
    for e in edge_types:
        
        src, dst = data.graph.edges(etype = e)

        row = src.numpy()
        col = dst.numpy()
        ones_data = np.ones_like(row)
    
        A = sp.csr_matrix((ones_data, (row, col)), shape=(nnodes, nnodes))
        D = (np.array(A.sum(1)).squeeze()) ** -1.0
    
        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW
    
        SE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(str_enc_dim - 1):
            M_power = M_power * M
            SE.append(torch.from_numpy(M_power.diagonal()).float())
            
        SE = torch.stack(SE, dim=-1)
        SE_dict[e]=SE
        
    return SE_dict