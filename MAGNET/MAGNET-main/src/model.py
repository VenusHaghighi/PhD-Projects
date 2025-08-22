import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv
import os
from dgl import DGLGraph
import sympy
import scipy
os.environ['DGLBACKEND'] = 'pytorch'



class Edge_Labelling(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super().__init__()
        self.Linear = nn.Linear(in_channel, out_channel)
        self.FLinear = nn.Linear(2*out_channel, 1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        edges = data.edges()
        src,dst = edges
    
        f_src = data.ndata['feature'][src.numpy()]
        f_dst = data.ndata['feature'][dst.numpy()]
    
        src = self.Linear(f_src)
        dst = self.Linear(f_dst)
        
        eFeats = torch.cat([src - dst, src + dst], dim=1)
        eFeats = self.dropout(eFeats)
        
        out = self.FLinear(eFeats).squeeze()
        out = self.tanh(out)
        
        return out





class MAGNET_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, d, graph):
        
        super(MAGNET_Layer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d = d
        self.num_LP = int(((self.d)+1)/2)
        self.num_HP = (self.d+1)-self.num_LP
        self.graph = graph
        self.activation = F.leaky_relu
        self.linear = nn.Linear(in_channels, out_channels)
        self.M_LP = nn.Linear(self.num_LP*self.out_channels, out_channels)
        self.M_HP = nn.Linear(self.num_HP*self.out_channels, out_channels)
        
    def forward(self, graph, feat, d):
        
        
        def Laplacian(graph, feat):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            
            #print("feat:", feat, feat.size())
            D_invsqrt = torch.pow((graph.in_degrees(v='__ALL__')+graph.out_degrees(u='__ALL__')).float().clamp(min=1), -0.5).unsqueeze(-1).to(feat.device)
            #print("D_invsqrt:", D_invsqrt, D_invsqrt.size())
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt
        
        
        def calculate_theta(d):
               thetas = []
               index = int((d+1)/2)
               x = sympy.symbols('x')
               for i in range(d+1):
                   f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
                   coeff = f.all_coeffs()
                   inv_coeff = []
                   for i in range(d+1):
                      inv_coeff.append(float(coeff[d-i]))
                   thetas.append(inv_coeff)
                   # split high pass and Low pass Filters
               LP_filter = thetas[0:index]
               
               #print("LP filter:", LP_filter, len(LP_filter))
               HP_filter = thetas[index:d+1]
               return thetas, LP_filter, HP_filter
           
            
        def homo_aggr(feat, LP_filter, graph):
            
            
            out = []
            
            feat = feat[:, 0:-1]
            for i in range (len(LP_filter)):
                k = len(LP_filter[i])
                h = LP_filter[i][0]*feat
                
                
                for j in range(1, k):
                   temp = Laplacian(graph, feat)
                   h += LP_filter[i][j] * temp
                out.append(h)
                
            out = torch.cat(out, -1)
           
            out = self.M_LP(out)
            return out
            
#        def homo_aggr(feat, LP_filter, graph):
#            
#            
#            out = []
#            feat = feat[:, 0:-1]
#            
#            i = 1
#            k = 1
#            h = LP_filter[i]*feat
#                
#                
#                
#            temp = Laplacian(graph, feat)
#            h = LP_filter[i] * temp
#            out.append(h)
#                
#            out = torch.cat(out, -1)
#           
#            out = self.M_LP(out)
#            return out
                
        
            
        def hetero_aggr(feat, HP_filter, graph):
           
           out = []
           feat = feat[:, 0:-1]
           for i in range (len(HP_filter)):
               k = len(HP_filter[i])
               h = HP_filter[i][0]*feat
               for j in range(1, k):
                  temp = Laplacian(graph, feat)
                  h += HP_filter[i][j] * temp
               out.append(h)
           out = torch.cat(out, -1)
           #print("out:", out, out.size())
           out = self.M_HP(out)
           return out
           
           
           

        with graph.local_scope():
            
            h2_rate = graph.ndata["h2_rate"] 
            h2_rate = h2_rate.view(-1,1)
            feat = self.linear(feat)
            feat = torch.cat([feat, h2_rate], dim = -1)
            graph.ndata["h"] = feat
            #edgeType = graph.edges['homo'].data['edgeType']
            thetas, LP_filter, HP_filter = calculate_theta(d)
            #print("thetas:", thetas, len(thetas))
            #print("LP_filter:", LP_filter, len(LP_filter))
            #print("HP_filter:", HP_filter, len(HP_filter))
            Condition_1 = feat[:, -1] == 1
            Condition_1 =  Condition_1.reshape(feat.size()[0], 1)
            # Condition_2 = feat[:, -1] == -1
            # Condition_2 =  Condition_2.reshape(feat.size()[0], 1)
            final_result = torch.where(Condition_1, homo_aggr(feat, LP_filter, graph) , hetero_aggr(feat, HP_filter, graph))
            graph.ndata["h"] = final_result
            return final_result
            
 

class MAGNET(nn.Module):
    def __init__(self, in_channels, h_channels, num_classes, graph, d):
        super().__init__()
        self.graph = graph
        self.in_channels = in_channels
        self.h_channels = h_channels
        self.num_classes = num_classes
        self.linear = nn.Linear(in_channels, h_channels)
        self.linear2 = nn.Linear(h_channels, h_channels)
        self.linear4 = nn.Linear(h_channels, num_classes)
        self.act = nn.ReLU()
        self.d = d
        self.conv1 = MAGNET_Layer(self.in_channels, self.h_channels, self.d, self.graph)
        self.conv2 = MAGNET_Layer(self.h_channels, self.num_classes, self.d, self.graph)

    def forward(self, graph, feat, d):
        #h = self.linear(feat)
        #h = self.act(h)
        #h = self.linear2(h)
        #h = self.act(h)
        h = self.conv1(graph, feat, d)
        h = self.act(h)
        h = self.conv2(graph, h, d)
        return torch.sigmoid(h),h


    


