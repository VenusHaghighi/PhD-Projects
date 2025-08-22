import argparse
import os
import numpy as np
import dgl
import torch
from tqdm import tqdm
import time
import dgl
import copy
import math
from collections import namedtuple
import data_preparation 
import torch.nn.functional as F
 

class Hypergraph_to_Sequence_Generator:
    def __init__(self, graph_data: namedtuple):
       
        default_feat = torch.zeros(graph_data.feat_dim)

        self.default_feat = default_feat
        
        self.graph = graph_data.graph
        self.relations = list(graph_data.graph.etypes)
        self.features = graph_data.features
        self.labels = graph_data.labels
        self.n_groups = graph_data.n_classes + 1


        
        self.train_nid = graph_data.train_nid
        self.val_nid = graph_data.val_nid
        self.test_nid = graph_data.test_nid
        self.train_nid_set = set(self.train_nid.tolist())
        self.val_nid_set = set(self.val_nid.tolist())
        self.test_nid_set = set(self.test_nid.tolist())
        
        
        
    def sequence_loader(self, nids: torch.Tensor, khop_neigh_dict_rel_1, khop_neigh_dict_rel_2, khop_neigh_dict_rel_3, one_hop_neigh_dict_rel_1, one_hop_neigh_dict_rel_2, one_hop_neigh_dict_rel_3, KNN_dict): 
     
        grp_feat_list = []
          
        for nid in tqdm(nids): 
           
            agg_feat = []
            nid = nid.item()
            print("node number:", nid)
            knn_nid = KNN_dict.get(nid)
            knn_agg_feat = self.knn_feat_aggregation(knn_nid)
            
            center_feat = self.features[nid]
            
            # get the khop neighbors list for a single node (nid) under different relations
            khop_neighbor_list_rel_1 = khop_neigh_dict_rel_1.get(nid)
            khop_neighbor_list_rel_2 = khop_neigh_dict_rel_2.get(nid)
            khop_neighbor_list_rel_3 = khop_neigh_dict_rel_3.get(nid) 
           
            
            # get the one_hop neighbors list for a single node (nid) under different relations
            one_hop_neighbor_list_rel_1 = one_hop_neigh_dict_rel_1.get(nid) 
            one_hop_neighbor_list_rel_2 = one_hop_neigh_dict_rel_2.get(nid)  
            one_hop_neighbor_list_rel_3 = one_hop_neigh_dict_rel_3.get(nid) 
            
       

            agg_feat.append(self.group_aggregation(nid, khop_neighbor_list_rel_1, one_hop_neighbor_list_rel_1))
             
            agg_feat.append(self.group_aggregation(nid, khop_neighbor_list_rel_2, one_hop_neighbor_list_rel_2))
            
            agg_feat.append(self.group_aggregation(nid, khop_neighbor_list_rel_3, one_hop_neighbor_list_rel_3))  
            
            
            # dimension:  (([n_relations*(2 * (n_classes))] + 1 + 1), E)
            # dimension:   ([number of relations * (Khop_based group aggregation + one_hop_based group aggregation)] + KNN-based aggregation + self_features) * E
            hop_agg_feat = torch.cat(agg_feat, dim=0)
            grp_feat = torch.cat([hop_agg_feat, knn_agg_feat.unsqueeze(0), center_feat.unsqueeze(0)], dim=0)
            
            grp_feat_list.append(grp_feat)
            
            
        final_seq_feats = torch.stack(grp_feat_list, dim=0)
        print("final shape of final_seq_feats",final_seq_feats.shape)  
        print("final output of sequence_loader",final_seq_feats)
        torch.save(final_seq_feats, "seq_features_Amazon_one_Plus_4_hop_5_knn.pt")
        
        return final_seq_feats  
            
            
            
            
            
    def group_aggregation(self, nid, khop_neighbor_list, one_hop_neighbor_list): 
         
         
         center_feat = self.features[nid]
         
         if len(khop_neighbor_list) == 1:
              
          
              khop_agg_feat = torch.stack([center_feat, center_feat], dim=0) 
         
         
         else:
         
              khop_nb_set = set(khop_neighbor_list)
              khop_pos_nid, khop_neg_nid = self.pos_neg_split(nid, khop_nb_set, self.features)
              h_0 = self.feat_aggregation(khop_neg_nid, center_feat)
           
              
              h_1 = self.feat_aggregation(khop_pos_nid, center_feat)
      
              
              khop_agg_feat = torch.stack([h_0, h_1], dim=0)
      
         
        
          
          
         if len(one_hop_neighbor_list) == 1:
              
              
              one_hop_agg_feat = torch.stack([center_feat, center_feat], dim=0) 
         
         
         else:
         
              one_hop_nb_set = set(one_hop_neighbor_list)
              one_hop_pos_nid, one_hop_neg_nid = self.pos_neg_split(nid, one_hop_nb_set, self.features)
              h_0 = self.feat_aggregation(one_hop_neg_nid, center_feat)
            
              
              h_1 = self.feat_aggregation(one_hop_pos_nid, center_feat)
             
              
              one_hop_agg_feat = torch.stack([h_0, h_1], dim=0)
           
         
         
         # generating the sequence of khop and one-hop aggregated features     
         feat_sequence = torch.cat([khop_agg_feat, one_hop_agg_feat], dim=0)
         return feat_sequence          
         
         
    def feat_aggregation(self, nids, center_feat):
    
        if len(nids) == 0:
        
            return center_feat
       
        else: 
        
            long_nids = torch.tensor(nids, dtype = torch.int64)
     
            feats = torch.index_select(self.features, dim=0, index=long_nids)
            
            feats = torch.mean(feats, dim=0)
    
            feats = feats * (1 / math.sqrt(long_nids.shape[0]))
            
    
            return feats
            
            
        
    def knn_feat_aggregation(self, nids):
    
        
            long_nids = torch.tensor(nids, dtype = torch.int64)
            
            feats = torch.index_select(self.features, dim=0, index=long_nids)
            
            feats = torch.mean(feats, dim=0)
    
            feats = feats * (1 / math.sqrt(long_nids.shape[0]))
            
    
            return feats 
            
               

    def pos_neg_split(self, nid, khop_nb_set, features):
    
        
        
        
        
        neighbors_features = {node_id: features[node_id] for node_id in khop_nb_set}
       
            
        # Calculate cosine similarity between the target node and selected nodes
        similarities = {}
        target_node_feature = features[nid]
      
            
        for node_id, features in neighbors_features.items():
            similarity = F.cosine_similarity(target_node_feature, features, dim=0)
            
            similarities[node_id] = similarity.item()
       
          
        # Find the most similar and dissimilar nodes
        similarity_threshold = 0.5
        pos_nids = [node_id for node_id, score in similarities.items() if score >= similarity_threshold]
      
        neg_nids = most_dissimilar_nodes = [node_id for node_id, score in similarities.items() if score < similarity_threshold]
      
        
        # returning a list of pos_nids and neg_nids 
        return pos_nids, neg_nids
        
        
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='data_preparation')
    parser.add_argument('--dataset', type=str, default='amazon',
                        help='Dataset name, [amazon, yelp]')

    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')

    parser.add_argument('--norm_feat', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--grp_norm', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--force_reload', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--add_self_loop', action='store_true', default=False,
                    help='add self-loop to all the nodes')

    parser.add_argument('--fanouts', type=int, default=[-1], nargs='+',
                        help='Sampling neighbors, default [-1] means full neighbors')


    parser.add_argument('--base_dir', type=str, default='~/.dgl',
                        help='Directory for loading graph data.')
    parser.add_argument('--save_dir', type=str, default='seq_data',
                        help='Directory for saving the processed sequence data.')
    
    args = vars(parser.parse_args())
    print(args)
    
    
    data, graph = data_preparation.prepare_data(args)
    print(data)
    
    nids = data.graph.nodes()
    
    
    ### load hyperedges
    
    khop_hyperedges_RSR_YelpChi = torch.load("4_hop_hyperedges_UPU_Amazon.pt")
    khop_hyperedges_RTR_YelpChi = torch.load("4_hop_hyperedges_USU_Amazon.pt")
    khop_hyperedges_RUR_YelpChi = torch.load("4_hop_hyperedges_UVU_Amazon.pt")
    one_hop_hyperedges_RSR_YelpChi = torch.load("1_hop_hyperedges_UPU_Amazon.pt")
    one_hop_hyperedges_RTR_YelpChi = torch.load("1_hop_hyperedges_USU_Amazon.pt")
    one_hop_hyperedges_RUR_YelpChi = torch.load("1_hop_hyperedges_UVU_Amazon.pt")
    kNN_hyperedges_YelpChi = torch.load("kNN_5_hyperedges_Amazon.pt")
    

    
    HTSG = Hypergraph_to_Sequence_Generator(data)
    final_seq_features = HTSG.sequence_loader(nids, khop_hyperedges_RSR_YelpChi, khop_hyperedges_RTR_YelpChi, khop_hyperedges_RUR_YelpChi, one_hop_hyperedges_RSR_YelpChi, one_hop_hyperedges_RTR_YelpChi, one_hop_hyperedges_RUR_YelpChi, kNN_hyperedges_YelpChi)
    print("final_seq_features:", final_seq_features, final_seq_features.shape)
  

