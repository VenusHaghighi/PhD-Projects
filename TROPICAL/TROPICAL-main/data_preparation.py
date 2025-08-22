import argparse
import pickle
import json
import dgl
import copy
import torch
import numpy as np
import math
from collections import namedtuple
from sklearn.cluster import KMeans
from tqdm import tqdm

from data import fraud_dataset, data_helper
from hypergraph_generation import generate_hyperedges


def load_data(dataset_name='yelp', raw_dir='~/.dgl/', train_size=0.4, val_size=0.1,
               seed=717, norm=True, force_reload=False, verbose=True) -> dict:
    """Loading dataset from dgl's FraudDataset.
    """
    if dataset_name in ['amazon', 'yelp']:
        fraud_data = fraud_dataset.FraudDataset(dataset_name, train_size=train_size, val_size=val_size,
                                                random_seed=seed, force_reload=force_reload)
    

    g = fraud_data[0]

    # Feature tensor dtpye is float64, change it to float32
    if norm:
        h = data_helper.row_normalize(g.ndata['feature'], dtype=np.float32)
        g.ndata['feature'] = torch.from_numpy(h)
    else:
        g.ndata['feature'] = g.ndata['feature'].float()

    # label shape is (n,1), reshape it to be (n, )
    # labels = g.ndata['label'].squeeze().long()
    # g.ndata['label'] = labels

    # graphs = {}
    # for etype in g.etypes:
    #     graphs[etype] = g.edge_type_subgraph([etype])
    #
    # g_homo = dgl.to_homogeneous(g)
    # graphs['homo'] = dgl.to_simple(g_homo)
    # for key, value in g.ndata.items():
    #     graphs['homo'].ndata[key] = value

    return g


def prepare_data(args, add_self_loop=False):
    g = load_data(dataset_name=args['dataset'], raw_dir=args['base_dir'],
                   train_size=args['train_size'], val_size=args['val_size'],
                   seed=args['seed'], norm=args['norm_feat'],
                   force_reload=args['force_reload'])
   
    relations = list(g.etypes)
    if add_self_loop is True:
        for etype in relations:
            g = dgl.remove_self_loop(g, etype=etype)
            g = dgl.add_self_loop(g, etype=etype)
        
        print('add self-loop for ', g)
    
    
    # Processing mask
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    train_nid = torch.nonzero(train_mask, as_tuple=True)[0]
    val_nid = torch.nonzero(val_mask, as_tuple=True)[0]
    test_nid = torch.nonzero(test_mask, as_tuple=True)[0]

    # Processing features and labels
    n_classes = 2
    n_relations = len(g.etypes)
    features = g.ndata['feature']
    feat_dim = features.shape[1]
    labels = g.ndata['label'].squeeze().long()

    print(f"[Global] Dataset <{args['dataset']}> Overview\n"
          f"\tEntire (postive/total) {torch.sum(labels):>6} / {labels.shape[0]:<6}\n"
          f"\tTrain  (postive/total) {torch.sum(labels[train_nid]):>6} / {labels[train_nid].shape[0]:<6}\n"
          f"\tValid  (postive/total) {torch.sum(labels[val_nid]):>6} / {labels[val_nid].shape[0]:<6}\n"
          f"\tTest   (postive/total) {torch.sum(labels[test_nid]):>6} / {labels[test_nid].shape[0]:<6}\n")

    Datatype = namedtuple('GraphData', ['graph', 'features', 'labels','train_mask', 'train_nid', 'val_mask', 'val_nid', 'test_mask',
                                        'test_nid', 'n_classes', 'feat_dim', 'n_relations'])
    graph_data = Datatype(graph = g, features=features, labels=labels, train_mask=train_mask, train_nid=train_nid,
                          val_mask=val_mask, val_nid=val_nid, test_mask=test_mask, test_nid=test_nid, n_classes=n_classes,
                          feat_dim=feat_dim, n_relations=n_relations)


    
    return graph_data, g





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
    
    
    data, graph = prepare_data(args)
    print(data)
    
  
    generated_hyperedges = generate_hyperedges.Generate_Different_Groups_of_Hyperedges(data)
    khop_neigh_dict_rel_1, khop_neigh_dict_rel_2, khop_neigh_dict_rel_3, KNN_dict, one_hop_dict_rel_1, one_hop_dict_rel_2, one_hop_dict_rel_3 = generated_hyperedges.load_generated_hyperedges(graph)
    
   
