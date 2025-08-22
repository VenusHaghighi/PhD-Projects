import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time

from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import model
from model import *
import positional_encoding
import data_preparation
import hypergraph_to_seq





### training and testing model
def train_model(model, data, seq_data, args):
    
        labels = labels = data.labels
        index = list(range(len(labels)))
        if args["dataset"] == 'amazon':
            index = list(range(3305, len(labels)))

        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index], train_size=args['train_size'], random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest, test_size=0.67, random_state=2, shuffle=True)
        train_mask = torch.zeros([len(labels)]).bool()
        val_mask = torch.zeros([len(labels)]).bool()
        test_mask = torch.zeros([len(labels)]).bool()

        train_mask[idx_train] = 1
        val_mask[idx_valid] = 1
        test_mask[idx_test] = 1
        print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
        best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

        weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
        print('cross entropy weight: ', weight)
        time_start = time.time()
        
        
        
        for e in range(1000):
            
            model.train()
            final_embed = model(seq_data)
            logits = torch.sigmoid(final_embed)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            probs = logits.softmax(1)
            f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
            preds = numpy.zeros_like(labels)
            preds[probs[:, 1] > thres] = 1
            trec = recall_score(labels[test_mask], preds[test_mask])
            tpre = precision_score(labels[test_mask], preds[test_mask], zero_division=1)
            tmf1 = f1_score(labels[test_mask], preds[test_mask], average='macro')
            tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

            if best_f1 < f1:
                best_f1 = f1
                final_trec = trec
                final_tpre = tpre
                final_tmf1 = tmf1
                final_tauc = tauc
            print('Epoch {}, loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(e, loss, f1, best_f1))

        time_end = time.time()
        print('time cost: ', time_end - time_start, 's')
        print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100,
                                                                         final_tpre*100, final_tmf1*100, final_tauc*100))
        return final_tmf1, final_tauc, probs, preds, final_embed, test_mask
    
    
    

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in numpy.linspace(0.05, 0.95, 19):
        preds = numpy.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TROPICAL-Fraud Detection Model')
    
    parser.add_argument("--dataset", type=str, default="yelp",
                        help="Dataset for this model (yelp/amazon)")
                         
    
    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
                        
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
                        
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')
                        
    parser.add_argument('--norm_feat', action='store_true', default=False,
                        help='Using group norm, default False')
                        
    parser.add_argument('--force_reload', action='store_true', default=False,
                        help='Using group norm, default False')                    

    parser.add_argument('--n_layers', type=int, default=3,
                        help='number of transformer encoderr layers. for Amazon:2 and for yelp:3')
                        
    parser.add_argument('--n_heads', type=int, default=4,
                        help='number of heads in multi head attention')
                        
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='all learnable encodings dimension. for Amazon: 16 and for yelp: 64')
                        
    parser.add_argument('--dim_feedforward', type=int, default=128,
                       help='feed forward layer dimension. for Amazon: 64 and for yelp: 128')

    parser.add_argument('--n_hops', type=int, default=2,
                         help='n_hops encoding strategy')
                         
    parser.add_argument('--n_feat', type=int, default=2,
                         help='feature encoding strategy')                     
                         
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate. for amazon: 0.0 and for yelp: 0.1')
                        
    parser.add_argument('--agg_type', type=str, default="cat",
                        help='aggregation type')                    

    parser.add_argument('--base_dir', type=str, default='~/.dgl',
                        help='Directory for loading graph data.')
  
    args = vars(parser.parse_args())
    print(args)  
    
    
    data, graph = data_preparation.prepare_data(args)
    print("Data is loading ...", data)
    print("Graph is loading ...", graph)
    
    
    graph = data.graph
    n_relations = data.n_relations
    features = data.features
    labels = data.labels
    n_classes = data.n_classes
    n_groups = data.n_classes 
    feat_dim = data.feat_dim
    emb_dim = args['emb_dim']
    n_hops = args['n_hops']
    dim_feedforward = args['dim_feedforward']
    n_heads = args['n_heads']
    n_layers = args['n_layers']
    n_feat = args['n_feat']
    dropout= args['dropout']
    agg_type=args['agg_type']

        
    train_nid = data.train_nid
    val_nid = data.val_nid
    test_nid = data.test_nid
    
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    # loading generated sequential data
    
    if args['dataset'] == "yelp":
    
    
          seq_data = torch.load("sequential_data/seq_features_YelpChi.pt") 
             
        
    else:
    
          seq_data = torch.load("sequential_data/seq_features_Amazon.pt") 

    
        
    model = TransformerNet(feat_dim, emb_dim, n_classes, n_hops, n_relations, n_heads, dim_feedforward, n_layers, n_feat, dropout, agg_type)
        
    final_tmf1, final_tauc, probs, preds, final_embed, test_mask = train_model(model, data, seq_data, args)
        
    
   