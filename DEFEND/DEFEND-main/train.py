import argparse
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import random

from load_data import *
from preprocessing import *
from edge_discriminator import *
from dual_channel_module import *

from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix





def generate_random_node_pairs(nnodes, nedges, backup=300):
    rand_edges = np.random.choice(nnodes, size=(nedges + backup) * 2, replace=True)
    rand_edges = rand_edges.reshape((2, nedges + backup))
    rand_edges = torch.from_numpy(rand_edges)
    rand_edges = rand_edges[:, rand_edges[0,:] != rand_edges[1,:]]
    rand_edges = rand_edges[:, 0: nedges]
    return rand_edges




def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre





def eval_mode(logits, labels, train_mask, val_mask, test_mask, epoch, DC_loss, rank_loss, best_f1):


    probs = logits.softmax(1)
    
    
    f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
    preds = np.zeros_like(labels)
    preds[probs[:, 1] > thres] = 1
    
    test_mask_np = test_mask.numpy().astype(bool)
    
   
    
    #probs = probs.detach().numpy()
    #labels = labels.detach().numpy()
    #test_mask = test_mask.numpy()
 
    trec = recall_score(labels[test_mask], preds[test_mask_np])
    tpre = precision_score(labels[test_mask], preds[test_mask_np], zero_division=1)
    tmf1 = f1_score(labels[test_mask], preds[test_mask_np], average='macro')
    tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())

    if best_f1 < f1:
       best_f1 = f1
       final_trec = trec
       final_tpre = tpre
       final_tmf1 = tmf1
       final_tauc = tauc
       
    print('Epoch {}, DC_loss: {:.4f}, val mf1: {:.4f}, (best {:.4f})'.format(epoch, DC_loss, f1, best_f1))
    
    

    return final_trec, final_tpre, final_tmf1, final_tauc
    
    
    
    
    
def train_DC_Module(DC_Module, discriminator, optimizer_DC_Module, data, features, struc_encoding, args):

    DC_Module.train()
    discriminator.eval()
    
    weight_dict = discriminator(data, features, struc_encoding)
    embedding = DC_Module(data, weight_dict, args["coeff"])
    
    
    logits = torch.sigmoid(embedding)
    
    labels = data.labels
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()
    
    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    #print('cross entropy weight: ', weight)
    
    loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
    
    optimizer_DC_Module.zero_grad()
    loss.backward()
    optimizer_DC_Module.step()
    
    return loss, logits
       
    


def train_disc(DC_Module, discriminator, optimizer_discriminator, data, features, struc_encoding, args, epoch):

    DC_Module.eval()
    discriminator.train()
    
    nnodes = data.features.size()[0]
    nfeats = data.features.size()[1]
    edge_types = data.graph.etypes
    se_dim = struc_encoding[edge_types[0]].shape[1]
    
    
    for e in edge_types:
    
        #print("training phase starts for discriminator ...", e)
    
        
        discriminator.train()
        weight_dict = discriminator(data, features, struc_encoding)
            
        rand_node_pairs = {}
        loss = {}
            
        adj_lp, adj_hp, weights_lp, weights_hp = weight_dict[e]
        #print("adj_lp:", adj_lp, adj_lp.size())
        #print("adj_hp:", adj_hp, adj_hp.size())
        #print("weights_lp:", weights_lp, weights_lp.size())
        #print("weights_hp:", weights_hp, weights_hp.size())
        nedges = data.graph.num_edges(etype = e)
        rand_np = generate_random_node_pairs(nnodes, nedges)
        rand_node_pairs[e] =  rand_np
        psu_label = torch.ones(nedges)
                
        edge_emb_sim = F.cosine_similarity(data.features[data.graph.edges(etype = e)[0]], data.features[data.graph.edges(etype = e)[1]])
                
                
        rnp_emb_sim_lp = F.cosine_similarity(data.features[rand_np[0]], data.features[rand_np[1]])
        loss_lp = F.margin_ranking_loss(edge_emb_sim, rnp_emb_sim_lp, psu_label, margin=args['margin_hom'], reduction='none')
        loss_lp *= torch.relu(weights_lp - 0.5)
    
        rnp_emb_sim_hp = F.cosine_similarity(data.features[rand_np[0]], data.features[rand_np[1]])
        loss_hp = F.margin_ranking_loss(rnp_emb_sim_hp, edge_emb_sim, psu_label, margin=args['margin_het'], reduction='none')
        loss_hp *= torch.relu(weights_hp - 0.5)
    
        rank_loss = (loss_lp.mean() + loss_hp.mean()) / 2
    
        optimizer_discriminator.zero_grad()
        rank_loss.backward()
        optimizer_discriminator.step()
                
        loss[e] = rank_loss.item()
        #print("[TRAIN] Epoch:{:04d} | RANK loss:{:.4f} ".format(epoch, rank_loss.item()))
            
  
    return loss




def main(args, data, struc_encoding):

    nnodes = data.features.size()[0]
    nfeats = data.features.size()[1]
    features = data.features
    edge_types = data.graph.etypes
    se_dim = struc_encoding[edge_types[0]].shape[1]
    n_layers = args["n_layers"]
    dropout = args["dropout"]
    coeff = args["coeff"]
    
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()
    
    results = []
    
    if args["dataset"] == "yelp":
         
         in_dim = 32
         proj_dim = 32
         hid_dim = 128
         out_dim = 2
    
    
    else:
    
         in_dim = 25
         proj_dim = 25
         hid_dim = 128
         out_dim = 2
       
    
    labels = data.labels
    best_f1 = 0.
    
    discriminator = Edge_Discriminator(nnodes, nfeats + se_dim , args["alpha"], args["sparse"]) 
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=args["lr_disc"], weight_decay=args["w_decay"])
    
    DC_Module = Dual_Channel_Module(data, n_layers, in_dim, hid_dim, out_dim, proj_dim, dropout, coeff)
    optimizer_DC_Module = torch.optim.Adam(DC_Module.parameters(), lr=args["lr_DCA"], weight_decay=args["w_decay"])
    
    
    
    for epoch in range(1, args['epochs'] + 1):
    
        DC_loss, logits = train_DC_Module(DC_Module, discriminator, optimizer_DC_Module, data, features, struc_encoding, args)
        rank_loss = train_disc(DC_Module, discriminator, optimizer_discriminator, data, features, struc_encoding, args, epoch)
        
        #print("[TRAIN] Epoch:{:04d} | DC Loss {:.4f}".format(epoch, DC_loss))
        
        
        if epoch % args['eval_freq'] == 0:
        
            DC_Module.eval()
            discriminator.eval()
            
            #weight_dict = discriminator(data, features, struc_encoding)
            #final_embedding = DC_Module(data, weight_dict, coeff)
            
            final_trec, final_tpre, final_tmf1, final_tauc = eval_mode(logits, labels, train_mask, val_mask, test_mask, epoch, DC_loss, rank_loss, best_f1)


    print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(final_trec*100, final_tpre*100, final_tmf1*100, final_tauc*100))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='load_data_preparation')
    parser.add_argument('--dataset', type=str, default='yelp',
                        help='Dataset name, [amazon, yelp]')

    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')
    parser.add_argument('--SE_dimension', type=int, default=16,
                        help='Structural Encoding Dimension.')

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
    
    
    parser.add_argument('-sparse', type=int, default=0)
   
    parser.add_argument('-epochs', type=int, default=500)
   
    parser.add_argument('-lr_disc', type=float, default=0.01)
    
    parser.add_argument('-lr_DCA', type=float, default=0.01)
  
    parser.add_argument('-w_decay', type=float, default=0.00005)
    
    parser.add_argument('-dropout', type=float, default=0.5)
    
    parser.add_argument('-eval_freq', type=int, default=1)
    
    parser.add_argument('-n_layers', type=int, default=1)
    
    parser.add_argument('-coeff', type=int, default=3)

    # DISC Module - Hyper-param
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-margin_hom', type=float, default=0.5)
    parser.add_argument('-margin_het', type=float, default=0.5)

    
    
    args = vars(parser.parse_args())
    print(args)
    
    
    data = prepare_data(args)
    print(data)
    
    
    # Compute Structural Encoding 
    
    path = '../data/se/{}'.format(args['dataset'])
    if not os.path.exists(path):
        os.makedirs(path)
    file_name = path + '/{}_{}.pt'.format(args['dataset'], args['SE_dimension'])
    if os.path.exists(file_name):
        print('Load exist structural encoding.')
        se = torch.load(file_name)
        #print("structural Encoding:", se)
        
    else:
        print('Computing structural encoding...')
        se = compute_structural_encoding(data, args['SE_dimension'])
        torch.save(se, file_name)
        print('Done. The structural encoding is saved as: {}.'.format(file_name))
        
        
    # train phase starts
    print("training phase starts...")
    main(args, data, se)
    
