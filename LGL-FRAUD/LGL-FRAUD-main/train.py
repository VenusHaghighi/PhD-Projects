import dgl
import torch
import torch.nn.functional as F
import numpy
import argparse
import time
from data_preparation import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
from BWGNN import *
from sklearn.model_selection import train_test_split


def train(model, g, args):
    features = g.ndata['feature']
    labels = g.ndata['label']
    index = list(range(len(labels)))
    if dataset_name == 'amazon':
        index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args['train_ratio'],
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    print('cross entropy weight: ', weight)
    time_start = time.time()
    for e in range(args['epoch']):
        model.train()
        logits = model(features)
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
        tpre = precision_score(labels[test_mask], preds[test_mask])
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
    torch.save(probs, 'final_embedding.pt')
    
    return final_tmf1, final_tauc, probs


# threshold adjusting for best macro f1
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LLM_Augmented Fraud Detection Model')
    parser.add_argument('--dataset', type=str, default='yelp',
                        help='Dataset name, [amazon, yelp]')
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=0, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=500, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")
    
    
    

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

    parser.add_argument('--base_dir', type=str, default='~/.dgl',
                        help='Directory for loading graph data.')
    
    
    args = vars(parser.parse_args())
    print(args)
    
    
    data, graph = prepare_data(args)
    print(data)

    dataset_name = args['dataset']
    homo = args['homo']
    order = args['order']
    h_feats = args['hid_dim']
    #graph = Dataset(dataset_name, homo).graph
    in_feats = data.feat_dim
    num_classes = data.n_classes

    if args['run'] == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
        else:
            model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
        train(model, graph, args)
        

    else:
        final_mf1s, final_aucs = [], []
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph, d=order)
            else:
                model = BWGNN_Hetero(in_feats, h_feats, num_classes, graph, d=order)
            mf1, auc, probs = train(model, graph, args)
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s),
                                                                100 * np.mean(final_aucs), 100 * np.std(final_aucs)))
        
   
