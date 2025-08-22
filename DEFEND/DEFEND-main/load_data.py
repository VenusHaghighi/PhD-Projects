import os
import torch
from scipy import io, sparse
import numpy as np
import dgl
import numpy as np
import torch
import argparse
from dgl import function as fn
from scipy import sparse as sp
from sklearn import preprocessing
from collections import namedtuple
from dgl.data.utils import save_graphs, load_graphs, _get_dgl_url
from dgl.convert import heterograph
from dgl.data import DGLBuiltinDataset
from dgl import backend as F
from preprocessing import *


class FraudDataset(DGLBuiltinDataset):
    r"""Fraud node prediction dataset.

    The dataset includes two multi-relational graphs extracted from Yelp and Amazon
    where nodes represent fraudulent reviews or fraudulent reviewers.

    It was first proposed in a CIKM'20 paper <https://arxiv.org/pdf/2008.08692.pdf> and
    has been used by a recent WWW'21 paper <https://ponderly.github.io/pub/PCGNN_WWW2021.pdf>
    as a benchmark. Another paper <https://arxiv.org/pdf/2104.01404.pdf> also takes
    the dataset as an example to study the non-homophilous graphs. This dataset is built
    upon industrial data and has rich relational information and unique properties like
    class-imbalance and feature inconsistency, which makes the dataset be a good instance
    to investigate how GNNs perform on real-world noisy graphs. These graphs are bidirected
    and not self connected.

    Reference: <https://github.com/YingtongDou/CARE-GNN>

    Parameters
    ----------
    name : str
        Name of the dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    random_seed : int
        Specifying the random seed in splitting the dataset.
        Default: 717
    train_size : float
        training set size of the dataset.
        Default: 0.7
    val_size : float
        validation set size of the dataset, and the
        size of testing set is (1 - train_size - val_size)
        Default: 0.1
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_classes : int
        Number of label classes
    graph : dgl.DGLGraph
        Graph structure, etc.
    seed : int
        Random seed in splitting the dataset.
    train_size : float
        Training set size of the dataset.
    val_size : float
        Validation set size of the dataset

    Examples
    --------
    >>> dataset = FraudDataset('yelp')
    >>> graph = dataset[0]
    >>> num_classes = dataset.num_classes
    >>> feat = graph.ndata['feature']
    >>> label = graph.ndata['label']
    """
    file_urls = {
        'yelp': 'dataset/FraudYelp.zip',
        'amazon': 'dataset/FraudAmazon.zip'    
    }
    relations = {
        'yelp': ['net_rsr', 'net_rtr', 'net_rur'],
        'amazon': ['net_upu', 'net_usu', 'net_uvu'],
    }
    file_names = {
        'yelp': 'YelpChi.mat',
        'amazon': 'Amazon.mat'
    }
    node_name = {
        'yelp': 'review',
        'amazon': 'user'
    }

    def __init__(self, name, raw_dir=None, random_seed=717, train_size=0.7,
                 val_size=0.1, force_reload=False, verbose=True):
        assert name in ['yelp', 'amazon', 'mimic'], "only supports 'yelp', 'amazon', 'mimic'"
        url = _get_dgl_url(self.file_urls[name])
        self.seed = random_seed
        self.train_size = train_size
        self.val_size = val_size
        super(FraudDataset, self).__init__(name=name,
                                           url=url,
                                           raw_dir=raw_dir,
                                           hash_key=(random_seed, train_size, val_size),
                                           force_reload=force_reload,
                                           verbose=verbose)

    def process(self):
        """process raw data to graph, labels, splitting masks"""
        file_path = os.path.join(self.raw_path, self.file_names[self.name])

        data = io.loadmat(file_path)

        if sparse.issparse(data['features']):
            node_features = data['features'].todense()
        else:
            node_features = data['features']
        # remove additional dimension of length 1 in raw .mat file
        node_labels = data['label'].squeeze()

        graph_data = {}
        for relation in self.relations[self.name]:
            adj = data[relation].tocoo()
            row, col = adj.row, adj.col
            graph_data[(self.node_name[self.name], relation, self.node_name[self.name])] = (row, col)
        g = heterograph(graph_data)

        g.ndata['feature'] = F.tensor(node_features, dtype=F.data_type_dict['float32'])
        g.ndata['label'] = F.tensor(node_labels, dtype=F.data_type_dict['int64'])
        self.graph = g

        self._random_split(g.ndata['feature'], self.seed, self.train_size, self.val_size)

    def __getitem__(self, idx):
        r""" Get graph object

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        :class:`dgl.DGLGraph`
            graph structure, node features, node labels and masks

            - ``ndata['feature']``: node features
            - ``ndata['label']``: node labels
            - ``ndata['train_mask']``: mask of training set
            - ``ndata['val_mask']``: mask of validation set
            - ``ndata['test_mask']``: mask of testing set
        """
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        """number of data examples"""
        return len(self.graph)

    @property
    def num_classes(self):
        """Number of classes.

        Return
        -------
        int
        """
        return 2

    def save(self):
        """save processed data to directory `self.save_path`"""
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph_{}.bin'.format(self.hash))
        save_graphs(str(graph_path), self.graph)

    def load(self):
        """load processed data from directory `self.save_path`"""
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph_{}.bin'.format(self.hash))
        graph_list, _ = load_graphs(str(graph_path))
        g = graph_list[0]
        self.graph = g

    def has_cache(self):
        """check whether there are processed data in `self.save_path`"""
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph_{}.bin'.format(self.hash))
        return os.path.exists(graph_path)

    def _random_split(self, x, seed=717, train_size=0.7, val_size=0.1):
        """split the dataset into training set, validation set and testing set"""

        assert 0 <= train_size + val_size <= 1, \
            "The sum of valid training set size and validation set size " \
            "must between 0 and 1 (inclusive)."

        N = x.shape[0]
        index = np.arange(N)
        if self.name == 'amazon':
            # 0-3304 are unlabeled nodes
            index = np.arange(3305, N)

        index = np.random.RandomState(seed).permutation(index)
        train_idx = index[:int(train_size * len(index))]
        val_idx = index[len(index) - int(val_size * len(index)):]
        test_idx = index[int(train_size * len(index)):len(index) - int(val_size * len(index))]
        train_mask = np.zeros(N, dtype=bool)
        val_mask = np.zeros(N, dtype=bool)
        test_mask = np.zeros(N, dtype=bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        self.graph.ndata['train_mask'] = F.tensor(train_mask)
        self.graph.ndata['val_mask'] = F.tensor(val_mask)
        self.graph.ndata['test_mask'] = F.tensor(test_mask)


class FraudYelpDataset(FraudDataset):
    r""" Fraud Yelp Dataset

    The Yelp dataset includes hotel and restaurant reviews filtered (spam) and recommended
    (legitimate) by Yelp. A spam review detection task can be conducted, which is a binary
    classification task. 32 handcrafted features from <http://dx.doi.org/10.1145/2783258.2783370>
    are taken as the raw node features. Reviews are nodes in the graph, and three relations are:

        1. R-U-R: it connects reviews posted by the same user
        2. R-S-R: it connects reviews under the same product with the same star rating (1-5 stars)
        3. R-T-R: it connects two reviews under the same product posted in the same month.

    Statistics:

    - Nodes: 45,954
    - Edges:

        - R-U-R: 98,630
        - R-T-R: 1,147,232
        - R-S-R: 6,805,486

    - Classes:

        - Positive (spam): 6,677
        - Negative (legitimate): 39,277

    - Positive-Negative ratio: 1 : 5.9
    - Node feature size: 32

    Parameters
    ----------
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    random_seed : int
        Specifying the random seed in splitting the dataset.
        Default: 717
    train_size : float
        training set size of the dataset.
        Default: 0.7
    val_size : float
        validation set size of the dataset, and the
        size of testing set is (1 - train_size - val_size)
        Default: 0.1
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Examples
    --------
    >>> dataset = FraudYelpDataset()
    >>> graph = dataset[0]
    >>> num_classes = dataset.num_classes
    >>> feat = graph.ndata['feature']
    >>> label = graph.ndata['label']
    """

    def __init__(self, raw_dir=None, random_seed=717, train_size=0.7,
                 val_size=0.1, force_reload=False, verbose=True):
        super(FraudYelpDataset, self).__init__(name='yelp',
                                               raw_dir=raw_dir,
                                               random_seed=random_seed,
                                               train_size=train_size,
                                               val_size=val_size,
                                               force_reload=force_reload,
                                               verbose=verbose)


class FraudAmazonDataset(FraudDataset):
    r""" Fraud Amazon Dataset

    The Amazon dataset includes product reviews under the Musical Instruments category.
    Users with more than 80% helpful votes are labelled as benign entities and users with
    less than 20% helpful votes are labelled as fraudulent entities. A fraudulent user
    detection task can be conducted on the Amazon dataset, which is a binary classification
    task. 25 handcrafted features from <https://arxiv.org/pdf/2005.10150.pdf> are taken as
    the raw node features.

    Users are nodes in the graph, and three relations are:
    1. U-P-U : it connects users reviewing at least one same product
    2. U-S-U : it connects users having at least one same star rating within one week
    3. U-V-U : it connects users with top 5% mutual review text similarities (measured by
    TF-IDF) among all users.

    Statistics:

    - Nodes: 11,944
    - Edges:

        - U-P-U: 351,216
        - U-S-U: 7,132,958
        - U-V-U: 2,073,474

    - Classes:

        - Positive (fraudulent): 821
        - Negative (benign): 7,818
        - Unlabeled: 3,305

    - Positive-Negative ratio: 1 : 10.5
    - Node feature size: 25

    Parameters
    ----------
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    random_seed : int
        Specifying the random seed in splitting the dataset.
        Default: 717
    train_size : float
        training set size of the dataset.
        Default: 0.7
    val_size : float
        validation set size of the dataset, and the
        size of testing set is (1 - train_size - val_size)
        Default: 0.1
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Examples
    --------
    >>> dataset = FraudAmazonDataset()
    >>> graph = dataset[0]
    >>> num_classes = dataset.num_classes
    >>> feat = graph.ndata['feature']
    >>> label = graph.ndata['label']
    """

    def __init__(self, raw_dir=None, random_seed=717, train_size=0.7,
                 val_size=0.1, force_reload=False, verbose=True):
        super(FraudAmazonDataset, self).__init__(name='amazon',
                                                 raw_dir=raw_dir,
                                                 random_seed=random_seed,
                                                 train_size=train_size,
                                                 val_size=val_size,
                                                 force_reload=force_reload,
                                                 verbose=verbose)




def normalize(feats, train_nid, dtype=np.float32):
    r"""Standardize features by removing the mean and scaling to unit variance.
    Reference: <sklearn.preprocessing.StandardScaler>
    
    Parameters
    ----------
    feats : np.ndarray
        Feature matrix of all nodes.
    train_nid : np.ndarray
        Node ids of training nodes.
    dtype : np.dtype
        Data type for normalized features. Default=np.float32

    Return : np.ndarray
        Normalized features.
    """
    train_feats = feats[train_nid]
    scaler = preprocessing.StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    return feats.astype(dtype)




def row_normalize(mx, dtype=np.float32):
    r"""Row-normalize sparse matrix.
    Reference: <https://github.com/williamleif/graphsage-simple>
    
    Parameters
    ----------
    mx : np.ndarray
        Feature matrix of all nodes.
    dtype : np.dtype
        Data type for normalized features. Default=np.float32

    Return : np.ndarray
        Normalized features.
    """
    rowsum = np.array(mx.sum(1)) + 0.01
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    
    return mx.astype(dtype)





def load_data(dataset_name='yelp', raw_dir='~/.dgl/', train_size=0.4, val_size=0.1,
               seed=717, norm=True, force_reload=False, verbose=True) -> dict:
    """Loading dataset from dgl's FraudDataset.
    """
    if dataset_name in ['amazon', 'yelp']:
        fraud_data = FraudDataset(dataset_name, train_size=train_size, val_size=val_size,
                                                random_seed=seed, force_reload=force_reload)
    

    g = fraud_data[0]

    # Feature tensor dtpye is float64, change it to float32
    if norm:
        h = row_normalize(g.ndata['feature'], dtype=np.float32)
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


    
    return graph_data



  
   