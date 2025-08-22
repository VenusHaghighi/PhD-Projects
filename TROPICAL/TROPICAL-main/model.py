import torch
import torch.nn as nn


import positional_encoding




class TransformerNet(nn.Module):

    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, dim_feedforward, n_layers, n_feat, dropout, agg_type='cat'):
        r"""
        Parameters
        ----------
        feat_dim : int
            Input feature size; i.e., number of  dimensions of the raw input feature.
        emb_dim : int
            Hidden size of all learning embeddings and hidden vectors. 
        n_classes : int
            Number of classes    
        n_hops : int
            Number of hops (mulit-hop neighborhood information)
        n_relations : int
            Number of relations. 
        n_heads : int
            Number of heads in MultiHeadAttention module.
        dim_feedforward : int
        n_layers : int
            Number of encoders layers. 
        n_feat : int
            Number of knn-based and self-feature    
        dropout: float
            Dropout rate on feature. Default=0.1.
        agg_type: str
            aggregation type, including 'cat' and 'mean'.
        
        Return : torch.Tensor
            Final representation of target nodes. 
         
        """ 
        super(TransformerNet, self).__init__()
        
        
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.n_feat = n_feat
        self.agg_type = agg_type
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = nn.Dropout(dropout)

        # encoder that provides hop, relation and group encodings
        self.feat_encoder = positional_encoding.Positional_Encoder(feat_dim=feat_dim,
                                                    emb_dim=emb_dim, n_classes=n_classes, n_hops=n_hops, n_relations=n_relations,
                                                    n_feat=n_feat, dropout=dropout)

        # define transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        # cross-relation aggregation type 
        if agg_type == 'cat':
            proj_emb_dim = emb_dim * n_relations
        elif agg_type == 'mean':
            proj_emb_dim = emb_dim

        # the MLP 
        self.projection = nn.Sequential(nn.Linear(proj_emb_dim, n_classes))
        

        

        self.init_weights()
        
        
        
    def cross_relation_agg(self, out):
    
        
        
        device = out.device
        n_tokens = out.shape[0]

        
        block_len = 1 + 1 + self.n_hops * (self.n_classes)
        indices = torch.arange(0, n_tokens, block_len, dtype=torch.int64).to(device)

     
        mr_feats = torch.index_select(out, dim=0, index=indices)
        if self.agg_type == 'cat':
        
            mr_feats = torch.split(mr_feats, 1, dim=0)

          
            agg_feats = torch.cat(mr_feats, dim=2).squeeze()

        elif self.agg_type == 'mean':
            
            agg_feats = torch.mean(mr_feats, dim=0)

        return agg_feats
        
          
        
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    
     

    def forward(self, src_emb, src_mask=None):
     
        # input feature sequence 
        src_emb = torch.transpose(src_emb, 1, 0)

        # positional encoding 
        out = self.feat_encoder(src_emb)

        # transformer encoder
        out = self.transformer_encoder(out, src_mask)

      
        out = self.cross_relation_agg(out)

        # prediction
        out = self.projection(out)

        return out 
    
