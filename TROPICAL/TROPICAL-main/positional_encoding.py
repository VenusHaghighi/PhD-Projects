import torch
import torch.nn as nn


class Positional_Encoder(nn.Module):

      def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations, n_feat, dropout=0.1):
        
            super(Positional_Encoder, self).__init__()
            self.hop_embedding = HopEmbedding(n_hops , emb_dim)
            self.relation_embedding = RelationEmbedding(n_relations, emb_dim)
            self.group_embedding = GroupEmbedding(n_classes , emb_dim)
            self.feat_embedding = FeatEmbedding(n_feat , emb_dim)
    
            self.MLP = nn.Sequential(nn.Linear(feat_dim, emb_dim),
                                     nn.ReLU())
    
            self.dropout = nn.Dropout(dropout)
    
            self.feat_dim = feat_dim
            self.emb_dim = emb_dim
            self.n_hops = n_hops
            self.n_relations = n_relations
            self.n_classes = n_classes
            self.n_feat = n_feat
    
          
            self.n_groups = n_classes
    
            # input sequence length under single relartion
            self.base_seq_len = (n_hops * (n_classes)) #+ 1 + 1

      
      def forward(self, x):
       
        device = x.device

        # the device of indices should consist with the learnable embeddings
        hop_idx = torch.arange(self.n_hops, dtype=torch.int64).to(device)
        rel_idx = torch.arange(self.n_relations, dtype=torch.int64).to(device)
        grp_idx = torch.arange(self.n_groups, dtype=torch.int64).to(device)
        feat_idx = torch.arange(self.n_feat, dtype=torch.int64).to(device)
        
        # -------- HOP ENCODING STRATEGY --------
      
        hop_emb = self.hop_embedding(hop_idx)
        
      
        feat_emb = self.feat_embedding(feat_idx)
        
        
        hop_emb_list = []
        
        for i in range(0, self.n_hops):
            hop_emb_list.append(hop_emb[i].repeat(self.n_groups, 1))
        
        
        hop_emb = torch.cat(hop_emb_list, dim=0).repeat(self.n_relations, 1)
        
        hop_emb = torch.cat([hop_emb, feat_emb[0].unsqueeze(0)], dim=0)
        
        hop_emb = torch.cat([hop_emb, feat_emb[1].unsqueeze(0)], dim=0)
        
        
       
        

        # -------- RELATION ENCODING STRATEGY --------
     
        rel_emb = self.relation_embedding(rel_idx)
    
        rel_emb = rel_emb.repeat(1, self.base_seq_len).view(-1, self.emb_dim)
        
        feat_emb = self.feat_embedding(feat_idx)
        
        rel_emb = torch.cat([rel_emb, feat_emb[0].unsqueeze(0)], dim=0)
        
        rel_emb = torch.cat([rel_emb, feat_emb[1].unsqueeze(0)], dim=0)
        
        
        
        # -------- GROUP ENCODING STRATEGY --------
   
       
        grp_emb = self.group_embedding(grp_idx)
        
        grp_emb = grp_emb.repeat(self.n_hops, 1)

        grp_emb = grp_emb.repeat(self.n_relations, 1)
        
        
        feat_emb = self.feat_embedding(feat_idx)
        
        grp_emb = torch.cat([grp_emb, feat_emb[0].unsqueeze(0)], dim=0)
        
        grp_emb = torch.cat([grp_emb, feat_emb[1].unsqueeze(0)], dim=0)
        
     
        
        

        # linear projection
        out = self.MLP(x)
        
        
        # broadcast x
        out = out + hop_emb.unsqueeze(1) + rel_emb.unsqueeze(1) + grp_emb.unsqueeze(1) 
        
        out = self.dropout(out)
        

        return out
        
        
        
class HopEmbedding(nn.Embedding):
    def __init__(self, max_len, emb_dim=128):
        """Hop Embeddings.

        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(HopEmbedding, self).__init__(max_len, emb_dim)


class RelationEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):
        """Relation Embeddings.
        
        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(RelationEmbedding, self).__init__(max_len, emb_dim)


class GroupEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):
        """Group Embeddings.
        
        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(GroupEmbedding, self).__init__(max_len, emb_dim)
        
        
class FeatEmbedding(nn.Embedding):
    def __init__(self, max_len: int, emb_dim=128):
        """Group Embeddings.
        
        Parameters
        ----------
        max_len: int
            Number of learnable embeddings.
        emb_dim: int
            Embedding size, i.e., number of dimensions of learnable embeddings.
        """
        super(FeatEmbedding, self).__init__(max_len, emb_dim)
        