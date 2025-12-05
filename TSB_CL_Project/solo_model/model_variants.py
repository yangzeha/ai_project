import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 基础组件 ---

class BicliqueEnhancedEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(BicliqueEnhancedEncoder, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, user_emb, item_emb, biclique_indices):
        H_v, H_u = biclique_indices
        # 1. Item -> Biclique
        biclique_features = torch.sparse.mm(H_v, item_emb)
        degree_v = torch.sparse.sum(H_v, dim=1).to_dense().view(-1, 1)
        degree_v[degree_v == 0] = 1.0
        biclique_features = biclique_features / degree_v
        
        # 2. Biclique -> User
        user_local_view = torch.sparse.mm(H_u, biclique_features)
        degree_u = torch.sparse.sum(H_u, dim=1).to_dense().view(-1, 1)
        degree_u[degree_u == 0] = 1.0
        user_local_view = user_local_view / degree_u
        
        return user_local_view

class LightGCNEncoder(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, n_layers=3):
        super(LightGCNEncoder, self).__init__()
        self.n_layers = n_layers
        
    def forward(self, user_emb, item_emb, adj_matrix):
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        users, items = torch.split(final_emb, [user_emb.shape[0], item_emb.shape[0]])
        return users, items

# --- 四个模型变体 ---

# 1. 纯 LightGCN
class PureLightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(PureLightGCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.global_encoder = LightGCNEncoder(num_users, num_items, embedding_dim, n_layers)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, biclique_matrices=None, user_history_state=None):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        return u_global, i_global

    def calculate_loss(self, u_out, i_out, users, pos_items, neg_items):
        u_curr = u_out[users]
        pos_i = i_out[pos_items]
        neg_i = i_out[neg_items]
        pos_scores = torch.mul(u_curr, pos_i).sum(dim=1)
        neg_scores = torch.mul(u_curr, neg_i).sum(dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss, bpr_loss, 0.0

# 2. Biclique GCN (无对比学习，直接融合)
class BicliqueGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(BicliqueGCN, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.global_encoder = LightGCNEncoder(num_users, num_items, embedding_dim, n_layers)
        self.local_encoder = BicliqueEnhancedEncoder(embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, biclique_matrices, user_history_state=None):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        u_local = self.local_encoder(u_emb, i_emb, biclique_matrices)
        
        # 直接融合：Global + Local
        u_final = u_global + u_local 
        return u_final, i_global

    def calculate_loss(self, u_out, i_out, users, pos_items, neg_items):
        # 标准 BPR
        u_curr = u_out[users]
        pos_i = i_out[pos_items]
        neg_i = i_out[neg_items]
        pos_scores = torch.mul(u_curr, pos_i).sum(dim=1)
        neg_scores = torch.mul(u_curr, neg_i).sum(dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return bpr_loss, bpr_loss, 0.0

# 3. Biclique + CL (无 RNN)
class BicliqueCL(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, tau=0.2):
        super(BicliqueCL, self).__init__()
        self.tau = tau
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.global_encoder = LightGCNEncoder(num_users, num_items, embedding_dim, n_layers)
        self.local_encoder = BicliqueEnhancedEncoder(embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, biclique_matrices, user_history_state=None):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        u_local = self.local_encoder(u_emb, i_emb, biclique_matrices)
        
        # 返回两个视图用于计算 CL Loss
        # 推荐时主要使用 Global 视图 (受 CL 约束)
        return u_global, u_local, i_global

    def calculate_loss(self, u_global, u_local, i_global, users, pos_items, neg_items):
        # 1. BPR Loss (on Global View)
        u_curr = u_global[users]
        pos_i = i_global[pos_items]
        neg_i = i_global[neg_items]
        pos_scores = torch.mul(u_curr, pos_i).sum(dim=1)
        neg_scores = torch.mul(u_curr, neg_i).sum(dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # 2. Contrastive Loss
        u_view1 = F.normalize(u_global[users], dim=1)
        u_view2 = F.normalize(u_local[users], dim=1)
        pos_sim = torch.sum(u_view1 * u_view2, dim=1) / self.tau
        all_sim = torch.mm(u_view1, u_view2.t()) / self.tau
        cl_loss = -torch.mean(pos_sim - torch.logsumexp(all_sim, dim=1))
        
        total_loss = bpr_loss + 0.1 * cl_loss
        return total_loss, bpr_loss, cl_loss

# 4. Full TSB-CL (Biclique + CL + RNN)
class FullTSBCL(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, tau=0.2):
        super(FullTSBCL, self).__init__()
        self.tau = tau
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.global_encoder = LightGCNEncoder(num_users, num_items, embedding_dim, n_layers)
        self.local_encoder = BicliqueEnhancedEncoder(embedding_dim)
        self.user_gru = nn.GRUCell(embedding_dim, embedding_dim) # RNN 组件
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, biclique_matrices, user_history_state=None):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        u_local = self.local_encoder(u_emb, i_emb, biclique_matrices)
        
        # RNN Update
        if user_history_state is None:
            user_history_state = torch.zeros_like(u_emb)
        new_user_state = self.user_gru(u_global, user_history_state)
        
        return new_user_state, u_local, new_user_state, i_global

    def calculate_loss(self, u_global, u_local, i_global, users, pos_items, neg_items):
        # 同 BicliqueCL
        u_curr = u_global[users]
        pos_i = i_global[pos_items]
        neg_i = i_global[neg_items]
        pos_scores = torch.mul(u_curr, pos_i).sum(dim=1)
        neg_scores = torch.mul(u_curr, neg_i).sum(dim=1)
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        u_view1 = F.normalize(u_global[users], dim=1)
        u_view2 = F.normalize(u_local[users], dim=1)
        pos_sim = torch.sum(u_view1 * u_view2, dim=1) / self.tau
        all_sim = torch.mm(u_view1, u_view2.t()) / self.tau
        cl_loss = -torch.mean(pos_sim - torch.logsumexp(all_sim, dim=1))
        
        total_loss = bpr_loss + 0.1 * cl_loss
        return total_loss, bpr_loss, cl_loss
