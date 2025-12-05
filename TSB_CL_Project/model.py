
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BicliqueEnhancedEncoder(nn.Module):
    """
    二团增强编码器 (Biclique-Enhanced Encoder)
    作用：将挖掘出的二团视为超边，聚合二团内的信息，形成局部视图表示。
    """
    def __init__(self, embedding_dim):
        super(BicliqueEnhancedEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        # 用于聚合二团特征的注意力机制或简单的平均池化
        self.project = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, user_emb, item_emb, biclique_indices):
        """
        :param user_emb: 当前时间步的用户Embedding
        :param item_emb: 当前时间步的物品Embedding
        :param biclique_indices: 一个列表，每个元素是一个二团 (users_list, items_list)
        :return: 用户在二团视图下的Embedding
        """
        # 初始化二团视图的用户Embedding，默认为0
        local_user_emb = torch.zeros_like(user_emb)
        
        # 这是一个简化的实现，实际中应使用稀疏矩阵乘法以提高效率
        # 这里为了演示逻辑，使用循环（在生产环境中请转换为矩阵操作）
        
        # 1. 计算每个二团的Embedding (Mean Pooling of Items in Biclique)
        # B_k = Mean({e_i | i in V_k})
        
        # 为了效率，我们假设 biclique_indices 已经被处理成稀疏矩阵形式
        # H_v: [num_bicliques, num_items]
        # H_u: [num_users, num_bicliques]
        
        H_v, H_u = biclique_indices # 解包稀疏矩阵
        
        # 计算二团特征: B = H_v * Item_Emb (归一化后)
        # H_v 是稀疏矩阵
        biclique_features = torch.sparse.mm(H_v, item_emb) 
        # 归一化：除以每个二团包含的物品数量
        degree_v = torch.sparse.sum(H_v, dim=1).to_dense().view(-1, 1)
        degree_v[degree_v == 0] = 1.0
        biclique_features = biclique_features / degree_v
        
        # 2. 将二团特征聚合回用户
        # U_local = H_u * B
        user_local_view = torch.sparse.mm(H_u, biclique_features)
        # 归一化：除以每个用户所属的二团数量
        degree_u = torch.sparse.sum(H_u, dim=1).to_dense().view(-1, 1)
        degree_u[degree_u == 0] = 1.0
        user_local_view = user_local_view / degree_u
        
        return user_local_view

class LightGCNEncoder(nn.Module):
    """
    全局视图编码器 (Global View Encoder)
    使用标准的LightGCN进行邻居聚合。
    """
    def __init__(self, num_users, num_items, embedding_dim, n_layers=3):
        super(LightGCNEncoder, self).__init__()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
    def forward(self, user_emb, item_emb, adj_matrix):
        """
        :param adj_matrix: 归一化的邻接矩阵 (Sparse Tensor)
        """
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        users, items = torch.split(final_emb, [user_emb.shape[0], item_emb.shape[0]])
        return users, items

class TSB_CL(nn.Module):
    """
    TSB-CL: 基于最大相似二团增强的时序图神经网络推荐算法
    """
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, tau=0.2):
        super(TSB_CL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.tau = tau # 对比学习温度参数
        
        # 基础Embedding
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 编码器
        self.global_encoder = LightGCNEncoder(num_users, num_items, embedding_dim, n_layers)
        self.local_encoder = BicliqueEnhancedEncoder(embedding_dim)
        
        # 时序演化模块 (GRU)
        # 用于更新用户Embedding，捕捉兴趣漂移
        self.user_gru = nn.GRUCell(embedding_dim, embedding_dim)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, biclique_matrices, user_history_state=None):
        """
        前向传播 (针对一个时间片)
        :param adj_matrix: 全局交互图邻接矩阵
        :param biclique_matrices: (H_v, H_u) 二团超图关联矩阵
        :param user_history_state: 上一时刻的用户状态 (Hidden State)
        """
        # 1. 获取当前的基础Embedding
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        # 2. 全局视图编码 (LightGCN)
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        
        # 3. 局部二团视图编码 (Biclique Encoder)
        # 注意：这里只增强用户表示，因为二团代表用户社群
        u_local = self.local_encoder(u_emb, i_emb, biclique_matrices)
        
        # 4. 时序演化 (Temporal Evolution)
        # 将全局视图特征融合进GRU状态
        if user_history_state is None:
            user_history_state = torch.zeros_like(u_emb)
            
        # GRU Update: h_t = GRU(input=u_global, hidden=h_{t-1})
        new_user_state = self.user_gru(u_global, user_history_state)
        
        return new_user_state, u_local, new_user_state, i_global

    def calculate_loss(self, u_global, u_local, i_global, users, pos_items, neg_items):
        """
        计算总损失 = BPR Loss + Contrastive Loss
        """
        # --- 1. BPR Loss (推荐任务) ---
        current_user_emb = u_global[users]
        pos_item_emb = i_global[pos_items]
        neg_item_emb = i_global[neg_items]
        
        pos_scores = torch.mul(current_user_emb, pos_item_emb).sum(dim=1)
        neg_scores = torch.mul(current_user_emb, neg_item_emb).sum(dim=1)
        
        bpr_loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        
        # --- 2. Contrastive Loss (结构增强) ---
        # 目标：最大化同一用户在 Global View 和 Local View 下的互信息
        # InfoNCE Loss
        
        # 归一化
        u_view1 = F.normalize(u_global[users], dim=1)
        u_view2 = F.normalize(u_local[users], dim=1)
        
        # 正样本相似度
        pos_sim = torch.sum(u_view1 * u_view2, dim=1) / self.tau
        
        # 负样本相似度 (Batch内其他用户)
        # 矩阵乘法: [batch, dim] * [dim, batch] = [batch, batch]
        all_sim = torch.mm(u_view1, u_view2.t()) / self.tau
        
        # LogSumExp trick for numerical stability
        cl_loss = -torch.mean(pos_sim - torch.logsumexp(all_sim, dim=1))
        
        # 总损失 (lambda 可以调节)
        total_loss = bpr_loss + 0.1 * cl_loss
        
        return total_loss, bpr_loss, cl_loss
