import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import DataUtils
import random

# --- Model Classes from model_variants.py ---

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
        u_final = u_global + u_local 
        return u_final, i_global

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
        return u_global, u_local, i_global

class FullTSBCL(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3, tau=0.2):
        super(FullTSBCL, self).__init__()
        self.tau = tau
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.global_encoder = LightGCNEncoder(num_users, num_items, embedding_dim, n_layers)
        self.local_encoder = BicliqueEnhancedEncoder(embedding_dim)
        self.user_gru = nn.GRUCell(embedding_dim, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix, biclique_matrices, user_history_state=None):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        u_local = self.local_encoder(u_emb, i_emb, biclique_matrices)
        if user_history_state is None:
            user_history_state = torch.zeros_like(u_emb)
        new_user_state = self.user_gru(u_global, user_history_state)
        return u_global, u_local, new_user_state, i_global

# --- Evaluation Logic ---

def evaluate_model(model, test_data, utils, device, model_type, biclique_matrices, top_k=20):
    model.eval()
    
    adj_matrix = utils.build_adj_matrix(test_data).to(device)
    H_v, H_u = biclique_matrices
    H_v = H_v.to(device)
    H_u = H_u.to(device)
    
    test_users = list(set([x[0] for x in test_data]))
    # Sample 1000 users for evaluation
    if len(test_users) > 1000:
        test_users = np.random.choice(test_users, 1000, replace=False)
        
    hits = 0
    ndcgs = 0
    
    with torch.no_grad():
        # Handle different return signatures
        if model_type == "PureLightGCN":
            u_out, i_out = model(adj_matrix)
        elif model_type == "BicliqueGCN":
            u_out, i_out = model(adj_matrix, (H_v, H_u))
        elif model_type == "BicliqueCL":
            u_out, _, i_out = model(adj_matrix, (H_v, H_u))
        elif model_type == "FullTSBCL":
            u_out, _, _, i_out = model(adj_matrix, (H_v, H_u))
        
        all_item_emb = i_out
        
        for u in test_users:
            ground_truth = set([x[1] for x in test_data if x[0] == u])
            if not ground_truth: continue
            
            u_emb = u_out[u].unsqueeze(0)
            scores = torch.mm(u_emb, all_item_emb.t()).squeeze()
            
            _, indices = torch.topk(scores, top_k)
            pred_items = indices.cpu().numpy()
            
            hit = 0
            dcg = 0
            idcg = 0
            
            for i, item in enumerate(pred_items):
                if item in ground_truth:
                    hit += 1
                    dcg += 1.0 / np.log2(i + 2)
            
            for i in range(min(len(ground_truth), top_k)):
                idcg += 1.0 / np.log2(i + 2)
                
            hits += hit / len(ground_truth)
            ndcgs += dcg / idcg if idcg > 0 else 0
            
    return hits / len(test_users), ndcgs / len(test_users)

def main():
    # --- Config ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", "datasets", "bi_github.txt")
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)
    EMBEDDING_DIM = 64
    NUM_SNAPSHOTS = 5
    
    MODEL_DIR = os.path.join(CURRENT_DIR, "model_path")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    test_data = snapshots[-1]
    print(f"Test data size: {len(test_data)} interactions")
    
    # Mine bicliques for test data (needed for BicliqueGCN)
    print("Mining bicliques for test data...")
    biclique_file = utils.run_msbe_mining(test_data, "test_mining", tau=2, epsilon=0.1)
    H_v, H_u = utils.parse_bicliques(biclique_file)
    print(f"Mined bicliques: {H_v.shape[0]}")
    
    models_to_test = [
        ("Baseline (LightGCN)", "lightgcn_best.pth", PureLightGCN, "PureLightGCN"),
        ("BicliqueGCN", "biclique_gcn_best.pth", BicliqueGCN, "BicliqueGCN"),
        ("BicliqueCL", "biclique_cl_best.pth", BicliqueCL, "BicliqueCL"),
        ("Full TSB-CL", "full_best.pth", FullTSBCL, "FullTSBCL")
    ]
    
    print("\n" + "="*50)
    print(f"{'Model':<25} | {'Recall@20':<10} | {'NDCG@20':<10}")
    print("-" * 50)
    
    for model_name, model_file, ModelClass, model_type in models_to_test:
        model_path = os.path.join(MODEL_DIR, model_file)
        if not os.path.exists(model_path):
            print(f"{model_name:<25} | File not found")
            continue
            
        # Initialize model
        model = ModelClass(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
        
        # Load weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            print(f"{model_name:<25} | Error loading: {e}")
            continue
            
        # Evaluate
        recall, ndcg = evaluate_model(model, test_data, utils, device, model_type, (H_v, H_u))
        
        print(f"{model_name:<25} | {recall:.4f}     | {ndcg:.4f}")
        
    print("="*50)

if __name__ == "__main__":
    main()
