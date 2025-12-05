import sys
import os
import torch
import numpy as np
import random
from data_utils import DataUtils
from model_variants import PureLightGCN, BicliqueGCN, BicliqueCL, FullTSBCL

# Configuration
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'datasets', 'bi_github.txt')
MSBE_EXE = os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'msbe')
if os.name == 'nt':
    MSBE_EXE += ".exe"

EMBEDDING_DIM = 64
NUM_SNAPSHOTS = 4
TOP_K = 20
TAU = 2
EPSILON = 0.1

def calculate_metrics_robust(u_emb, i_emb, test_data, k=20):
    # u_emb, i_emb are tensors
    u_emb = u_emb.cpu().numpy()
    i_emb = i_emb.cpu().numpy()
    
    test_users = list(set([u for u, _, _ in test_data]))
    # Use ALL users for robust evaluation (or a large sample)
    # sample_users = random.sample(test_users, 1000) 
    sample_users = test_users 
    print(f"Evaluating on {len(sample_users)} users...")
    
    hits = 0
    ndcg = 0
    
    # Ground Truth
    user_pos_items = {}
    for u, i, _ in test_data:
        if u not in user_pos_items:
            user_pos_items[u] = []
        user_pos_items[u].append(i)
        
    for idx, u in enumerate(sample_users):
        if idx % 1000 == 0:
            print(f"  Processed {idx}/{len(sample_users)} users...")

        if u not in user_pos_items: continue
        
        # u is already the index
        if u >= len(u_emb): continue
        
        u_vec = u_emb[u]
        scores = np.dot(i_emb, u_vec)
        
        top_k_items = np.argsort(scores)[::-1][:k]
        
        true_items = user_pos_items[u] # Already indices
        
        # Recall
        hit_count = len(set(top_k_items) & set(true_items))
        hits += hit_count / len(true_items) if len(true_items) > 0 else 0
        
        # NDCG
        dcg = 0
        idcg = 0
        for i, item_idx in enumerate(top_k_items):
            if item_idx in true_items:
                dcg += 1 / np.log2(i + 2)
        
        for i in range(min(len(true_items), k)):
            idcg += 1 / np.log2(i + 2)
            
        ndcg += dcg / idcg if idcg > 0 else 0
        
    return hits / len(sample_users), ndcg / len(sample_users)

def evaluate_saved_model(model_class, model_path, model_type, utils, test_data, device):
    print(f"\nLoading {model_type} from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return
        
    model = model_class(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        # Prepare test data structures
        if model_type != "lightgcn":
            print("Mining bicliques for test data...")
            biclique_file = utils.run_msbe_mining(test_data, "solo_test_eval", tau=TAU, epsilon=EPSILON)
            H_v_test, H_u_test = utils.parse_bicliques(biclique_file)
            H_v_test, H_u_test = H_v_test.to(device), H_u_test.to(device)
        else:
            H_v_test, H_u_test = None, None
        
        test_adj = utils.build_adj_matrix(test_data).to(device)
        
        # Forward pass
        # Note: user_history_state is None for static evaluation
        user_history_state = None 
        
        if model_type == "lightgcn":
            u_emb, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
        elif model_type == "biclique_gcn":
            u_emb, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
        elif model_type == "biclique_cl":
            u_emb, _, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
        elif model_type == "full":
            u_emb, _, _, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
        
        print("Calculating metrics...")
        recall, ndcg = calculate_metrics_robust(u_emb, i_emb, test_data, k=TOP_K)
        print(f"Result for {model_type}: Recall@20 = {recall:.4f}, NDCG@20 = {ndcg:.4f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    test_data = snapshots[-1]
    
    # Evaluate Baseline
    evaluate_saved_model(PureLightGCN, "lightgcn_best.pth", "lightgcn", utils, test_data, device)
    
    # Evaluate Biclique GCN
    evaluate_saved_model(BicliqueGCN, "biclique_gcn_best.pth", "biclique_gcn", utils, test_data, device)
