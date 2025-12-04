import torch
import numpy as np
import sys
import os
import random

# Add path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import DataUtils
from solo_model.model_variants import PureLightGCN, BicliqueGCN

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'datasets', 'bi_github.txt')
MSBE_EXE = os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'msbe')
if os.name == 'nt': MSBE_EXE += ".exe"
EMBEDDING_DIM = 64
NUM_SNAPSHOTS = 4
TOP_K = 20

def load_model(model_class, path, num_users, num_items, device):
    model = model_class(num_users, num_items, EMBEDDING_DIM).to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Model file {path} not found.")
        return None

def get_recommendations(model, u_idx, adj_matrix, biclique_matrices, device, model_type="lightgcn"):
    with torch.no_grad():
        if model_type == "lightgcn":
            u_emb, i_emb = model(adj_matrix)
        elif model_type == "biclique_gcn":
            u_emb, i_emb = model(adj_matrix, biclique_matrices)
            
        u_vec = u_emb[u_idx].cpu().numpy()
        i_emb = i_emb.cpu().numpy()
        scores = np.dot(i_emb, u_vec)
        top_k_indices = np.argsort(scores)[::-1][:TOP_K]
        return top_k_indices, scores

def compare_predictions():
    print(f"\n{'='*20} Comparing Predictions {'='*20}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    test_data = snapshots[-1]
    
    # Load Models
    model_base = load_model(PureLightGCN, "lightgcn_best.pth", utils.num_users, utils.num_items, device)
    model_bi = load_model(BicliqueGCN, "biclique_gcn_best.pth", utils.num_users, utils.num_items, device)
    
    if not model_base or not model_bi:
        print("Please run training first to generate model files.")
        return

    # Prepare Graph Data for Inference
    adj_matrix = utils.build_adj_matrix(test_data).to(device)
    # For Biclique model, we need bicliques. Let's try to load existing ones or use empty if not found
    # In a real scenario, we should mine them. Here we assume they might exist from training or we mine quickly.
    # To be safe and fast, let's try to mine "solo_test" again (it should be cached)
    biclique_file = utils.run_msbe_mining(test_data, "solo_test", tau=2, epsilon=0.1) # Use tau=2 as per new config
    H_v, H_u = utils.parse_bicliques(biclique_file)
    H_v, H_u = H_v.to(device), H_u.to(device)
    
    # Find a user where Biclique wins
    test_users = list(set([u for u, _, _ in test_data]))
    random.shuffle(test_users)
    
    print("\nSearching for a case where Biclique GCN outperforms Baseline...")
    
    user_pos_items = {}
    for u, i, _ in test_data:
        if u not in user_pos_items: user_pos_items[u] = []
        user_pos_items[u].append(i)

    found_count = 0
    target_count = 10  # Find 10 cases
    
    print(f"\nSearching for {target_count} cases where Biclique GCN outperforms Baseline...\n")
    
    for u in test_users: # Iterate all users
        if found_count >= target_count: break
        
        u_idx = utils.u_map.get(u)
        if u_idx is None: continue
        
        true_items = [utils.v_map[i] for i in user_pos_items[u] if i in utils.v_map]
        if not true_items: continue
        
        # Get Recs
        recs_base, _ = get_recommendations(model_base, u_idx, adj_matrix, None, device, "lightgcn")
        recs_bi, _ = get_recommendations(model_bi, u_idx, adj_matrix, (H_v, H_u), device, "biclique_gcn")
        
        # Calculate Hits
        hits_base = len(set(recs_base) & set(true_items))
        hits_bi = len(set(recs_bi) & set(true_items))
        
        # Condition: Biclique wins significantly (e.g., hits >= 1 while base hits == 0, or hits_bi >= hits_base + 2)
        if hits_bi > hits_base and hits_bi >= 1:
            print(f"Case #{found_count+1} [User ID: {u}]")
            print(f"  True Items (Internal IDs): {true_items}")
            print(f"  Baseline Hits: {hits_base} | Top 5 Recs: {recs_base[:5]}")
            print(f"  Biclique Hits: {hits_bi} | Top 5 Recs: {recs_bi[:5]}")
            
            # Show which items were hit by Biclique but missed by Baseline
            hit_items_bi = set(recs_bi) & set(true_items)
            hit_items_base = set(recs_base) & set(true_items)
            unique_hits = hit_items_bi - hit_items_base
            print(f"  >>> Unique Hits by Biclique: {list(unique_hits)}")
            print("-" * 50)
            
            found_count += 1
    
    if found_count == 0:
        print("No significant difference found in random sample. Try training longer or adjusting parameters.")

if __name__ == "__main__":
    compare_predictions()
