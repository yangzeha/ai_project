import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import DataUtils
import random
from solo_model.model_variants import PureLightGCN, BicliqueGCN, BicliqueCL, FullTSBCL

# --- Evaluation Logic ---

def evaluate_model(model, test_data, utils, device, model_type, biclique_matrices, top_k=20, user_history_state=None):
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
            # Pass the history state here
            u_out, _, _, i_out = model(adj_matrix, (H_v, H_u), user_history_state)
        
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
            
        # Warmup for FullTSBCL (Compute RNN State)
        user_history_state = None
        if model_type == "FullTSBCL":
            # print(f"  > Warming up RNN state for {model_name}...")
            train_snapshots = snapshots[:-1]
            with torch.no_grad():
                for t, snapshot_data in enumerate(train_snapshots):
                    # Mine/Load Bicliques for history snapshots
                    # Note: Using same tau/eps as training is ideal. Assuming tau=2, eps=0.1 here.
                    biclique_file_warm = utils.run_msbe_mining(snapshot_data, f"warmup_{t}", tau=2, epsilon=0.1)
                    H_v_warm, H_u_warm = utils.parse_bicliques(biclique_file_warm)
                    H_v_warm, H_u_warm = H_v_warm.to(device), H_u_warm.to(device)
                    
                    adj_warm = utils.build_adj_matrix(snapshot_data).to(device)
                    
                    # Forward pass to update state
                    _, _, new_state, _ = model(adj_warm, (H_v_warm, H_u_warm), user_history_state)
                    user_history_state = new_state.detach()

        # Evaluate
        recall, ndcg = evaluate_model(model, test_data, utils, device, model_type, (H_v, H_u), user_history_state=user_history_state)
        
        print(f"{model_name:<25} | {recall:.4f}     | {ndcg:.4f}")
        
    print("="*50)

if __name__ == "__main__":
    main()
