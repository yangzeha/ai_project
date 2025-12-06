import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from model import TSB_CL
from data_utils import DataUtils
import sys

# --- Configuration ---
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 50 
EMBEDDING_DIM = 64
TAU = 2
EPSILON = 0.1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(2024)

# --- Define TSB_Direct Model (Fusion instead of CL) ---
class TSB_Direct(TSB_CL):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(TSB_Direct, self).__init__(num_users, num_items, embedding_dim)
    
    def forward(self, adj_matrix, biclique_matrices):
        u_emb = self.user_embedding.weight
        i_emb = self.item_embedding.weight
        
        # 1. Global View (LightGCN)
        u_global, i_global = self.global_encoder(u_emb, i_emb, adj_matrix)
        
        # 2. Local View (Biclique)
        u_local = self.local_encoder(u_emb, i_emb, biclique_matrices)
        
        # 3. FUSION: Directly combine Global and Local views
        u_final = u_global + u_local 
        
        return u_global, u_local, u_final, i_global

def run_direct_comparison():
    print("=== Running Comparison: TSB (Direct Fusion) vs LightGCN ===")
    
    # 1. Setup Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(CURRENT_DIR, "datasets", "yelp2018.txt")
    
    # Check if dataset exists, if not, try to download
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}. Attempting to download...")
        try:
            # Add current dir to path to import prepare_yelp2018
            sys.path.append(CURRENT_DIR)
            from prepare_yelp2018 import download_yelp2018
            download_yelp2018()
        except ImportError:
            print("Error: prepare_yelp2018.py not found or failed.")
            # Fallback: try running it as a subprocess
            os.system(f"python {os.path.join(CURRENT_DIR, 'prepare_yelp2018.py')}")

    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load Data
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    
    # 3. Split Data (4:1 Ratio -> 80% Train, 20% Test)
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    print(f"Total interactions: {len(all_data)}")
    print(f"Train size (80%): {len(train_data)}")
    print(f"Test size (20%): {len(test_data)}")
    
    # 4. Prepare Graph Structures
    print("Building Adjacency Matrix...")
    adj_matrix = utils.build_adj_matrix(train_data).to(device)
    
    print("Mining Bicliques from Training Data...")
    biclique_file = utils.run_msbe_mining(train_data, "yelp_train_full", tau=TAU, epsilon=EPSILON)
    H_v, H_u = utils.parse_bicliques(biclique_file)
    H_v = H_v.to(device)
    H_u = H_u.to(device)
    
    num_bicliques = H_u.shape[1]
    print(f"Found {num_bicliques} bicliques.")

    # 5. Define Models
    print("Initializing Models...")
    model_tsb = TSB_Direct(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    model_base = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device) 
    
    opt_tsb = optim.Adam(model_tsb.parameters(), lr=LR)
    opt_base = optim.Adam(model_base.parameters(), lr=LR)
    
    # 6. Evaluation Setup
    test_users = list(set([x[0] for x in test_data]))
    # Evaluate on a subset for speed, but large enough to be representative
    if len(test_users) > 2000:
        eval_users_subset = random.sample(test_users, 2000)
    else:
        eval_users_subset = test_users
        
    test_user_ground_truth = {}
    for u, i, _ in test_data:
        if u not in test_user_ground_truth: test_user_ground_truth[u] = set()
        test_user_ground_truth[u].add(i)
        
    train_user_items = {}
    for u, i, _ in train_data:
        if u not in train_user_items: train_user_items[u] = set()
        train_user_items[u].add(i)

    def evaluate(model, use_biclique):
        model.eval()
        with torch.no_grad():
            if use_biclique:
                u_g, u_l, u_final, i_emb = model(adj_matrix, (H_v, H_u))
            else:
                dummy_Hv = torch.sparse_coo_tensor(size=(1, utils.num_items)).to(device)
                dummy_Hu = torch.sparse_coo_tensor(size=(utils.num_users, 1)).to(device)
                u_g, u_l, u_final, i_emb = model(adj_matrix, (dummy_Hv, dummy_Hu))
            
            u_emb = u_final
            hits = 0
            ndcgs = 0
            
            for u in eval_users_subset:
                if u not in test_user_ground_truth: continue
                gt = test_user_ground_truth[u]
                
                scores = torch.matmul(u_emb[u], i_emb.t())
                if u in train_user_items:
                    mask_indices = list(train_user_items[u])
                    scores[mask_indices] = -float('inf')
                
                _, indices = torch.topk(scores, 20)
                preds = indices.cpu().numpy()
                
                # Recall
                hit = len(set(preds) & gt)
                hits += hit / len(gt)
                
                # NDCG
                dcg = 0
                idcg = 0
                for i, item in enumerate(preds):
                    if item in gt:
                        dcg += 1.0 / np.log2(i + 2)
                for i in range(min(len(gt), 20)):
                    idcg += 1.0 / np.log2(i + 2)
                ndcgs += dcg / idcg if idcg > 0 else 0
                
            return hits / len(eval_users_subset), ndcgs / len(eval_users_subset)

    # 7. Training Loop
    print(f"Starting Training for {EPOCHS} epochs...")
    
    users_np = np.array([x[0] for x in train_data])
    items_np = np.array([x[1] for x in train_data])
    num_batches = len(train_data) // BATCH_SIZE
    
    history = {
        'epoch': [], 
        'loss_tsb': [], 'loss_base': [],
        'recall_tsb': [], 'recall_base': [],
        'ndcg_tsb': [], 'ndcg_base': []
    }
    
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(train_data))
        users_np = users_np[perm]
        items_np = items_np[perm]
        
        model_tsb.train()
        model_base.train()
        
        loss_sum_tsb = 0
        loss_sum_base = 0
        
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(train_data))
            
            batch_users = torch.LongTensor(users_np[start_idx:end_idx]).to(device)
            batch_pos = torch.LongTensor(items_np[start_idx:end_idx]).to(device)
            batch_neg = torch.randint(0, utils.num_items, (len(batch_users),)).to(device)
            
            # --- Train TSB ---
            u_g, u_l, u_final, i_final = model_tsb(adj_matrix, (H_v, H_u))
            u_emb = u_final[batch_users]
            pos_emb = i_final[batch_pos]
            neg_emb = i_final[batch_neg]
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            loss_tsb = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
            
            opt_tsb.zero_grad()
            loss_tsb.backward()
            opt_tsb.step()
            loss_sum_tsb += loss_tsb.item()
            
            # --- Train Base ---
            dummy_Hv = torch.sparse_coo_tensor(size=(1, utils.num_items)).to(device)
            dummy_Hu = torch.sparse_coo_tensor(size=(utils.num_users, 1)).to(device)
            u_g_b, _, u_final_b, i_final_b = model_base(adj_matrix, (dummy_Hv, dummy_Hu))
            u_emb_b = u_final_b[batch_users]
            pos_emb_b = i_final_b[batch_pos]
            neg_emb_b = i_final_b[batch_neg]
            pos_scores_b = torch.sum(u_emb_b * pos_emb_b, dim=1)
            neg_scores_b = torch.sum(u_emb_b * neg_emb_b, dim=1)
            loss_base = -torch.mean(torch.log(torch.sigmoid(pos_scores_b - neg_scores_b) + 1e-8))
            
            opt_base.zero_grad()
            loss_base.backward()
            opt_base.step()
            loss_sum_base += loss_base.item()

        # Evaluate
        r_tsb, n_tsb = evaluate(model_tsb, True)
        r_base, n_base = evaluate(model_base, False)
        
        history['epoch'].append(epoch + 1)
        history['loss_tsb'].append(loss_sum_tsb / num_batches)
        history['loss_base'].append(loss_sum_base / num_batches)
        history['recall_tsb'].append(r_tsb)
        history['recall_base'].append(r_base)
        history['ndcg_tsb'].append(n_tsb)
        history['ndcg_base'].append(n_base)
        
        print(f"Ep {epoch+1} | TSB: R={r_tsb:.4f} N={n_tsb:.4f} L={loss_sum_tsb/num_batches:.4f} | Base: R={r_base:.4f} N={n_base:.4f} L={loss_sum_base/num_batches:.4f}")

    # Plotting
    plt.figure(figsize=(18, 6))
    
    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(history['epoch'], history['loss_tsb'], label='TSB (Fusion)', marker='o')
    plt.plot(history['epoch'], history['loss_base'], label='LightGCN', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # Recall
    plt.subplot(1, 3, 2)
    plt.plot(history['epoch'], history['recall_tsb'], label='TSB (Fusion)', marker='o')
    plt.plot(history['epoch'], history['recall_base'], label='LightGCN', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.title('Recall@20')
    plt.legend()
    plt.grid(True)
    
    # NDCG
    plt.subplot(1, 3, 3)
    plt.plot(history['epoch'], history['ndcg_tsb'], label='TSB (Fusion)', marker='o')
    plt.plot(history['epoch'], history['ndcg_base'], label='LightGCN', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('NDCG@20')
    plt.title('NDCG@20')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("Results saved to comparison_results.png")

if __name__ == "__main__":
    run_direct_comparison()
