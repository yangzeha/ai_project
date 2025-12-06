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

# --- Configuration ---
LR = 0.001
BATCH_SIZE = 2048
EPOCHS = 50  # Fast proof
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
        # This allows the model to use Biclique info for prediction without CL
        # We add u_local to u_global. 
        # Note: u_local might need scaling or a gate, but simple addition is a good start.
        u_final = u_global + u_local 
        
        return u_global, u_local, u_final, i_global

def run_direct_comparison():
    print("=== Running Comparison: TSB (Direct Fusion) vs LightGCN ===")
    
    # 1. Setup Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(CURRENT_DIR, "datasets", "yelp2018.txt")
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Load Data
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    
    # 3. Split Data
    split_idx = int(len(all_data) * 0.8)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]
    print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
    
    # 4. Prepare Graph Structures
    print("Building Adjacency Matrix...")
    adj_matrix = utils.build_adj_matrix(train_data).to(device)
    
    print("Mining Bicliques from Training Data...")
    biclique_file = utils.run_msbe_mining(train_data, "yelp_train_full", tau=TAU, epsilon=EPSILON)
    H_v, H_u = utils.parse_bicliques(biclique_file)
    H_v = H_v.to(device)
    H_u = H_u.to(device)
    
    # --- Biclique Analysis ---
    num_bicliques = H_u.shape[1]
    print(f"\n" + "="*40)
    print(f" [Biclique Analysis]")
    print(f" Found {num_bicliques} similar bicliques.")
    print(f" Parameters: tau={TAU}, epsilon={EPSILON}")
    
    if num_bicliques < 1000:
        print(" >> STATUS: TOO FEW")
        print(" >> ADVICE: The algorithm will likely fail to improve over LightGCN.")
        print(" >> ACTION: Increase epsilon (e.g., 0.3) or decrease tau (e.g., 1).")
    elif num_bicliques > 1000000:
        print(" >> STATUS: TOO MANY")
        print(" >> ADVICE: Training will be very slow. Noise might be high.")
        print(" >> ACTION: Decrease epsilon or increase tau.")
    else:
        print(" >> STATUS: GOOD RANGE")
        print(" >> ADVICE: Proceed with training.")
    print("="*40 + "\n")
    
    # 5. Define Models
    print("Initializing Models...")
    # Model A: TSB Direct (Fusion)
    model_tsb = TSB_Direct(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    
    # Model B: LightGCN (Base)
    model_base = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device) 
    
    opt_tsb = optim.Adam(model_tsb.parameters(), lr=LR)
    opt_base = optim.Adam(model_base.parameters(), lr=LR)
    
    # 6. Evaluation Setup
    test_users = list(set([x[0] for x in test_data]))
    if len(test_users) > 1000:
        eval_users_subset = random.sample(test_users, 1000)
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
            
            for u in eval_users_subset:
                if u not in test_user_ground_truth: continue
                gt = test_user_ground_truth[u]
                scores = torch.matmul(u_emb[u], i_emb.t())
                if u in train_user_items:
                    mask_indices = list(train_user_items[u])
                    scores[mask_indices] = -float('inf')
                _, indices = torch.topk(scores, 20)
                preds = indices.cpu().numpy()
                hit = len(set(preds) & gt)
                hits += hit / len(gt)
                
            return hits / len(eval_users_subset)

    # 7. Training Loop
    print(f"Starting Training for {EPOCHS} epochs...")
    
    users_np = np.array([x[0] for x in train_data])
    items_np = np.array([x[1] for x in train_data])
    num_batches = len(train_data) // BATCH_SIZE
    
    history = {'epoch': [], 'tsb': [], 'base': []}
    
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(train_data))
        users_np = users_np[perm]
        items_np = items_np[perm]
        
        model_tsb.train()
        model_base.train()
        
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(train_data))
            
            batch_users = torch.LongTensor(users_np[start_idx:end_idx]).to(device)
            batch_pos = torch.LongTensor(items_np[start_idx:end_idx]).to(device)
            batch_neg = torch.randint(0, utils.num_items, (len(batch_users),)).to(device)
            
            # --- 1. Train TSB Direct (BPR Only) ---
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
            
            # --- 2. Train Baseline (LightGCN) ---
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

        # Evaluate
        r_tsb = evaluate(model_tsb, True)
        r_base = evaluate(model_base, False)
        
        history['epoch'].append(epoch + 1)
        history['tsb'].append(r_tsb)
        history['base'].append(r_base)
        
        print(f"Ep {epoch+1} | TSB (Fusion): {r_tsb:.4f} | LightGCN: {r_base:.4f} | Diff: {r_tsb - r_base:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['tsb'], label='TSB (Direct Fusion)', marker='o')
    plt.plot(history['epoch'], history['base'], label='LightGCN', marker='x')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.title('TSB (Direct Fusion) vs LightGCN')
    plt.legend()
    plt.grid(True)
    plt.savefig('direct_comparison.png')
    print("Results saved to direct_comparison.png")

if __name__ == "__main__":
    run_direct_comparison()
