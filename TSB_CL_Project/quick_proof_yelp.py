import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
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

def run_quick_proof():
    print("=== Running Quick Proof on Yelp2018 (Full Data) ===")
    
    # 1. Setup Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    # Path to the newly prepared Yelp2018 dataset
    DATA_PATH = os.path.join(CURRENT_DIR, "datasets", "yelp2018.txt")
    
    # Path to MSBE executable
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please run 'prepare_yelp2018.py' first.")
        return

    if not os.path.exists(MSBE_EXE):
        print(f"Warning: MSBE executable not found at {MSBE_EXE}")
        print("Biclique mining might fail if not cached.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data Path: {DATA_PATH}")

    # 2. Load Data
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    
    # 3. Split Data
    # Use Full Data (No Sampling)
    print("Using Full Dataset...")
    # random.shuffle(all_data) # Shuffle is good, but let's keep it consistent
    
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
    
    print(f"Bicliques loaded. H_u shape: {H_u.shape}, H_v shape: {H_v.shape}")
    
    # 5. Define Models
    print("Initializing Models...")
    # TSB-CL Model (Static Version)
    model_tsb = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    # Baseline (LightGCN equivalent)
    model_base = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device) 
    
    opt_tsb = optim.Adam(model_tsb.parameters(), lr=LR)
    opt_base = optim.Adam(model_base.parameters(), lr=LR)
    
    # 6. Evaluation Function
    test_users = list(set([x[0] for x in test_data]))
    # Evaluate on 1000 users for speed during training, full eval at end if needed
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

    def evaluate(model, use_biclique, target_users=None):
        if target_users is None:
            target_users = eval_users_subset
            
        model.eval()
        with torch.no_grad():
            if use_biclique:
                u_g, u_l, u_final, i_emb = model(adj_matrix, (H_v, H_u))
            else:
                dummy_Hv = torch.sparse_coo_tensor(size=(1, utils.num_items)).to(device)
                dummy_Hu = torch.sparse_coo_tensor(size=(utils.num_users, 1)).to(device)
                u_g, u_l, u_final, i_emb = model(adj_matrix, (dummy_Hv, dummy_Hu))
            
            # Use u_final for prediction
            u_emb = u_final
            
            hits = 0
            ndcgs = 0
            
            for u in target_users:
                if u not in test_user_ground_truth: continue
                gt = test_user_ground_truth[u]
                
                scores = torch.matmul(u_emb[u], i_emb.t())
                
                # Mask training items
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
                
            return hits / len(target_users), ndcgs / len(target_users)

    # 7. Training Loop
    print(f"Starting Training for {EPOCHS} epochs...")
    
    users_np = np.array([x[0] for x in train_data])
    items_np = np.array([x[1] for x in train_data])
    num_batches = len(train_data) // BATCH_SIZE
    
    # History for plotting
    history = {
        'epoch': [],
        'loss_tsb': [],
        'loss_base': [],
        'recall_tsb': [],
        'recall_base': [],
        'ndcg_tsb': [],
        'ndcg_base': []
    }

    best_recall_tsb = 0.0
    best_epoch_tsb = 0
    best_recall_base = 0.0
    best_epoch_base = 0
    
    # Learning Rate Schedulers
    scheduler_tsb = torch.optim.lr_scheduler.StepLR(opt_tsb, step_size=10, gamma=0.9)
    scheduler_base = torch.optim.lr_scheduler.StepLR(opt_base, step_size=10, gamma=0.9)

    for epoch in range(EPOCHS):
        # Shuffle
        perm = np.random.permutation(len(train_data))
        users_np = users_np[perm]
        items_np = items_np[perm]
        
        model_tsb.train()
        model_base.train()
        
        total_loss_tsb = 0
        total_loss_base = 0
        
        # Dynamic SSL Weight: Decay over time to prevent overfitting to the auxiliary task
        # Start at 0.1, decay slightly every epoch
        current_ssl_weight = 0.1 * (0.95 ** epoch)
        
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(train_data))
            
            batch_users = torch.LongTensor(users_np[start_idx:end_idx]).to(device)
            batch_pos = torch.LongTensor(items_np[start_idx:end_idx]).to(device)
            batch_neg = torch.randint(0, utils.num_items, (len(batch_users),)).to(device)
            
            # --- Train TSB-CL ---
            u_g, u_l, u_final, i_final = model_tsb(adj_matrix, (H_v, H_u))
            
            u_emb = u_final[batch_users]
            pos_emb = i_final[batch_pos]
            neg_emb = i_final[batch_neg]
            
            pos_scores = torch.sum(u_emb * pos_emb, dim=1)
            neg_scores = torch.sum(u_emb * neg_emb, dim=1)
            
            loss_bpr = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))
            
            # Contrastive Loss
            u_g_norm = torch.nn.functional.normalize(u_g[batch_users], dim=1)
            u_l_norm = torch.nn.functional.normalize(u_l[batch_users], dim=1)
            pos_sim = torch.sum(u_g_norm * u_l_norm, dim=1)
            loss_cl = -torch.mean(torch.log(torch.sigmoid(pos_sim / 0.2) + 1e-8))
            
            loss_tsb = loss_bpr + current_ssl_weight * loss_cl
            
            opt_tsb.zero_grad()
            loss_tsb.backward()
            opt_tsb.step()
            total_loss_tsb += loss_tsb.item()
            
            # --- Train Baseline ---
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
            total_loss_base += loss_base.item()
        
        # Step Schedulers
        scheduler_tsb.step()
        scheduler_base.step()
            
        # Record Loss
        history['epoch'].append(epoch + 1)
        history['loss_tsb'].append(total_loss_tsb)
        history['loss_base'].append(total_loss_base)
        
        # Evaluate every epoch to catch the peak
        r_tsb, n_tsb = evaluate(model_tsb, True)
        r_base, n_base = evaluate(model_base, False)
        
        history['recall_tsb'].append(r_tsb)
        history['recall_base'].append(r_base)
        history['ndcg_tsb'].append(n_tsb)
        history['ndcg_base'].append(n_base)
        
        # Track Best
        if r_tsb > best_recall_tsb:
            best_recall_tsb = r_tsb
            best_epoch_tsb = epoch + 1
            torch.save(model_tsb.state_dict(), 'tsb_cl_best_model.pth')
        
        if r_base > best_recall_base:
            best_recall_base = r_base
            best_epoch_base = epoch + 1
            torch.save(model_base.state_dict(), 'baseline_best_model.pth')
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss TSB: {total_loss_tsb:.4f} | Loss Base: {total_loss_base:.4f} | SSL W: {current_ssl_weight:.4f}")
        print(f"    TSB-CL   -> Recall: {r_tsb:.4f} | NDCG: {n_tsb:.4f} (Best Recall: {best_recall_tsb:.4f} at Ep {best_epoch_tsb})")
        print(f"    Baseline -> Recall: {r_base:.4f} | NDCG: {n_base:.4f} (Best Recall: {best_recall_base:.4f} at Ep {best_epoch_base})")

    print("\n=== Final Summary ===")
    print(f"TSB-CL   Best Recall: {best_recall_tsb:.4f} at Epoch {best_epoch_tsb}")
    print(f"Baseline Best Recall: {best_recall_base:.4f} at Epoch {best_epoch_base}")
    
    # Save Final Models
    torch.save(model_tsb.state_dict(), 'tsb_cl_final_model.pth')
    torch.save(model_base.state_dict(), 'baseline_final_model.pth')
    print("Models saved: tsb_cl_best_model.pth, baseline_best_model.pth, tsb_cl_final_model.pth, baseline_final_model.pth")
    
    print("\n=== Final Full Evaluation on Test Set (All Users) ===")
    # Use the final trained models for a full check
    r_tsb_final, n_tsb_final = evaluate(model_tsb, True, target_users=test_users)
    r_base_final, n_base_final = evaluate(model_base, False, target_users=test_users)
    
    print(f"Final TSB-CL   -> Recall: {r_tsb_final:.4f} | NDCG: {n_tsb_final:.4f}")
    print(f"Final Baseline -> Recall: {r_base_final:.4f} | NDCG: {n_base_final:.4f}")
    
    if r_tsb_final > r_base_final:
        print(">>> Winner: TSB-CL <<<")
    else:
        print(">>> Winner: Baseline <<<")
    
    # --- Plotting ---
    print("Generating Plots...")
    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['loss_tsb'], label='TSB-CL Loss')
    plt.plot(history['epoch'], history['loss_base'], label='Baseline Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot Recall
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['recall_tsb'], label='TSB-CL Recall@20')
    plt.plot(history['epoch'], history['recall_base'], label='Baseline Recall@20')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.title('Recall Comparison')
    plt.legend()
    plt.grid(True)
    if len(eval_epochs_x) > 0 and eval_epochs_x[-1] > EPOCHS:
         eval_epochs_x[-1] = EPOCHS
         
    plt.plot(eval_epochs_x, history['recall_tsb'], marker='o', label='TSB-CL Recall@20')
    plt.plot(eval_epochs_x, history['recall_base'], marker='x', label='Baseline Recall@20')
    plt.xlabel('Epochs')
    plt.ylabel('Recall@20')
    plt.title('Test Recall Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    print("Plots saved to training_results.png")

if __name__ == "__main__":
    run_quick_proof()
