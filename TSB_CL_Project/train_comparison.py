import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from model import TSB_CL
from data_utils import DataUtils
import random
import matplotlib.pyplot as plt
import numpy as np
import time

def evaluate(model, test_data, utils, device, top_k=20):
    model.eval()
    
    # Build test graph
    # For simplicity in this demo, we use the test snapshot itself to build adjacency
    # In a real scenario, we should use the training graph + test interactions
    adj_matrix = utils.build_adj_matrix(test_data).to(device)
    
    # For bicliques in test, we can either mine them or pass empty
    # Here we pass empty to focus on the model's learned embeddings
    H_v = torch.sparse_coo_tensor(size=(1, utils.num_items)).to(device)
    H_u = torch.sparse_coo_tensor(size=(utils.num_users, 1)).to(device)
    
    test_users = list(set([x[0] for x in test_data]))
    # Sample for speed
    if len(test_users) > 500:
        test_users = np.random.choice(test_users, 500, replace=False)
        
    hits = 0
    ndcgs = 0
    
    with torch.no_grad():
        # Forward pass
        # Note: We are not passing history state here for simplicity
        u_global, _, _, i_global = model(adj_matrix, (H_v, H_u), None)
        all_item_emb = i_global
        
        for u in test_users:
            ground_truth = set([x[1] for x in test_data if x[0] == u])
            if not ground_truth: continue
            
            u_emb = u_global[u].unsqueeze(0)
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

def run_experiment(enable_biclique, epochs=5, tau=2, epsilon=0.1):
    # --- Config ---
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    DATA_PATH = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", "datasets", "bi_github.txt")
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)
    EMBEDDING_DIM = 64
    LR = 0.001
    BATCH_SIZE = 2048
    NUM_SNAPSHOTS = 1 # Use full dataset to ensure bicliques are found
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    
    # If NUM_SNAPSHOTS = 1, we can't split train/test by snapshots easily
    # So we just use the same data for train and test for this demo
    # or split manually
    if NUM_SNAPSHOTS == 1:
        snapshots = [all_data]
        train_snapshots = [all_data]
        test_data = all_data # Evaluating on training set (just for demo)
    else:
        snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
        train_snapshots = snapshots[:-1]
        test_data = snapshots[-1]
    
    model = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    metrics = {
        'loss': [],
        'recall': [],
        'ndcg': [],
        'time': []
    }
    
    user_history_state = None
    
    print(f"--- Starting Experiment: Biclique={'Enabled' if enable_biclique else 'Disabled'} ---")
    
    total_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        # Train on snapshots
        for t, snapshot_data in enumerate(train_snapshots):
            # 1. Mining
            if enable_biclique:
                # Use proven parameters
                biclique_file = utils.run_msbe_mining(snapshot_data, f"exp_{t}", tau=tau, epsilon=epsilon)
                H_v, H_u = utils.parse_bicliques(biclique_file)
            else:
                # Empty bicliques
                H_v = torch.sparse_coo_tensor(size=(1, utils.num_items))
                H_u = torch.sparse_coo_tensor(size=(utils.num_users, 1))
            
            H_v = H_v.to(device)
            H_u = H_u.to(device)
            
            # 2. Adjacency
            adj_matrix = utils.build_adj_matrix(snapshot_data).to(device)
            
            # 3. Training
            pos_interactions = [(u, i) for u, i, _ in snapshot_data]
            random.shuffle(pos_interactions)
            
            model.train()
            for i in range(0, len(pos_interactions), BATCH_SIZE):
                batch_samples = pos_interactions[i:i+BATCH_SIZE]
                optimizer.zero_grad()
                
                users = torch.LongTensor([x[0] for x in batch_samples]).to(device)
                pos_items = torch.LongTensor([x[1] for x in batch_samples]).to(device)
                neg_items = torch.randint(0, utils.num_items, (len(users),)).to(device)
                
                if user_history_state is not None:
                    current_history_state = user_history_state.detach().to(device)
                else:
                    current_history_state = None
                    
                # Forward pass
                u_global, u_local, new_state, i_global = model(
                    adj_matrix, (H_v, H_u), current_history_state
                )
                
                # Calculate loss
                loss, _, _ = model.calculate_loss(
                    u_global, u_local, i_global,
                    users, pos_items, neg_items
                )
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Detach state for next step to avoid backprop through time indefinitely
                if new_state is not None:
                    user_history_state = new_state.detach()
        
        epoch_time = time.time() - epoch_start_time
        metrics['time'].append(epoch_time)
        metrics['loss'].append(epoch_loss)
        
        # Evaluate
        recall, ndcg = evaluate(model, test_data, utils, device)
        metrics['recall'].append(recall)
        metrics['ndcg'].append(ndcg)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Recall@20: {recall:.4f} | NDCG@20: {ndcg:.4f} | Time: {epoch_time:.2f}s")
        
    total_time = time.time() - total_start_time
    print(f"Total Training Time: {total_time:.2f}s")
    
    return metrics

def plot_comparison(baseline_metrics, enhanced_metrics):
    epochs = range(1, len(baseline_metrics['loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    # 1. Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, baseline_metrics['loss'], 'b--', label='Baseline (No Biclique)')
    plt.plot(epochs, enhanced_metrics['loss'], 'r-', label='TSB-CL (With Biclique)')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Recall
    plt.subplot(1, 3, 2)
    plt.plot(epochs, baseline_metrics['recall'], 'b--', label='Baseline')
    plt.plot(epochs, enhanced_metrics['recall'], 'r-', label='TSB-CL')
    plt.title('Recall@20')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    # 3. Time
    plt.subplot(1, 3, 3)
    plt.bar(['Baseline', 'TSB-CL'], [sum(baseline_metrics['time']), sum(enhanced_metrics['time'])], color=['blue', 'red'])
    plt.title('Total Training Time')
    plt.ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig('comparison_results.png')
    print("Comparison plot saved to comparison_results.png")

if __name__ == "__main__":
    # Run Baseline (No Biclique)
    print("\n=== Running Baseline (LightGCN only) ===")
    baseline_res = run_experiment(enable_biclique=False, epochs=10)
    
    # Run Enhanced (TSB-CL)
    # Using tau=3, epsilon=0.1 for better performance/coverage balance
    print("\n=== Running TSB-CL (Biclique Enhanced) ===")
    enhanced_res = run_experiment(enable_biclique=True, epochs=10, tau=3, epsilon=0.1)
    
    plot_comparison(baseline_res, enhanced_res)
