import os
import sys
import time
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from data_utils import DataUtils
from solo_model.model_variants import FullTSBCL, PureLightGCN

def evaluate(model, test_data, utils, device, model_type, top_k=20, user_history_state=None):
    model.eval()
    adj_matrix = utils.build_adj_matrix(test_data).to(device)
    
    # Mine bicliques for test data (only needed for TSB-CL)
    H_v, H_u = None, None
    if model_type == "FullTSBCL":
        biclique_file = utils.run_msbe_mining(test_data, "test_eval", tau=2, epsilon=0.1)
        H_v, H_u = utils.parse_bicliques(biclique_file)
        H_v, H_u = H_v.to(device), H_u.to(device)
    
    test_users = list(set([x[0] for x in test_data]))
    if len(test_users) > 1000:
        test_users = np.random.choice(test_users, 1000, replace=False)
        
    hits = 0
    
    with torch.no_grad():
        if model_type == "FullTSBCL":
            u_out, _, _, i_out = model(adj_matrix, (H_v, H_u), user_history_state)
        else:
            u_out, i_out = model(adj_matrix)
        
        for u in test_users:
            ground_truth = set([x[1] for x in test_data if x[0] == u])
            if not ground_truth: continue
            
            u_emb = u_out[u].unsqueeze(0)
            scores = torch.mm(u_emb, i_out.t()).squeeze()
            
            _, indices = torch.topk(scores, top_k)
            pred_items = indices.cpu().numpy()
            
            hit = 0
            for i, item in enumerate(pred_items):
                if item in ground_truth:
                    hit += 1
            hits += hit / len(ground_truth)
            
    return hits / len(test_users)

def run_experiment(model_class, model_name, epochs=30):
    # Config
    PROJECT_ROOT = os.path.dirname(current_dir)
    DATA_PATH = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", "datasets", "bi_github.txt")
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)
    
    EMBEDDING_DIM = 64
    LR = 0.001
    NUM_SNAPSHOTS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Running {model_name} ===")
    
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    
    train_snapshots = snapshots[:-1]
    test_data = snapshots[-1]
    
    model = model_class(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_recall = 0
    recalls = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        user_history_state = None 
        
        for t, snapshot_data in enumerate(train_snapshots):
            adj_matrix = utils.build_adj_matrix(snapshot_data).to(device)
            
            # Prepare Bicliques (Only for TSB-CL)
            H_v, H_u = None, None
            if model_name == "FullTSBCL":
                biclique_file = utils.run_msbe_mining(snapshot_data, f"train_{t}", tau=2, epsilon=0.1)
                H_v, H_u = utils.parse_bicliques(biclique_file)
                H_v, H_u = H_v.to(device), H_u.to(device)
            
            # Prepare batches (Simplified)
            users = list(set([x[0] for x in snapshot_data]))
            if len(users) > 2048: batch_users = np.random.choice(users, 2048, replace=False)
            else: batch_users = np.array(users)
                
            pos_items, neg_items, valid_users = [], [], []
            user_item_map = {u: set() for u in batch_users}
            for u, i, _ in snapshot_data:
                if u in user_item_map: user_item_map[u].add(i)
            
            for u in batch_users:
                if not user_item_map[u]: continue
                pos = np.random.choice(list(user_item_map[u]))
                while True:
                    neg = np.random.randint(0, utils.num_items)
                    if neg not in user_item_map[u]: break
                valid_users.append(u)
                pos_items.append(pos)
                neg_items.append(neg)
            
            if not valid_users: continue
            valid_users = torch.LongTensor(valid_users).to(device)
            pos_items = torch.LongTensor(pos_items).to(device)
            neg_items = torch.LongTensor(neg_items).to(device)
            
            optimizer.zero_grad()
            
            if model_name == "FullTSBCL":
                u_final, u_local, new_state, i_global = model(adj_matrix, (H_v, H_u), user_history_state)
                loss, _, _ = model.calculate_loss(u_final, u_local, i_global, valid_users, pos_items, neg_items)
                user_history_state = new_state.detach()
            else:
                u_global, i_global = model(adj_matrix)
                loss, _, _ = model.calculate_loss(u_global, i_global, valid_users, pos_items, neg_items)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Evaluate
        recall = evaluate(model, test_data, utils, device, model_name, user_history_state=user_history_state)
        recalls.append(recall)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Recall: {recall:.4f}")
        
        if recall > best_recall:
            best_recall = recall
            
    return best_recall, recalls

if __name__ == "__main__":
    # 1. Run Baseline
    base_best, base_curve = run_experiment(PureLightGCN, "PureLightGCN", epochs=30)
    
    # 2. Run TSB-CL
    tsb_best, tsb_curve = run_experiment(FullTSBCL, "FullTSBCL", epochs=30)
    
    print("\n" + "="*30)
    print(f"Baseline Best Recall: {base_best:.4f}")
    print(f"TSB-CL   Best Recall: {tsb_best:.4f}")
    print("="*30)
    
    # Plot
    plt.plot(base_curve, label='Baseline (LightGCN)')
    plt.plot(tsb_curve, label='Full TSB-CL')
    plt.legend()
    plt.title('Recall Comparison')
    plt.savefig('comparison.png')
