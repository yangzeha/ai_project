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
from solo_model.model_variants import FullTSBCL

def evaluate(model, test_data, utils, device, top_k=20, user_history_state=None):
    model.eval()
    adj_matrix = utils.build_adj_matrix(test_data).to(device)
    
    # Mine bicliques for test data
    biclique_file = utils.run_msbe_mining(test_data, "test_eval", tau=2, epsilon=0.1)
    H_v, H_u = utils.parse_bicliques(biclique_file)
    H_v, H_u = H_v.to(device), H_u.to(device)
    
    test_users = list(set([x[0] for x in test_data]))
    if len(test_users) > 1000:
        test_users = np.random.choice(test_users, 1000, replace=False)
        
    hits = 0
    ndcgs = 0
    
    with torch.no_grad():
        u_out, _, _, i_out = model(adj_matrix, (H_v, H_u), user_history_state)
        
        for u in test_users:
            ground_truth = set([x[1] for x in test_data if x[0] == u])
            if not ground_truth: continue
            
            u_emb = u_out[u].unsqueeze(0)
            scores = torch.mm(u_emb, i_out.t()).squeeze()
            
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

def train():
    # Config
    PROJECT_ROOT = os.path.dirname(current_dir)
    DATA_PATH = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", "datasets", "bi_github.txt")
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)
    
    EMBEDDING_DIM = 64
    LR = 0.001
    EPOCHS = 50
    NUM_SNAPSHOTS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    
    # Train on first N-1, Test on last
    train_snapshots = snapshots[:-1]
    test_data = snapshots[-1]
    
    model = FullTSBCL(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Start Training Full TSB-CL...")
    best_recall = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        user_history_state = None 
        
        for t, snapshot_data in enumerate(train_snapshots):
            # Mine Bicliques
            biclique_file = utils.run_msbe_mining(snapshot_data, f"train_{t}", tau=2, epsilon=0.1)
            H_v, H_u = utils.parse_bicliques(biclique_file)
            H_v, H_u = H_v.to(device), H_u.to(device)
            
            adj_matrix = utils.build_adj_matrix(snapshot_data).to(device)
            
            # Prepare batches
            users = list(set([x[0] for x in snapshot_data]))
            if len(users) > 2048:
                batch_users = np.random.choice(users, 2048, replace=False)
            else:
                batch_users = np.array(users)
                
            pos_items = []
            neg_items = []
            user_item_map = {u: set() for u in batch_users}
            for u, i, _ in snapshot_data:
                if u in user_item_map: user_item_map[u].add(i)
                
            valid_users = []
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
            
            # Forward
            u_final, u_local, new_state, i_global = model(adj_matrix, (H_v, H_u), user_history_state)
            
            loss, _, _ = model.calculate_loss(u_final, u_local, i_global, valid_users, pos_items, neg_items)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Update state for next snapshot
            user_history_state = new_state.detach()
            
        # Evaluate
        recall, ndcg = evaluate(model, test_data, utils, device, user_history_state=user_history_state)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Recall: {recall:.4f} | NDCG: {ndcg:.4f}")
        
        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), "full_tsbcl_best.pth")
            print("  > Saved best model")

if __name__ == "__main__":
    train()
