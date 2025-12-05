import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import torch.optim as optim
import random
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到路径以导入 data_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_utils import DataUtils

# 全局配置
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'datasets', 'bi_github.txt'))
MSBE_EXE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'msbe'))
if os.name == 'nt':
    MSBE_EXE += ".exe"

BATCH_SIZE = 2048
EMBEDDING_DIM = 64
LR = 0.001
NUM_SNAPSHOTS = 4
TOP_K = 20

def calculate_metrics(u_emb, i_emb, test_data, k=20):
    # u_emb, i_emb are tensors
    u_emb = u_emb.cpu().numpy()
    i_emb = i_emb.cpu().numpy()
    
    test_users = list(set([u for u, _, _ in test_data]))
    # Sample 1000 users for speed
    if len(test_users) > 1000:
        sample_users = random.sample(test_users, 1000)
    else:
        sample_users = test_users
    
    hits = 0
    ndcg = 0
    
    # Ground Truth
    user_pos_items = {}
    for u, i, _ in test_data:
        if u not in user_pos_items:
            user_pos_items[u] = []
        user_pos_items[u].append(i)
        
    for u in sample_users:
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

def run_training(model_class, model_name, model_type="full", epochs=5, tau=3, epsilon=0.1):
    print(f"\n{'='*20} Running {model_name} {'='*20}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    train_snapshots = snapshots[:-1]
    test_data = snapshots[-1]
    
    model = model_class(utils.num_users, utils.num_items, EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    metrics_history = {'loss': [], 'recall': [], 'ndcg': []}
    best_recall = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        steps = 0
        
        # Reset history state at the start of each epoch for sequential training
        user_history_state = None

        for t, snapshot_data in enumerate(train_snapshots):
            next_history_state = None
            
            # 1. Mining
            if model_type != "lightgcn":
                biclique_file = utils.run_msbe_mining(snapshot_data, f"solo_{t}", tau=tau, epsilon=epsilon)
                H_v, H_u = utils.parse_bicliques(biclique_file)
                H_v, H_u = H_v.to(device), H_u.to(device)
            else:
                H_v, H_u = None, None
            
            # 2. Adj
            adj_matrix = utils.build_adj_matrix(snapshot_data).to(device)
            
            # 3. Train
            pos_interactions = [(u, i) for u, i, _ in snapshot_data]
            random.shuffle(pos_interactions)
            
            model.train()
            for i in range(0, len(pos_interactions), BATCH_SIZE):
                batch = pos_interactions[i:i+BATCH_SIZE]
                optimizer.zero_grad()
                
                users = torch.LongTensor([x[0] for x in batch]).to(device)
                pos_items = torch.LongTensor([x[1] for x in batch]).to(device)
                neg_items = torch.randint(0, utils.num_items, (len(users),)).to(device)
                
                # Forward & Loss
                if model_type == "lightgcn":
                    u_out, i_out = model(adj_matrix, (H_v, H_u), user_history_state)
                    loss, _, _ = model.calculate_loss(u_out, i_out, users, pos_items, neg_items)
                elif model_type == "biclique_gcn":
                    u_out, i_out = model(adj_matrix, (H_v, H_u), user_history_state)
                    loss, _, _ = model.calculate_loss(u_out, i_out, users, pos_items, neg_items)
                elif model_type == "biclique_cl":
                    u_global, u_local, i_global = model(adj_matrix, (H_v, H_u), user_history_state)
                    loss, _, _ = model.calculate_loss(u_global, u_local, i_global, users, pos_items, neg_items)
                elif model_type == "full":
                    u_global, u_local, new_state, i_global = model(adj_matrix, (H_v, H_u), user_history_state)
                    loss, _, _ = model.calculate_loss(u_global, u_local, i_global, users, pos_items, neg_items)
                    next_history_state = new_state.detach()

                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            
            # Update history state after each snapshot
            if next_history_state is not None:
                user_history_state = next_history_state.detach()
        
        avg_loss = epoch_loss / steps if steps > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f} ({time.time()-start_time:.2f}s)")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Prepare test data
            if model_type != "lightgcn":
                biclique_file = utils.run_msbe_mining(test_data, "solo_test", tau=tau, epsilon=epsilon)
                H_v_test, H_u_test = utils.parse_bicliques(biclique_file)
                H_v_test, H_u_test = H_v_test.to(device), H_u_test.to(device)
            else:
                H_v_test, H_u_test = None, None
            
            test_adj = utils.build_adj_matrix(test_data).to(device)
            
            if model_type == "lightgcn":
                u_emb, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
            elif model_type == "biclique_gcn":
                u_emb, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
            elif model_type == "biclique_cl":
                u_emb, _, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
            elif model_type == "full":
                u_emb, _, _, i_emb = model(test_adj, (H_v_test, H_u_test), user_history_state)
            
            recall, ndcg = calculate_metrics(u_emb, i_emb, test_data, k=20)
            print(f"  Test Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")
            metrics_history['recall'].append(recall)
            metrics_history['ndcg'].append(ndcg)
            metrics_history['loss'].append(avg_loss)
            
            # Save Best Model
            if recall > best_recall:
                best_recall = recall
                save_path = f"{model_type}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"  >>> New Best Model Saved to {save_path} (Recall: {best_recall:.4f})")

    return metrics_history
