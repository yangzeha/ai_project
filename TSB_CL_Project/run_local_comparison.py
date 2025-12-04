import os
import sys
import subprocess
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import numpy as np
from data_utils import DataUtils
from model import TSB_CL
import random

# 1. Compile MSBE
print(">>> Compiling MSBE with -D_PrintResults_...")
# Get absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
msbe_dir = os.path.join(project_root, "Similar-Biclique-Idx-main")
msbe_exe = os.path.join(msbe_dir, "msbe.exe")
cpp_src = os.path.join(msbe_dir, "main.cpp")

# Check if g++ is available
try:
    subprocess.run(["g++", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    print("Error: g++ not found. Please install MinGW or ensure g++ is in PATH.")
    sys.exit(1)

# Compile command
cmd = ["g++", "-O3", cpp_src, "-o", msbe_exe, "-D_PrintResults_"]
try:
    subprocess.run(cmd, check=True)
    print(f"✅ Compilation successful: {msbe_exe}")
except subprocess.CalledProcessError as e:
    print("❌ Compilation failed.")
    sys.exit(1)

# 2. Define Experiment
def evaluate_model(model, test_data, utils, device):
    model.eval()
    hits = 0
    # Sample test users
    test_users = list(set([x[0] for x in test_data]))
    if len(test_users) > 200: test_users = random.sample(test_users, 200)
    
    user_items = {}
    for u, i, _ in test_data:
        if u not in user_items: user_items[u] = []
        user_items[u].append(i)
        
    # Build test adj (simplified)
    adj = utils.build_adj_matrix(test_data).to(device)
    # Empty bicliques for test
    H_v = torch.sparse_coo_tensor(size=(0, utils.num_items), device=device)
    H_u = torch.sparse_coo_tensor(size=(utils.num_users, 0), device=device)
    
    with torch.no_grad():
        u_g, _, _, i_g = model(adj, (H_v, H_u), None)
        
        for u in test_users:
            if u >= len(u_g): continue
            target = user_items[u][0]
            
            # Predict
            u_emb = u_g[u]
            scores = torch.matmul(i_g, u_emb)
            
            _, indices = torch.topk(scores, 10)
            recs = indices.cpu().numpy().tolist()
            
            if target in recs:
                hits += 1
                
    return hits / len(test_users)

def run_experiment(mode, utils, device, epochs=3): # Reduced epochs for speed
    print(f"\n🚀 Running Experiment: {mode}")
    
    # Load Data
    all_data = utils.load_data()
    train_data = all_data[:int(len(all_data)*0.8)]
    test_data = all_data[int(len(all_data)*0.8):]
    snapshots = utils.split_snapshots(train_data, 3)
    
    # Initialize Model
    model = TSB_CL(utils.num_users, utils.num_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    metrics = {'hr': [], 'time': []}
    total_time = 0
    user_history_state = None
    
    for t, snapshot in enumerate(snapshots):
        start = time.time()
        
        # Biclique Mining
        if mode == "Enhanced (TSB-CL)":
            # Use looser parameters to ensure we find something
            # tau=2, epsilon=0.1 are generally good starting points
            biclique_file = utils.run_msbe_mining(snapshot, t, tau=2, epsilon=0.1)
            H_v, H_u = utils.parse_bicliques(biclique_file)
        else:
            # Baseline: Empty bicliques
            H_v = torch.sparse_coo_tensor(size=(0, utils.num_items), device=device)
            H_u = torch.sparse_coo_tensor(size=(utils.num_users, 0), device=device)
            
        H_v = H_v.to(device)
        H_u = H_u.to(device)
        adj = utils.build_adj_matrix(snapshot).to(device)
        
        # Training Loop (Simplified for snapshot)
        model.train()
        # Just run a few batches to simulate training
        pos_items = [x[1] for x in snapshot]
        users = [x[0] for x in snapshot]
        
        # Batching
        batch_size = 1024
        num_batches = (len(users) + batch_size - 1) // batch_size
        
        for _ in range(epochs):
            perm = np.random.permutation(len(users))
            users_np = np.array(users)[perm]
            pos_items_np = np.array(pos_items)[perm]
            
            for b in range(num_batches):
                batch_idx = np.arange(b*batch_size, min((b+1)*batch_size, len(users)))
                u_batch = torch.LongTensor(users_np[batch_idx]).to(device)
                pos_batch = torch.LongTensor(pos_items_np[batch_idx]).to(device)
                neg_batch = torch.randint(0, utils.num_items, (len(batch_idx),)).to(device)
                
                optimizer.zero_grad()
                
                # Forward
                u_global, u_local, new_state, i_global = model(adj, (H_v, H_u), user_history_state)
                
                # Loss
                loss, bpr, cl = model.calculate_loss(u_global, u_local, i_global, u_batch, pos_batch, neg_batch)
                
                loss.backward()
                optimizer.step()
                
                # Detach state to prevent backprop through time indefinitely
                user_history_state = new_state.detach()

        elapsed = time.time() - start
        total_time += elapsed
        
        # Evaluate
        hr = evaluate_model(model, test_data, utils, device)
        metrics['hr'].append(hr)
        metrics['time'].append(total_time)
        print(f"  Snapshot {t+1}: Time={elapsed:.1f}s | HR@10={hr:.4f}")
        
    return metrics

# 3. Main Execution
if __name__ == "__main__":
    # Path to data
    data_path = os.path.join(msbe_dir, "datasets", "bi_github.txt")
    
    utils = DataUtils(data_path, msbe_exe)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    res_base = run_experiment("Baseline", utils, device)
    res_enh = run_experiment("Enhanced (TSB-CL)", utils, device)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(res_base['hr'], label='Baseline', marker='o', linestyle='--')
    plt.plot(res_enh['hr'], label='TSB-CL (Enhanced)', marker='s', linewidth=2)
    plt.title('Performance Comparison: Baseline vs TSB-CL')
    plt.xlabel('Snapshot')
    plt.ylabel('HR@10')
    plt.legend()
    plt.grid(True)
    
    output_plot = os.path.join(current_dir, "comparison_result.png")
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()
