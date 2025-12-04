import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
from model import TSB_CL
from data_utils import DataUtils
import random
import matplotlib.pyplot as plt
import numpy as np

def train():
    # --- 配置参数 ---
    # Get the directory of the current script (TSB_CL_Project)
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    
    DATA_PATH = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", "datasets", "bi_github.txt")
    
    # Determine executable name based on OS
    exe_name = "msbe.exe" if os.name == 'nt' else "msbe"
    MSBE_EXE = os.path.join(PROJECT_ROOT, "Similar-Biclique-Idx-main", exe_name)

    EMBEDDING_DIM = 64
    LR = 0.001
    EPOCHS = 20  # 增加训练轮数到 20，以便观察 Loss 下降曲线
    BATCH_SIZE = 2048 
    NUM_SNAPSHOTS = 1 # 不切分，使用全量数据
    
    # --- GPU 设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 数据准备 ---
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    
    # 使用全量数据作为训练集
    train_snapshots = snapshots
    print(f"Total Snapshots: {NUM_SNAPSHOTS}. Training on full dataset (No Split).")
    
    # --- 2. 模型初始化 ---
    model = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM)
    model = model.to(device) # 移动模型到 GPU
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("Start Training TSB-CL...")
    
    # 用户历史状态 (GRU Hidden State)
    user_history_state = None
    
    # 记录训练过程中的指标
    history = {
        'loss': [],
        'bpr_loss': [],
        'cl_loss': [],
        'accuracy': []
    }
    
    # --- 3. 时序训练循环 ---
    for t, snapshot_data in enumerate(train_snapshots):
        print(f"\n=== Time Snapshot {t+1}/{len(train_snapshots)} (Total Progress: {t+1}/{NUM_SNAPSHOTS}) ===")
        
        # A. 挖掘当前时刻的二团 (调用C++或模拟)
        biclique_file = utils.run_msbe_mining(snapshot_data, t)
        H_v, H_u = utils.parse_bicliques(biclique_file)
        
        # 移动稀疏矩阵到 GPU
        H_v = H_v.to(device)
        H_u = H_u.to(device)

        # B. 构建邻接矩阵
        adj_matrix = utils.build_adj_matrix(snapshot_data)
        adj_matrix = adj_matrix.to(device) # 移动邻接矩阵到 GPU
        
        # C. 准备训练样本
        pos_interactions = [(u, i) for u, i, _ in snapshot_data]
        
        # D. Epoch 训练
        for epoch in range(EPOCHS):
            model.train()
            
            # 打乱数据
            random.shuffle(pos_interactions)
            
            epoch_loss = 0.0
            epoch_bpr = 0.0
            epoch_cl = 0.0
            epoch_acc = 0.0
            steps = 0
            
            # Mini-batch 训练
            for i in range(0, len(pos_interactions), BATCH_SIZE):
                batch_samples = pos_interactions[i:i+BATCH_SIZE]
                
                optimizer.zero_grad()
                
                users = torch.LongTensor([x[0] for x in batch_samples]).to(device) # 移动数据到 GPU
                pos_items = torch.LongTensor([x[1] for x in batch_samples]).to(device)
                
                # 随机负采样
                neg_items = torch.randint(0, utils.num_items, (len(users),)).to(device)
                
                # 前向传播
                # 注意：user_history_state 需要 detach
                if user_history_state is not None:
                    current_history_state = user_history_state.detach().to(device)
                else:
                    current_history_state = None
                    
                u_global, u_local, new_state, i_global = model(adj_matrix, (H_v, H_u), current_history_state)
                
                # 计算损失
                loss, bpr, cl = model.calculate_loss(u_global, u_local, i_global, users, pos_items, neg_items)
                
                # 计算准确率 (Hit Ratio: Pos Score > Neg Score)
                current_user_emb = u_global[users]
                pos_item_emb = i_global[pos_items]
                neg_item_emb = i_global[neg_items]
                pos_scores = torch.mul(current_user_emb, pos_item_emb).sum(dim=1)
                neg_scores = torch.mul(current_user_emb, neg_item_emb).sum(dim=1)
                accuracy = (pos_scores > neg_scores).float().mean().item()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_bpr += bpr.item()
                epoch_cl += cl.item()
                epoch_acc += accuracy
                steps += 1
            
            # 记录平均指标
            avg_loss = epoch_loss / steps
            avg_bpr = epoch_bpr / steps
            avg_cl = epoch_cl / steps
            avg_acc = epoch_acc / steps
            
            history['loss'].append(avg_loss)
            history['bpr_loss'].append(avg_bpr)
            history['cl_loss'].append(avg_cl)
            history['accuracy'].append(avg_acc)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f} (Acc={avg_acc:.4f})")
        
        # 更新历史状态供下一个时间片使用 (取最后一个batch的状态作为近似，或者需要更复杂的处理)
        # 这里简单处理：使用最后一次前向传播产生的 new_state
        user_history_state = new_state.detach()
        
    print("\nTraining Finished!")
    
    # --- 4. 保存模型 ---
    torch.save(model.state_dict(), "tsb_cl_model.pth")
    print("Model saved to tsb_cl_model.pth")
    
    # --- 5. 绘制图表 (保存为文件) ---
    plot_metrics(history)

def plot_metrics(history):
    """
    绘制损失和准确率曲线并保存为文件
    """
    plt.figure(figsize=(12, 5))
    
    # 绘制 Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Total Loss')
    plt.plot(history['bpr_loss'], label='BPR Loss')
    plt.plot(history['cl_loss'], label='CL Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制 Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Training Accuracy', color='green')
    plt.title('Training Accuracy (Pos > Neg)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # 保存图片而不是直接显示
    plt.savefig('training_plot.png')
    print("Plot saved to training_plot.png")
    # plt.show()
    
    plt.tight_layout()
    
    # 弹出窗口显示
    print("Displaying metrics plot...")
    plt.show()

if __name__ == "__main__":
    train()
