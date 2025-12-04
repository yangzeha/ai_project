import sys
import os
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
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'datasets', 'bi_github.txt')
MSBE_EXE = os.path.join(os.path.dirname(__file__), '..', '..', 'Similar-Biclique-Idx-main', 'msbe')
if os.name == 'nt':
    MSBE_EXE += ".exe"

BATCH_SIZE = 2048
EMBEDDING_DIM = 64
LR = 0.001
NUM_SNAPSHOTS = 4
TOP_K = 20

def evaluate(model, test_data, utils, device, model_type="full"):
    model.eval()
    
    # 构建测试集的邻接矩阵
    adj_matrix = utils.build_adj_matrix(test_data).to(device)
    
    # 构建空的二团矩阵 (测试阶段通常不重新挖掘，或者使用最后一个快照的二团)
    # 这里为了简化，传入空矩阵，因为测试集通常只看推荐效果
    # 注意：对于 BicliqueGCN，如果依赖 u_local，这里可能需要传入真实的二团
    # 为了公平对比，我们假设测试时使用最后一个训练快照的二团结构，或者不使用（取决于模型依赖）
    
    # 简单起见，构造空二团
    H_v = torch.sparse_coo_tensor(size=(1, utils.num_items)).to(device)
    H_u = torch.sparse_coo_tensor(size=(utils.num_users, 1)).to(device)
    
    with torch.no_grad():
        # 根据模型类型调用 forward
        if model_type == "lightgcn":
            u_emb, i_emb = model(adj_matrix)
        elif model_type == "biclique_gcn":
            u_emb, i_emb = model(adj_matrix, (H_v, H_u))
        elif model_type == "biclique_cl":
            u_global, u_local, i_emb = model(adj_matrix, (H_v, H_u))
            u_emb = u_global # 推荐使用 Global 视图
        else: # full
            u_global, u_local, _, i_emb = model(adj_matrix, (H_v, H_u))
            u_emb = u_global

    # 评估逻辑 (简化版，只计算部分用户以节省时间)
    test_users = list(set([u for u, _, _ in test_data]))
    sample_users = test_users[:1000] # 采样1000个用户进行评估
    
    hits = 0
    ndcg = 0
    
    u_emb = u_emb.cpu().numpy()
    i_emb = i_emb.cpu().numpy()
    
    # 构建 Ground Truth
    user_pos_items = {}
    for u, i, _ in test_data:
        if u not in user_pos_items:
            user_pos_items[u] = []
        user_pos_items[u].append(i)
        
    for u in sample_users:
        if u not in user_pos_items: continue
        
        # 获取用户 u 的 Embedding
        u_idx = utils.u_map.get(u)
        if u_idx is None: continue
        
        u_vec = u_emb[u_idx]
        
        # 计算所有物品的得分
        scores = np.dot(i_emb, u_vec)
        
        # 排除训练集中的物品 (这里简化，未排除)
        
        # Top-K
        top_k_items = np.argsort(scores)[::-1][:TOP_K]
        
        # 真实物品索引
        true_items = [utils.v_map[i] for i in user_pos_items[u] if i in utils.v_map]
        
        # 计算 Recall
        hit_count = len(set(top_k_items) & set(true_items))
        hits += hit_count / len(true_items) if len(true_items) > 0 else 0
        
        # 计算 NDCG
        dcg = 0
        idcg = 0
        for i, item_idx in enumerate(top_k_items):
            if item_idx in true_items:
                dcg += 1 / np.log2(i + 2)
        
        for i in range(min(len(true_items), TOP_K)):
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
    
    metrics = {'loss': [], 'recall': [], 'ndcg': []}
    user_history_state = None
    
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        for t, snapshot_data in enumerate(train_snapshots):
            # 1. Mining (仅当模型需要 Biclique 时)
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
                
                # Forward
                # 注意：DataUtils.load_data 已经将原始 ID 映射为内部 ID (0 ~ num_users-1)
                # 所以这里的 x[0] 和 x[1] 已经是映射后的 ID，不需要再查 u_map/v_map
                # 除非我们是在 save_binary_graph 里重新映射了一次，但 load_data 是全局映射
                
                # 修正：直接使用 x[0] 和 x[1]，因为 load_data 已经做了全局映射
                users = torch.LongTensor([x[0] for x in batch]).to(device)
                pos_items = torch.LongTensor([x[1] for x in batch]).to(device)
                neg_items = torch.randint(0, utils.num_items, (len(users),)).to(device)
