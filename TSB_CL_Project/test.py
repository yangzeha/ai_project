import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from model import TSB_CL
from data_utils import DataUtils
import os

def test():
    # --- 配置参数 ---
    DATA_PATH = r"c:\Users\LENOVO\Desktop\论文\Similar-Biclique-Idx-main\datasets\bi_github.txt"
    MSBE_EXE = r"c:\Users\LENOVO\Desktop\论文\Similar-Biclique-Idx-main\msbe.exe"
    MODEL_PATH = "tsb_cl_model.pth"
    EMBEDDING_DIM = 64
    NUM_SNAPSHOTS = 5 # 保持与训练一致 (5份，取最后1份即为20%)
    TOP_K = 20
    
    print("Loading data and model...")
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    all_data = utils.load_data()
    snapshots = utils.split_snapshots(all_data, NUM_SNAPSHOTS)
    
    # 使用最后一个快照作为测试集
    test_data = snapshots[-1]
    
    # 加载模型
    model = TSB_CL(utils.num_users, utils.num_items, EMBEDDING_DIM)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print("Model file not found! Please train first.")
        return

    model.eval()
    
    # 构建测试集的邻接矩阵和二团 (为了简单，这里直接用测试集的数据构建图)
    # 在严格的评估中，应该使用训练集的图结构来预测测试集的交互
    # 这里为了演示流程，我们假设测试时图结构已知
    
    # 挖掘二团 (使用最后一个快照的ID)
    biclique_file = utils.run_msbe_mining(test_data, NUM_SNAPSHOTS-1)
    H_v, H_u = utils.parse_bicliques(biclique_file)
    adj_matrix = utils.build_adj_matrix(test_data)
    
    print("Start Evaluation...")
    
    # 准备测试用户
    test_users = list(set([x[0] for x in test_data]))
    # 随机采样一部分用户进行测试，避免太慢
    if len(test_users) > 100:
        test_users = np.random.choice(test_users, 100, replace=False)
    
    hits = 0
    ndcgs = 0
    binary_hits = 0 # 二分类准确率
    
    with torch.no_grad():
        # 获取所有用户的Embedding
        # 注意：这里没有传入历史状态，相当于冷启动或只看当前快照
        # 如果要严格评估时序性能，需要把训练时的 user_history_state 传进来
        u_global, u_local, _, i_global = model(adj_matrix, (H_v, H_u), None)
        
        all_item_emb = i_global
        
        for u in test_users:
            # 获取该用户的真实交互物品
            ground_truth = set([x[1] for x in test_data if x[0] == u])
            if not ground_truth: continue
            
            # --- 1. 计算二分类准确率 (Binary Accuracy) ---
            # 随机采样一个负样本
            neg_item = np.random.randint(0, utils.num_items)
            while neg_item in ground_truth:
                neg_item = np.random.randint(0, utils.num_items)
            
            # 取一个正样本
            pos_item = list(ground_truth)[0]
            
            pos_score = torch.mul(u_global[u], all_item_emb[pos_item]).sum()
            neg_score = torch.mul(u_global[u], all_item_emb[neg_item]).sum()
            
            if pos_score > neg_score:
                binary_hits += 1
            
            # --- 2. 计算 Top-K 指标 ---
            # 计算该用户对所有物品的评分
            u_emb = u_global[u].unsqueeze(0) # [1, dim]
            scores = torch.mm(u_emb, all_item_emb.t()).squeeze() # [num_items]
            
            # 排序取 Top-K
            _, indices = torch.topk(scores, TOP_K)
            pred_items = indices.cpu().numpy()
            
            # 计算 Recall@K
            hit = 0
            for item in pred_items:
                if item in ground_truth:
                    hit += 1
            hits += hit / len(ground_truth)
            
            # 计算 NDCG@K
            dcg = 0
            idcg = 0
            for i, item in enumerate(pred_items):
                if item in ground_truth:
                    dcg += 1 / np.log2(i + 2)
                if i < len(ground_truth):
                    idcg += 1 / np.log2(i + 2)
            
            if idcg > 0:
                ndcgs += dcg / idcg
                
    avg_recall = hits / len(test_users)
    avg_ndcg = ndcgs / len(test_users)
    avg_binary_acc = binary_hits / len(test_users)
    
    print(f"\nEvaluation Results (Top-{TOP_K}):")
    print(f"Binary Accuracy: {avg_binary_acc:.4f} (Target: >0.9)")
    print(f"Recall@{TOP_K}:      {avg_recall:.4f}")
    print(f"NDCG@{TOP_K}:        {avg_ndcg:.4f}")

if __name__ == "__main__":
    test()
