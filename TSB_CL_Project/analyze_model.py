import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model import TSB_CL

def analyze_results():
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "tsb_cl_model.pth")
    plot_path = os.path.join(script_dir, "training_plot.png")
    
    print(f"=== 分析结果 ===")
    
    # 1. 检查图片是否存在
    if os.path.exists(plot_path):
        print(f"✅ 找到训练曲线图: {plot_path}")
        print("   请打开该图片查看：")
        print("   - 左图 (Loss): 曲线应呈下降趋势，表示模型正在学习。")
        print("   - 右图 (Accuracy): 曲线应呈上升趋势，表示推荐准确率在提高。")
    else:
        print(f"❌ 未找到图片: {plot_path}")

    # 2. 加载模型并分析
    if os.path.exists(model_path):
        print(f"\n✅ 找到模型文件: {model_path}")
        try:
            # 我们需要知道 num_users 和 num_items 才能初始化模型
            # 这里我们先加载 state_dict 看看形状
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            num_users = state_dict['user_embedding.weight'].shape[0]
            num_items = state_dict['item_embedding.weight'].shape[0]
            emb_dim = state_dict['user_embedding.weight'].shape[1]
            
            print(f"   - 用户数量: {num_users}")
            print(f"   - 物品数量: {num_items}")
            print(f"   - 向量维度: {emb_dim}")
            
            # 分析 Embedding 分布
            user_embs = state_dict['user_embedding.weight'].numpy()
            item_embs = state_dict['item_embedding.weight'].numpy()
            
            u_mean = np.mean(user_embs)
            u_std = np.std(user_embs)
            i_mean = np.mean(item_embs)
            i_std = np.std(item_embs)
            
            print(f"\n   [参数统计]")
            print(f"   - 用户向量均值: {u_mean:.6f}, 标准差: {u_std:.6f}")
            print(f"   - 物品向量均值: {i_mean:.6f}, 标准差: {i_std:.6f}")
            
            if u_std > 0.01 and i_std > 0.01:
                print("\n   ✅ 模型参数分布正常，看起来已经经过了训练（不是随机初始化的）。")
            else:
                print("\n   ⚠️ 模型参数方差极小，可能训练不充分或发生了坍塌。")
                
        except Exception as e:
            print(f"   ❌ 分析模型时出错: {e}")
    else:
        print(f"❌ 未找到模型文件: {model_path}")

if __name__ == "__main__":
    analyze_results()