import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = range(1, 31)

# Model 1: Pure LightGCN
loss_1 = [0.5716, 0.2816, 0.2115, 0.1877, 0.1760, 0.1689, 0.1646, 0.1593, 0.1567, 0.1532, 0.1490, 0.1457, 0.1427, 0.1408, 0.1385, 0.1350, 0.1340, 0.1333, 0.1295, 0.1266, 0.1237, 0.1221, 0.1200, 0.1173, 0.1168, 0.1145, 0.1118, 0.1106, 0.1068, 0.1052]
recall_1 = [0.0401, 0.0415, 0.0391, 0.0341, 0.0399, 0.0379, 0.0459, 0.0516, 0.0427, 0.0382, 0.0476, 0.0623, 0.0586, 0.0458, 0.0410, 0.0532, 0.0494, 0.0494, 0.0493, 0.0465, 0.0463, 0.0441, 0.0435, 0.0377, 0.0359, 0.0383, 0.0476, 0.0434, 0.0581, 0.0426]
ndcg_1 = [0.0263, 0.0298, 0.0318, 0.0221, 0.0271, 0.0227, 0.0311, 0.0322, 0.0281, 0.0279, 0.0328, 0.0414, 0.0403, 0.0310, 0.0256, 0.0345, 0.0332, 0.0370, 0.0343, 0.0321, 0.0310, 0.0275, 0.0267, 0.0266, 0.0252, 0.0305, 0.0334, 0.0284, 0.0389, 0.0264]

# Model 2: Biclique GCN
loss_2 = [0.5498, 0.2743, 0.2122, 0.1900, 0.1794, 0.1710, 0.1634, 0.1609, 0.1553, 0.1538, 0.1474, 0.1468, 0.1427, 0.1396, 0.1350, 0.1337, 0.1325, 0.1279, 0.1253, 0.1238, 0.1215, 0.1180, 0.1156, 0.1136, 0.1111, 0.1092, 0.1078, 0.1043, 0.1023, 0.0999]
recall_2 = [0.0416, 0.0373, 0.0419, 0.0411, 0.0381, 0.0414, 0.0412, 0.0472, 0.0447, 0.0416, 0.0416, 0.0572, 0.0371, 0.0443, 0.0563, 0.0537, 0.0486, 0.0480, 0.0358, 0.0397, 0.0442, 0.0609, 0.0504, 0.0594, 0.0485, 0.0603, 0.0553, 0.0542, 0.0629, 0.0680]
ndcg_2 = [0.0291, 0.0293, 0.0270, 0.0265, 0.0243, 0.0231, 0.0289, 0.0274, 0.0300, 0.0293, 0.0269, 0.0381, 0.0234, 0.0281, 0.0335, 0.0358, 0.0268, 0.0297, 0.0259, 0.0260, 0.0317, 0.0405, 0.0359, 0.0428, 0.0335, 0.0372, 0.0372, 0.0383, 0.0443, 0.0479]

plt.figure(figsize=(18, 5))

# 1. Loss Comparison
plt.subplot(1, 3, 1)
plt.plot(epochs, loss_1, label='Baseline (LightGCN)', marker='o', markersize=3)
plt.plot(epochs, loss_2, label='Biclique GCN', marker='s', markersize=3)
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Recall Comparison
plt.subplot(1, 3, 2)
plt.plot(epochs, recall_1, label='Baseline (LightGCN)', marker='o', markersize=3)
plt.plot(epochs, recall_2, label='Biclique GCN', marker='s', markersize=3)
plt.title('Test Recall@20 Comparison')
plt.xlabel('Epochs')
plt.ylabel('Recall@20')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. NDCG Comparison
plt.subplot(1, 3, 3)
plt.plot(epochs, ndcg_1, label='Baseline (LightGCN)', marker='o', markersize=3)
plt.plot(epochs, ndcg_2, label='Biclique GCN', marker='s', markersize=3)
plt.title('Test NDCG@20 Comparison')
plt.xlabel('Epochs')
plt.ylabel('NDCG@20')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ablation_comparison_1_vs_2.png', dpi=300)
print("Comparison plot saved to ablation_comparison_1_vs_2.png")
plt.show()
