import matplotlib.pyplot as plt
import numpy as np

# Data
epochs = range(1, 31)

# Model 1: Pure LightGCN (Baseline)
loss_1 = [0.5721, 0.2805, 0.2110, 0.1880, 0.1755, 0.1691, 0.1610, 0.1587, 0.1546, 0.1510, 0.1470, 0.1439, 0.1414, 0.1395, 0.1366, 0.1350, 0.1315, 0.1283, 0.1269, 0.1259, 0.1230, 0.1212, 0.1187, 0.1159, 0.1142, 0.1151, 0.1088, 0.1086, 0.1058, 0.1028]
recall_1 = [0.0489, 0.0417, 0.0353, 0.0303, 0.0471, 0.0387, 0.0408, 0.0432, 0.0550, 0.0479, 0.0455, 0.0488, 0.0444, 0.0510, 0.0541, 0.0542, 0.0451, 0.0497, 0.0426, 0.0472, 0.0547, 0.0569, 0.0507, 0.0510, 0.0364, 0.0397, 0.0397, 0.0419, 0.0644, 0.0437]
ndcg_1 = [0.0313, 0.0275, 0.0252, 0.0250, 0.0310, 0.0246, 0.0271, 0.0313, 0.0428, 0.0277, 0.0301, 0.0329, 0.0307, 0.0320, 0.0347, 0.0363, 0.0339, 0.0376, 0.0286, 0.0346, 0.0360, 0.0362, 0.0294, 0.0338, 0.0257, 0.0262, 0.0285, 0.0275, 0.0462, 0.0292]

# Model 2: Biclique GCN (tau=2)
loss_2 = [0.5320, 0.2596, 0.2041, 0.1852, 0.1733, 0.1677, 0.1615, 0.1585, 0.1507, 0.1500, 0.1462, 0.1412, 0.1381, 0.1356, 0.1328, 0.1302, 0.1275, 0.1235, 0.1220, 0.1185, 0.1166, 0.1140, 0.1105, 0.1085, 0.1059, 0.1036, 0.1013, 0.0986, 0.0962, 0.0952]
recall_2 = [0.0454, 0.0430, 0.0363, 0.0446, 0.0458, 0.0375, 0.0399, 0.0527, 0.0407, 0.0613, 0.0614, 0.0513, 0.0481, 0.0561, 0.0558, 0.0512, 0.0393, 0.0652, 0.0623, 0.0523, 0.0541, 0.0687, 0.0637, 0.0644, 0.0727, 0.0770, 0.0779, 0.0736, 0.0817, 0.0938]
ndcg_2 = [0.0293, 0.0295, 0.0273, 0.0320, 0.0327, 0.0261, 0.0285, 0.0367, 0.0277, 0.0394, 0.0405, 0.0378, 0.0353, 0.0362, 0.0365, 0.0387, 0.0306, 0.0448, 0.0422, 0.0371, 0.0431, 0.0511, 0.0407, 0.0456, 0.0568, 0.0549, 0.0541, 0.0546, 0.0554, 0.0677]

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
