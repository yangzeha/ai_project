
import os
import subprocess
import numpy as np
import scipy.sparse as sp
import torch
import struct
import re

class DataUtils:
    """
    数据处理工具类
    负责：
    1. 读取原始交互数据
    2. 调用C++程序挖掘二团
    3. 构建PyTorch所需的稀疏矩阵
    """
    def __init__(self, data_path, msbe_exe_path, temp_dir=None):
        self.data_path = data_path
        self.msbe_exe_path = msbe_exe_path
        
        if temp_dir is None:
            # Use absolute path for temp directory in the same folder as this script
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self.temp_dir = os.path.join(base_dir, 'temp')
        else:
            self.temp_dir = temp_dir
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        self.user_map = {}
        self.item_map = {}
        self.num_users = 0
        self.num_items = 0
        
    def load_data(self):
        """
        读取原始数据
        适配 bi_github.txt 格式:
        第一行: num_users num_items num_edges
        后续行: user_id item_id (无时间戳，模拟生成)
        """
        print(f"Loading data from {self.data_path}...")
        data = []
        with open(self.data_path, 'r') as f:
            # Skip header
            first_line = f.readline()
            
            # Read interactions
            # Simulate timestamp with line number
            timestamp = 0
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    u = int(parts[0])
                    i = int(parts[1])
                    
                    # Use sequential timestamp
                    t = timestamp
                    timestamp += 1
                    
                    if u not in self.user_map:
                        self.user_map[u] = len(self.user_map)
                    if i not in self.item_map:
                        self.item_map[i] = len(self.item_map)
                    
                    data.append((self.user_map[u], self.item_map[i], t))
                except ValueError:
                    continue
                    
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        
        # 按时间排序 (already sorted by simulation)
        # data.sort(key=lambda x: x[2])
        print(f"Loaded {len(data)} interactions. Users: {self.num_users}, Items: {self.num_items}")
        return data

    def split_snapshots(self, data, num_snapshots=5):
        """
        将数据按时间切分为多个快照
        """
        chunk_size = len(data) // num_snapshots
        snapshots = []
        for i in range(num_snapshots):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_snapshots - 1 else len(data)
            snapshots.append(data[start:end])
        return snapshots

    def save_binary_graph(self, snapshot_data, file_prefix):
        """
        将快照数据转换为 MSBE 所需的二进制格式 (_b_degree.bin, _b_adj.bin)
        """
        # 1. 提取并重映射节点 ID
        # MSBE 要求节点 ID 从 0 到 n-1
        # 我们将 User 映射到 [0, n1-1], Item 映射到 [n1, n1+n2-1]
        
        us = set()
        vs = set()
        for u, v, _ in snapshot_data:
            us.add(u)
            vs.add(v)
            
        sorted_us = sorted(list(us))
        sorted_vs = sorted(list(vs))
        
        n1 = len(sorted_us)
        n2 = len(sorted_vs)
        n = n1 + n2
        
        u_map = {u: i for i, u in enumerate(sorted_us)}
        v_map = {v: i + n1 for i, v in enumerate(sorted_vs)}
        
        u_rev = {i: u for u, i in u_map.items()}
        v_rev = {i: v for v, i in v_map.items()}
        
        # 2. 构建邻接表 (无向图)
        adj = [[] for _ in range(n)]
        edges_count = 0
        
        for u, v, _ in snapshot_data:
            uid = u_map[u]
            vid = v_map[v]
            
            # 添加双向边
            adj[uid].append(vid)
            adj[vid].append(uid)
            edges_count += 2
            
        # 排序邻接表 (MSBE 可能需要有序)
        for i in range(n):
            adj[i].sort()
            
        # 3. 写入 _b_degree.bin
        # Format: sizeof(ui), n1, n2, m, degree_array
        degree_file = file_prefix + "_b_degree.bin"
        with open(degree_file, 'wb') as f:
            f.write(struct.pack('I', 4)) # sizeof(ui) = 4
            f.write(struct.pack('I', n1))
            f.write(struct.pack('I', n2))
            f.write(struct.pack('I', edges_count))
            
            degrees = [len(adj[i]) for i in range(n)]
            f.write(struct.pack(f'{n}I', *degrees))
            
        # 4. 写入 _b_adj.bin
        # Format: edge_array (concatenated)
        adj_file = file_prefix + "_b_adj.bin"
        with open(adj_file, 'wb') as f:
            flat_adj = []
            for i in range(n):
                flat_adj.extend(adj[i])
            f.write(struct.pack(f'{edges_count}I', *flat_adj))
            
        return n1, n2, u_rev, v_rev

    def run_msbe_mining(self, snapshot_data, snapshot_id, tau=3, epsilon=0.5):
        """
        真实调用 MSBE 程序挖掘二团
        """
        # 1. 准备二进制数据
        # Use relative paths and change cwd to temp_dir to avoid path length/encoding issues
        # graph name for MSBE (no extension, no path)
        graph_name = f"graph_{snapshot_id}"
        
        # Full paths for preparation (save_binary_graph needs full path prefix)
        input_prefix = os.path.join(self.temp_dir, graph_name)
        fake_txt_path = input_prefix + ".txt" 
        
        # Create empty fake txt file if it doesn't exist
        with open(fake_txt_path, 'w') as f:
            f.write("dummy")

        n1, n2, u_rev, v_rev = self.save_binary_graph(snapshot_data, input_prefix)
        
        output_filename = f"bicliques_{snapshot_id}_tau{tau}_eps{epsilon}.txt"
        output_file = os.path.join(self.temp_dir, output_filename)
        
        # Check if file exists and is not empty
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"Snapshot {snapshot_id}: Biclique file already exists. Skipping mining.")
            return output_file
        
        # 2. Build Index
        # Execute in temp_dir
        cmd_build = [
            self.msbe_exe_path,
            f"{graph_name}.txt",
            "1", "1", "0.3", "GRL3"
        ]
        
        print(f"Snapshot {snapshot_id}: Building Index...")
        try:
            subprocess.run(cmd_build, cwd=self.temp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Error building index (might be okay if exists):", e)
            pass

        # 3. Enumerate Bicliques
        # Parameters: epsilon, tau, noRSim=2
        cmd_enum = [
            self.msbe_exe_path,
            f"{graph_name}.txt",
            "0", "1", "0.3", "GRL3",
            "1", "GRL3",
            "0", "0", "heu", "4",
            str(epsilon), str(tau), "2"
        ]
        
        print(f"Snapshot {snapshot_id}: Enumerating Bicliques (tau={tau}, epsilon={epsilon})...")
        try:
            # Capture output
            result = subprocess.run(cmd_enum, cwd=self.temp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
            output_str = result.stdout
        except subprocess.CalledProcessError as e:
            print("Error enumerating:", e)
            print(e.stdout)
            print(e.stderr)
            return output_file

        # Debug: Print output if empty
        if "CL :" not in output_str:
            print(f"DEBUG: No bicliques found in output. Raw output start:\n{output_str[:500]}")
            print(f"DEBUG: Command used: {' '.join(cmd_enum)}")

        # 4. 解析输出并保存
        # Output format:
        # Results : 
        # CL : 1, 2, 3, 
        # CR : 4, 5, 6, 
        # ----------------------------------------
        
        bicliques = []
        current_cl = []
        current_cr = []
        
        lines = output_str.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("CL :"):
                # Extract numbers, remove trailing comma
                parts = line[4:].strip().split(',')
                current_cl = [int(x) for x in parts if x.strip()]
            elif line.startswith("CR :"):
                parts = line[4:].strip().split(',')
                current_cr = [int(x) for x in parts if x.strip()]
            elif (line.startswith("---") or line.startswith("----------------")) and current_cl and current_cr:
                # Save biclique
                # Map back to original IDs
                orig_us = [u_rev[uid] for uid in current_cl if uid in u_rev]
                orig_vs = [v_rev[vid] for vid in current_cr if vid in v_rev]
                
                if orig_us and orig_vs:
                    bicliques.append((orig_us, orig_vs))
                
                current_cl = []
                current_cr = []
                
        # Write to file
        with open(output_file, 'w') as f:
            for us, vs in bicliques:
                f.write(f"{len(us)} {len(vs)}\n")
                f.write(" ".join(map(str, us)) + "\n")
                f.write(" ".join(map(str, vs)) + "\n")
                
        print(f"Snapshot {snapshot_id}: Found {len(bicliques)} bicliques.")
        return output_file

    def parse_bicliques(self, biclique_file):
        """
        解析二团文件，构建稀疏矩阵 H_u, H_v
        H_u: [num_users, num_bicliques]
        H_v: [num_bicliques, num_items]
        """
        biclique_users = []
        biclique_items = []
        b_idx = 0
        
        with open(biclique_file, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                try:
                    counts = lines[i].strip().split()
                    if not counts: break
                    n_u, n_v = map(int, counts)
                    
                    us = list(map(int, lines[i+1].strip().split()))
                    vs = list(map(int, lines[i+2].strip().split()))
                    
                    # 记录索引 (row, col)
                    for u in us:
                        biclique_users.append((u, b_idx))
                    for v in vs:
                        biclique_items.append((b_idx, v))
                        
                    b_idx += 1
                    i += 3
                except IndexError:
                    break
                    
        # 构建 PyTorch Sparse Tensor
        if b_idx == 0:
            # 防止空数据报错
            H_u = torch.sparse_coo_tensor(size=(self.num_users, 1))
            H_v = torch.sparse_coo_tensor(size=(1, self.num_items))
        else:
            # H_u
            u_indices = torch.LongTensor(biclique_users).t()
            u_values = torch.ones(len(biclique_users))
            H_u = torch.sparse_coo_tensor(u_indices, u_values, size=(self.num_users, b_idx))
            
            # H_v
            v_indices = torch.LongTensor(biclique_items).t()
            v_values = torch.ones(len(biclique_items))
            H_v = torch.sparse_coo_tensor(v_indices, v_values, size=(b_idx, self.num_items))
            
        return H_v, H_u

    def build_adj_matrix(self, snapshot_data):
        """
        构建归一化的邻接矩阵 (用于LightGCN)
        A = D^-0.5 * (R + R^T) * D^-0.5
        """
        users = [x[0] for x in snapshot_data]
        items = [x[1] for x in snapshot_data]
        
        # R matrix
        indices = torch.LongTensor([users, items])
        values = torch.ones(len(users))
        R = torch.sparse_coo_tensor(indices, values, size=(self.num_users, self.num_items))
        
        # 构建全图邻接矩阵
        # [0, R]
        # [R^T, 0]
        
        # 简化处理：直接返回 R 用于演示，实际 LightGCN 需要完整的 A
        # 这里我们构建标准的 A_hat
        
        # 转换为 Scipy 稀疏矩阵方便计算
        R_sp = sp.coo_matrix((np.ones(len(users)), (users, items)), shape=(self.num_users, self.num_items))
        
        # A = [0, R; R.T, 0]
        A = sp.vstack([
            sp.hstack([sp.csr_matrix((self.num_users, self.num_users)), R_sp]),
            sp.hstack([R_sp.T, sp.csr_matrix((self.num_items, self.num_items))])
        ])
        
        # 归一化
        rowsum = np.array(A.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_A = d_mat_inv_sqrt.dot(A).dot(d_mat_inv_sqrt)
        
        # 转回 Tensor
        coo = norm_A.tocoo()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        
        return torch.sparse_coo_tensor(indices, values, size=coo.shape)
