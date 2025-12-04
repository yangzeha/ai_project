
import os
import subprocess
import struct

class SimpleDataUtils:
    def __init__(self, data_path, msbe_exe_path):
        self.data_path = data_path
        self.msbe_exe_path = msbe_exe_path
        
        # Use absolute path for temp directory in the same folder as this script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.temp_dir = os.path.join(base_dir, 'temp')
        
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            
        self.user_map = {}
        self.item_map = {}
        
    def load_data(self, limit=None):
        print(f"Loading data from {self.data_path}...")
        data = []
        with open(self.data_path, 'r') as f:
            # Skip header
            first_line = f.readline()
            
            timestamp = 0
            for line in f:
                if limit and len(data) >= limit:
                    break
                try:
                    parts = line.strip().split()
                    if len(parts) < 2: continue
                    
                    u = int(parts[0])
                    i = int(parts[1])
                    
                    t = timestamp
                    timestamp += 1
                    
                    if u not in self.user_map:
                        self.user_map[u] = len(self.user_map)
                    if i not in self.item_map:
                        self.item_map[i] = len(self.item_map)
                    
                    data.append((self.user_map[u], self.item_map[i], t))
                except ValueError:
                    continue
                    
        print(f"Loaded {len(data)} interactions. Users: {len(self.user_map)}, Items: {len(self.item_map)}")
        return data

    def save_binary_graph(self, snapshot_data, file_prefix):
        # 1. Extract and re-map node IDs for this snapshot (or full data)
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
        
        # 2. Build Adjacency List (Undirected)
        adj = [[] for _ in range(n)]
        edges_count = 0
        
        for u, v, _ in snapshot_data:
            uid = u_map[u]
            vid = v_map[v]
            
            # Add bidirectional edges
            adj[uid].append(vid)
            adj[vid].append(uid)
            edges_count += 2
            
        # Sort adjacency lists
        for i in range(n):
            adj[i].sort()
            
        # 3. Write _b_degree.bin
        degree_file = file_prefix + "_b_degree.bin"
        with open(degree_file, 'wb') as f:
            f.write(struct.pack('I', 4)) # sizeof(ui) = 4
            f.write(struct.pack('I', n1))
            f.write(struct.pack('I', n2))
            f.write(struct.pack('I', edges_count))
            
            degrees = [len(adj[i]) for i in range(n)]
            f.write(struct.pack(f'{n}I', *degrees))
            
        # 4. Write _b_adj.bin
        adj_file = file_prefix + "_b_adj.bin"
        with open(adj_file, 'wb') as f:
            flat_adj = []
            for i in range(n):
                flat_adj.extend(adj[i])
            f.write(struct.pack(f'{edges_count}I', *flat_adj))
            
        return n1, n2

    def run_msbe_mining(self, snapshot_data, snapshot_id="full", tau=3, epsilon=0.5):
        # Use a simple ASCII name for the graph files to avoid path issues
        temp_graph_name = "g_test"
        input_prefix = os.path.join(self.temp_dir, temp_graph_name)
        fake_txt_path = input_prefix + ".txt" 
        
        # Create empty fake txt file if it doesn't exist
        with open(fake_txt_path, 'w') as f:
            f.write("dummy")

        print(f"Saving binary graph to {input_prefix}...")
        self.save_binary_graph(snapshot_data, input_prefix)
        
        output_file = os.path.join(self.temp_dir, f"bicliques_{snapshot_id}_tau{tau}_eps{epsilon}.txt")
        
        # 2. Build Index
        # Use cwd=self.temp_dir to avoid path length/char issues
        # Pass only the filename "g_test.txt"
        cmd_build = [
            self.msbe_exe_path,
            f"{temp_graph_name}.txt",
            "1", "1", "0.3", "GRL3"
        ]
        
        print(f"Building Index...")
        try:
            subprocess.run(cmd_build, cwd=self.temp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print("Error building index (might be okay if exists):", e)

        # 3. Enumerate Bicliques
        # Parameters: epsilon, tau, noRSim=2
        cmd_enum = [
            self.msbe_exe_path,
            f"{temp_graph_name}.txt",
            "0", "1", "0.3", "GRL3",
            "1", "GRL3",
            "0", "0", "heu", "4",
            str(epsilon), str(tau), "2"
        ]
        
        print(f"Enumerating Bicliques (tau={tau}, epsilon={epsilon})...")
        try:
            # Capture output as text
            result = subprocess.run(cmd_enum, cwd=self.temp_dir, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
            output_str = result.stdout
        except subprocess.CalledProcessError as e:
            print("Error enumerating:", e)
            print(e.stdout)
            print(e.stderr)
            return

        # 4. Parse Output
        import re
        count = 0
        
        # Try to find the count in the summary section
        match = re.search(r"\|\s*results size\s*=\s*(\d+)", output_str)
        if match:
            count = int(match.group(1))
            print(f"Found {count} bicliques (parsed from summary).")
        else:
            # Fallback: count lines if summary not found (though summary is expected)
            print("Warning: Could not parse 'results size' from output. Checking raw output...")
            
        print(f"Found {count} bicliques on full dataset with tau={tau}, epsilon={epsilon}.")
        
        # Save raw output for inspection
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_str)
        print(f"Raw output saved to {output_file}")

def main():
    DATA_PATH = r"c:\Users\LENOVO\Desktop\论文\Similar-Biclique-Idx-main\datasets\bi_github.txt"
    MSBE_EXE = r"c:\Users\LENOVO\Desktop\论文\Similar-Biclique-Idx-main\msbe.exe"
    
    if not os.path.exists(MSBE_EXE):
        print(f"Error: MSBE executable not found at {MSBE_EXE}")
        return

    utils = SimpleDataUtils(DATA_PATH, MSBE_EXE)
    
    # Load full data
    print("--- Loading FULL data for verification ---")
    all_data = utils.load_data()
    
    # Try parameters tau=3, epsilon=0.5
    print("\n--- Trying tau=3, epsilon=0.5 on FULL dataset ---")
    utils.run_msbe_mining(all_data, "full_verification_tau3_eps05", tau=3, epsilon=0.5)

if __name__ == "__main__":
    main()
