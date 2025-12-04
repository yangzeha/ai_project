import os
import sys
import subprocess
import struct

# 1. Compile MSBE with -D_PrintResults_
print(">>> [Step 1] Compiling MSBE with -D_PrintResults_...")
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
msbe_dir = os.path.join(project_root, "Similar-Biclique-Idx-main")
msbe_exe = os.path.join(msbe_dir, "msbe.exe")
cpp_src = os.path.join(msbe_dir, "main.cpp")

cmd = ["g++", "-O3", cpp_src, "-o", msbe_exe, "-D_PrintResults_"]
try:
    subprocess.run(cmd, check=True)
    print(f"✅ Compilation successful: {msbe_exe}")
except subprocess.CalledProcessError as e:
    print("❌ Compilation failed.")
    sys.exit(1)
except FileNotFoundError:
    print("❌ g++ not found. Please install MinGW.")
    sys.exit(1)

# 2. Minimal DataUtils for Verification
class DataUtils:
    def __init__(self, data_path, msbe_exe_path):
        self.data_path = data_path
        self.msbe_exe_path = msbe_exe_path
        self.temp_dir = os.path.join(current_dir, 'temp')
        if not os.path.exists(self.temp_dir): os.makedirs(self.temp_dir)
        self.user_map = {}
        self.item_map = {}

    def load_data(self):
        print(f"Loading data from {self.data_path}...")
        data = []
        with open(self.data_path, 'r') as f:
            f.readline()
            t = 0
            for line in f:
                try:
                    parts = line.strip().split()
                    if len(parts)<2: continue
                    u, i = int(parts[0]), int(parts[1])
                    if u not in self.user_map: self.user_map[u] = len(self.user_map)
                    if i not in self.item_map: self.item_map[i] = len(self.item_map)
                    data.append((self.user_map[u], self.item_map[i], t))
                    t += 1
                except: continue
        return data

    def save_binary_graph(self, snapshot_data, file_prefix):
        us, vs = set(), set()
        for u, v, _ in snapshot_data: us.add(u); vs.add(v)
        sorted_us, sorted_vs = sorted(list(us)), sorted(list(vs))
        n1, n2 = len(sorted_us), len(sorted_vs)
        n = n1 + n2
        u_map = {u: i for i, u in enumerate(sorted_us)}
        v_map = {v: i + n1 for i, v in enumerate(sorted_vs)}
        u_rev = {i: u for u, i in u_map.items()}
        v_rev = {i: v for v, i in v_map.items()}

        adj = [[] for _ in range(n)]
        edges_count = 0
        for u, v, _ in snapshot_data:
            uid, vid = u_map[u], v_map[v]
            adj[uid].append(vid); adj[vid].append(uid)
            edges_count += 2
        for i in range(n): adj[i].sort()

        with open(file_prefix + "_b_degree.bin", 'wb') as f:
            f.write(struct.pack('4I', 4, n1, n2, edges_count))
            f.write(struct.pack(f'{n}I', *[len(adj[i]) for i in range(n)]))
        with open(file_prefix + "_b_adj.bin", 'wb') as f:
            flat = [x for sub in adj for x in sub]
            f.write(struct.pack(f'{edges_count}I', *flat))
        return u_rev, v_rev

    def run_msbe_mining(self, snapshot_data, snapshot_id, tau=2, epsilon=0.1):
        graph_name = f"graph_{snapshot_id}"
        input_prefix = os.path.join(self.temp_dir, graph_name)
        with open(input_prefix + ".txt", 'w') as f: f.write("dummy")

        u_rev, v_rev = self.save_binary_graph(snapshot_data, input_prefix)

        print(f">>> [Step 2] Running MSBE for Snapshot {snapshot_id}...")
        # Build Index
        subprocess.run([self.msbe_exe_path, f"{graph_name}.txt", "1", "1", "0.3", "GRL3"],
                     cwd=self.temp_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Enumerate
        output_txt_path = os.path.join(self.temp_dir, "msbe_output.txt")
        try:
            with open(output_txt_path, "w") as outfile:
                subprocess.run([self.msbe_exe_path, f"{graph_name}.txt", "0", "1", "0.3", "GRL3",
                                    "1", "GRL3", "0", "0", "heu", "4", str(epsilon), str(tau), "2"],
                                    cwd=self.temp_dir, check=True, stdout=outfile, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')
            
            with open(output_txt_path, "r", encoding='utf-8', errors='ignore') as f:
                output_str = f.read()
                
        except Exception as e:
            print(f"Mining failed: {e}")
            return 0

        bicliques = []
        cl, cr = [], []
        
        # Parsing Logic
        lines = output_str.split('\n')
        print(f"    Raw output lines: {len(lines)}")
        
        for line in lines:
            line = line.strip()
            if line.startswith("CL : "): 
                cl = [int(x) for x in line[4:].split(',') if x.strip()]
            elif line.startswith("CR : "): 
                cr = [int(x) for x in line[4:].split(',') if x.strip()]
            elif (line.startswith("---") or line.startswith("----------------")) and cl and cr:
                bicliques.append((cl, cr))
                cl, cr = [], []

        print(f"✅ Snapshot {snapshot_id}: Found {len(bicliques)} bicliques!")
        if len(bicliques) > 0:
            print("    Sample Biclique 0:")
            print(f"    Users: {bicliques[0][0]}")
            print(f"    Items: {bicliques[0][1]}")
        else:
            print("⚠️ Warning: Found 0 bicliques. Check parameters or compilation.")
            print("    FULL OUTPUT:")
            print('\n'.join(lines))
            
        return len(bicliques)

# 3. Run Verification
if __name__ == "__main__":
    data_path = os.path.join(msbe_dir, "datasets", "bi_github.txt")
    utils = DataUtils(data_path, msbe_exe)
    data = utils.load_data()
    # Take a small chunk
    snapshot = data[:len(data)//3]
    
    count = utils.run_msbe_mining(snapshot, 0, tau=2, epsilon=0.1)
    
    if count > 0:
        print("\n🎉 SUCCESS: The '0 bicliques' issue is FIXED.")
        print("The MSBE executable is now correctly compiled and outputting results.")
        print("You can now run the full training script (once PyTorch is fixed).")
    else:
        print("\n❌ FAILURE: Still found 0 bicliques.")
