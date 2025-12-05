import os
import urllib.request
import shutil

def download_yelp2018():
    # 1. Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "datasets")
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    output_path = os.path.join(dataset_dir, "yelp2018.txt")
    
    # URLs from kuandeng/LightGCN repository
    base_url = "https://raw.githubusercontent.com/kuandeng/LightGCN/master/Data/yelp2018"
    files = ["train.txt", "test.txt"]
    
    print("Downloading Yelp2018 dataset...")
    
    all_interactions = []
    
    for filename in files:
        url = f"{base_url}/{filename}"
        print(f"Downloading {filename} from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
                
                # Parse LightGCN format: UserID ItemID1 ItemID2 ...
                for line in content.strip().split('\n'):
                    parts = list(map(int, line.strip().split()))
                    if len(parts) < 2: continue
                    user = parts[0]
                    items = parts[1:]
                    for item in items:
                        all_interactions.append((user, item))
                        
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return

    if not all_interactions:
        print("Error: No data downloaded.")
        return

    # Remap IDs to ensure continuity (0...N-1)
    print("Processing data...")
    users = sorted(list(set(x[0] for x in all_interactions)))
    items = sorted(list(set(x[1] for x in all_interactions)))
    
    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: i for i, i in enumerate(items)} # Items usually start from 0 in LightGCN data but let's be safe
    
    num_users = len(users)
    num_items = len(items)
    num_edges = len(all_interactions)
    
    print(f"Stats: Users={num_users}, Items={num_items}, Interactions={num_edges}")
    
    # Save to txt format for data_utils.py
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        # Header
        f.write(f"{num_users} {num_items} {num_edges}\n")
        # Data
        for u, i in all_interactions:
            f.write(f"{user_map[u]} {item_map[i]}\n")
            
    print("Done! Yelp2018 dataset is ready.")
    print(f"Path: {output_path}")

if __name__ == "__main__":
    download_yelp2018()
