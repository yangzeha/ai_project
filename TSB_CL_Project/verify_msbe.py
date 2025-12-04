
import os
from data_utils import DataUtils

def verify_msbe():
    # --- 配置参数 ---
    DATA_PATH = r"c:\Users\LENOVO\Desktop\论文\Similar-Biclique-Idx-main\datasets\bi_github.txt"
    MSBE_EXE = r"c:\Users\LENOVO\Desktop\论文\Similar-Biclique-Idx-main\msbe.exe"
    
    if not os.path.exists(MSBE_EXE):
        print(f"Error: MSBE executable not found at {MSBE_EXE}")
        return

    print("Initializing DataUtils...")
    utils = DataUtils(DATA_PATH, MSBE_EXE)
    
    print("Loading full dataset...")
    # Load full data
    all_data = utils.load_data()
    print(f"Total interactions: {len(all_data)}")
    
    print("Running MSBE on full dataset (No Splitting)...")
    # Treat the whole dataset as snapshot 999 to avoid overwriting other temp files
    try:
        biclique_file = utils.run_msbe_mining(all_data, 999)
        
        print(f"\n--- Verification Complete ---")
        print(f"Biclique file generated at: {biclique_file}")
        
        # Double check the file content
        count = 0
        if os.path.exists(biclique_file):
            with open(biclique_file, 'r') as f:
                lines = f.readlines()
                # Each biclique takes 3 lines
                count = len(lines) // 3
        
        print(f"Total Maximal Similar Bicliques Found: {count}")
        
    except Exception as e:
        print(f"An error occurred during MSBE mining: {e}")

if __name__ == "__main__":
    verify_msbe()
