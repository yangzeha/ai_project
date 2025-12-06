# How to Run the Direct Fusion Experiment on Kaggle

1. **Upload the Code**: Upload the entire `ai_project` folder or clone it from GitHub.
2. **Run the Notebook**: Open `Run_on_Kaggle.ipynb`.
3. **Execute All Cells**: The notebook has been updated to run `TSB_CL_Project/quick_proof_direct.py`.

## Why this change?
- The previous script `quick_proof_yelp.py` was for the Contrastive Learning (CL) model, which was showing performance similar to the baseline and had a plotting bug.
- The new script `quick_proof_direct.py` implements the **Direct Fusion** model (TSB + LightGCN weighted sum), which is expected to perform better and is more stable.

## Manual Execution
If you prefer to run it in a terminal:
```bash
python TSB_CL_Project/quick_proof_direct.py
```
