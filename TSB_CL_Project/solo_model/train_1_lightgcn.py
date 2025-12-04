from train_utils import run_training
from model_variants import PureLightGCN

if __name__ == "__main__":
    metrics = run_training(
        model_class=PureLightGCN,
        model_name="Pure LightGCN (Baseline)",
        model_type="lightgcn",
        epochs=5
    )
    print("\nFinal Metrics:", metrics)
