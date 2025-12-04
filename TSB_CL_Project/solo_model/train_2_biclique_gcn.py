from train_utils import run_training
from model_variants import BicliqueGCN

if __name__ == "__main__":
    metrics = run_training(
        model_class=BicliqueGCN,
        model_name="Biclique GCN (No CL, No RNN)",
        model_type="biclique_gcn",
        epochs=5,
        tau=3,
        epsilon=0.1
    )
    print("\nFinal Metrics:", metrics)
