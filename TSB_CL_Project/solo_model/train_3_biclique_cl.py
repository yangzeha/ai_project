from train_utils import run_training
from model_variants import BicliqueCL

if __name__ == "__main__":
    metrics = run_training(
        model_class=BicliqueCL,
        model_name="Biclique + CL (No RNN)",
        model_type="biclique_cl",
        epochs=30,
        tau=3,
        epsilon=0.1
    )
    print("\nFinal Metrics:", metrics)
