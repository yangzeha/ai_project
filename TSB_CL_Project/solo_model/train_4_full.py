from train_utils import run_training
from model_variants import FullTSBCL

if __name__ == "__main__":
    metrics = run_training(
        model_class=FullTSBCL,
        model_name="Full TSB-CL (Biclique + CL + RNN)",
        model_type="full",
        epochs=5,
        tau=3,
        epsilon=0.1
    )
    print("\nFinal Metrics:", metrics)
