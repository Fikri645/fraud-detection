"""
Run the full pipeline end-to-end. Convenience wrapper around the stage scripts.

    python scripts/run_all.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("Download data", "scripts/download_data.py"),
    ("Feature engineering", "scripts/run_features.py"),
    ("Train supervised + imbalance study", "scripts/run_training.py"),
    ("Autoencoder", "scripts/run_autoencoder.py"),
    ("GNN", "scripts/run_gnn.py"),
    ("Drift report", "scripts/run_drift.py"),
    ("Streaming benchmark", "streaming/simulate_stream.py"),
    ("Cross-dataset validation (ULB)", "scripts/run_cross_dataset.py"),
]


def main():
    for name, script in STAGES:
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if result.returncode != 0:
            print(f"[run_all] stage failed: {name}")
            sys.exit(result.returncode)
    print("\n[run_all] all stages complete.")


if __name__ == "__main__":
    main()
