"""Run the whole pipeline in order: data -> features -> models -> analysis -> router -> summary."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
STEPS = ["01_preprocess.py", "02_features.py", "02b_embeddings.py", "03_classifiers.py",
         "03b_gpu_network.py", "04_regression.py", "05_clustering.py", "06_association.py",
         "07_build_router.py", "08_summary.py", "09_check_onnx_parity.py"]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else STEPS[0][:2]
    for step in STEPS:
        if step[:3].rstrip("_") < start:
            continue
        t = time.time()
        print(f"\n===== {step} =====", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / step)], check=True)
        print(f"----- {step} done in {time.time() - t:.0f}s", flush=True)
