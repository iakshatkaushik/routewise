"""Run the whole pipeline in order: data -> features -> models -> analysis -> router -> summary."""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
# Step 3e trains the teachers on 5 folds each (Qwen3-1.7B with its student rounds, then Qwen3-4B with
# --teacher-4b; the Qwen3-4B teacher is the one used), and step 3c trains the chosen student: the configuration
# picked by the validation-only student search (scripts/03f_student_search.py, three-seed means). The larger-model
# study (03d), the student search (03f), the ensemble check (03g) and the Qwen3-8B teacher (03e --teacher-8b)
# informed the choice but are not needed to rebuild the router, so they are not part of this pipeline.
FINETUNE = ["--final", "--base", "BAAI/bge-small-en-v1.5", "--pooling", "mean", "--lr", "8e-5", "--batch-size", "64",
            "--epochs", "4", "--max-tokens", "512", "--deficit-weight", "1", "--seed", "42", "--precision", "bf16",
            "--teacher", "models/teacher/teacher_oof_qwen3_4b.npz", "--distill-alpha", "0.1"]
STEPS = ["01_preprocess.py", "02_features.py", "02b_embeddings.py", "03_classifiers.py",
         "03b_gpu_network.py", "03e_distill.py", ("03e_distill.py", ["--teacher-4b"]), ("03c_finetune.py", FINETUNE),
         "04_regression.py", "05_clustering.py",
         "06_association.py", "07a_export_text_model.py", "07_build_router.py", "08_summary.py",
         "09_check_onnx_parity.py"]

if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else STEPS[0][:2]
    for step in STEPS:
        step, args = step if isinstance(step, tuple) else (step, [])
        if step[:3].rstrip("_") < start:            # e.g. "03e" < "07": resume from a later step
            continue
        t = time.time()
        print(f"\n===== {step} {' '.join(args)} =====", flush=True)
        subprocess.run([sys.executable, "-u", str(HERE / step), *args], check=True)
        print(f"----- {step} done in {time.time() - t:.0f}s", flush=True)
