import os
import sys
import time
import subprocess

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_suite(suite_name: str) -> int:
    cmd = [
        sys.executable, "-u",
        os.path.join("benchmarks", "run_ensemble_comparison_benchmark.py"),
        "--suite", suite_name
    ]
    print("=" * 80, flush=True)
    print(f" STARTING BENCHMARK SUITE: {suite_name.upper()} SOKOBAN", flush=True)
    print(f" Command: {' '.join(cmd)}", flush=True)
    print("=" * 80, flush=True)
    
    start_time = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    print("-" * 80, flush=True)
    print(f" Suite {suite_name.upper()} finished with returncode {proc.returncode} in {elapsed / 60.0:.2f} minutes.", flush=True)
    print("-" * 80, flush=True)
    return proc.returncode

def main():
    start_all = time.time()
    print("\n" + "#" * 80, flush=True)
    print(" STARTING SEQUENTIAL OVERNIGHT BENCHMARK RUN (3-BOX THEN 5-BOX)", flush=True)
    print("#" * 80 + "\n", flush=True)

    # 1. Run 3-Box Suite (In-Distribution, 200 Mazes)
    ret_3box = run_suite("3box")

    # 2. Run 5-Box Suite (Out-of-Distribution, 100 Mazes)
    ret_5box = run_suite("5box")

    total_elapsed = time.time() - start_all
    print("\n" + "#" * 80, flush=True)
    print(f" ALL OVERNIGHT BENCHMARKS COMPLETED in {total_elapsed / 60.0:.2f} minutes!", flush=True)
    print(f"  - 3-Box Suite Return Code: {ret_3box}", flush=True)
    print(f"  - 5-Box Suite Return Code: {ret_5box}", flush=True)
    print("#" * 80 + "\n", flush=True)

if __name__ == "__main__":
    main()
