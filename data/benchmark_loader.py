#!/usr/bin/env python3
import os
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor

def load_simplified_drawings(path):
    """Read and parse every line of the .ndjson file."""
    drawings = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get('recognized', False):
                        drawings.append(obj['drawing'])
                except:
                    pass
    except Exception as e:
        # ignore errors, or print if you prefer
        pass
    return drawings

def benchmark(files, num_threads):
    """Load all files in parallel with num_threads and return elapsed seconds."""
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = [ex.submit(load_simplified_drawings, fn) for fn in files]
        for fut in futures:
            fut.result()
    return time.perf_counter() - start

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark QuickDraw loader over 1–16 threads"
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="downloaded_data/quickdraw_raw",
        help="Directory containing .ndjson files"
    )
    parser.add_argument(
        "--sample_files",
        type=int,
        default=None,
        help="If set, only benchmark on the first N files"
    )
    args = parser.parse_args()

    # collect all .ndjson files
    files = [
        os.path.join(args.raw_dir, f)
        for f in os.listdir(args.raw_dir)
        if f.endswith(".ndjson")
    ]
    files.sort()
    if args.sample_files:
        files = files[: args.sample_files]

    print(f"Benchmarking loader on {len(files)} files (full load)")
    results = {}
    for threads in range(1, 17):
        elapsed = benchmark(files, threads)
        results[threads] = elapsed
        print(f" Threads = {threads:2d} → {elapsed:.2f}s")

    best = min(results, key=results.get)
    print(f"\nFastest: {best} threads ({results[best]:.2f}s)")

if __name__ == "__main__":
    main()
