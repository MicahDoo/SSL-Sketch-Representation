#!/usr/bin/env python3
# data/preprocess_dataset_quickdraw.py

import os
import json
import random
import argparse
import time
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool, cpu_count

# --- Configuration ---
BASE_DOWNLOAD_URL   = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
CATEGORIES_LIST_URL = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
DEFAULT_CATEGORIES  = ["cat","dog","apple","car","tree","house","bicycle","bird","face","airplane"]
FALLBACK_CATEGORIES = DEFAULT_CATEGORIES
TRAIN_RATIO         = 0.80
VAL_RATIO           = 0.10
RASTER_IMG_SIZE     = 224
CONFIG_FILENAME     = "quickdraw_config.json"

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
RAW_DOWNLOAD_DIR   = os.path.join(SCRIPT_DIR, "..", "downloaded_data")
PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "processed_data")


def download_all_categories_list(url=CATEGORIES_LIST_URL):
    print(f"Attempting to download category list from {url}...")
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        cats = [line.strip() for line in resp.text.splitlines() if line.strip()]
        if cats:
            print(f"Successfully downloaded {len(cats)} categories.")
            return sorted(cats)
    except Exception as e:
        print(f"Failed to download categories: {e}")
    return None


def download_category_file(cat_name, target_dir, skip_download=False):
    safe    = cat_name.replace(" ", "%20")
    raw_dir = os.path.join(target_dir, "quickdraw_raw")
    os.makedirs(raw_dir, exist_ok=True)
    fname   = f"{cat_name}.ndjson"
    outpath = os.path.join(raw_dir, fname)

    if os.path.exists(outpath):
        return outpath, True
    if skip_download:
        return outpath, False

    url = BASE_DOWNLOAD_URL + safe + ".ndjson"
    try:
        resp = requests.get(url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(outpath, "wb") as f, tqdm(
            desc=f"Downloading {cat_name:20}",
            total=total, unit="iB", unit_scale=True, leave=False, position=0
        ) as bar:
            for chunk in resp.iter_content(4096):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return outpath, True
    except Exception:
        if os.path.exists(outpath):
            os.remove(outpath)
        return outpath, False


def load_simplified_drawings(path,
                             max_items=None,
                             desc=None,
                             position=None):
    """Wrap the file iterator in tqdm so you see a per-line progress bar."""
    drawings = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            total = max_items
            bar = tqdm(
                f,
                desc=desc or f"Loading {os.path.basename(path)}",
                total=total,
                unit="line",
                leave=True,
                position=position
            )
            for i, line in enumerate(bar):
                if max_items is not None and i >= max_items:
                    break
                try:
                    obj = json.loads(line)
                    if obj.get("recognized", False):
                        drawings.append(obj["drawing"])
                except:
                    continue
    except FileNotFoundError:
        pass
    except Exception as e:
        tqdm.write(f"Error loading {path}: {e}")
    return drawings


def convert_raw_sketch_to_delta_sequence(strokes):
    pts = []
    lx = ly = 0
    n  = len(strokes)
    if n == 0:
        return np.zeros((0,5), dtype=np.float32)
    for si, stroke in enumerate(strokes):
        xs, ys = stroke
        if not xs or len(xs) != len(ys):
            continue
        dx0, dy0 = xs[0] - lx, ys[0] - ly
        if si > 0:
            pts.append([dx0, dy0, 0, 1, 0])
            dx0, dy0 = 0, 0
        is_last = (si == n - 1 and len(xs) == 1)
        state   = [0, 0, 1] if is_last else [1, 0, 0]
        pts.append([dx0, dy0] + state)
        lx, ly = xs[0], ys[0]
        for i in range(1, len(xs)):
            dx, dy = xs[i] - lx, ys[i] - ly
            lx, ly = xs[i], ys[i]
            is_last = (si == n - 1 and i == len(xs) - 1)
            state   = [0, 0, 1] if is_last else [1, 0, 0]
            pts.append([dx, dy] + state)
        if si < n - 1 and pts and pts[-1][4] == 0:
            pts[-1][2], pts[-1][3] = 0, 1
    if pts:
        pts[-1][2:] = [0, 0, 1]
    return np.array(pts, dtype=np.float32)


def rasterize_sequence_to_pil_image(seq, size, thickness=2, pad=0.02):
    img  = Image.new("L", (size, size), "white")
    draw = ImageDraw.Draw(img)
    segs = []
    x0 = y0 = 0.0
    for dx, dy, pd, pu, pe in seq:
        x1, y1 = x0 + dx, y0 + dy
        if pd > 0.5 and (x0, y0) != (0, 0):
            segs.append(((x0, y0), (x1, y1)))
        x0, y0 = x1, y1
        if pe > 0.5:
            break
    if not segs:
        return img.convert("1")
    xs = [a for s in segs for a in (s[0][0], s[1][0])]
    ys = [a for s in segs for a in (s[0][1], s[1][1])]
    if not xs or not ys:
        return img.convert("1")
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    w, h = maxx - minx, maxy - miny
    if w < 1e-6 and h < 1e-6:
        mx, my = segs[0][0]
        cx = size/2 + (mx - minx)
        cy = size/2 + (my - miny)
        draw.ellipse([(cx-1, cy-1), (cx+1, cy+1)], fill="black")
        return img.convert("1")
    cw, ch = size*(1-2*pad), size*(1-2*pad)
    scale = min(cw/w if w>0 else 1, ch/h if h>0 else 1)
    ox, oy = (size - w*scale)/2, (size - h*scale)/2
    for (ax, ay), (bx, by) in segs:
        sx0 = (ax - minx)*scale + ox
        sy0 = (ay - miny)*scale + oy
        sx1 = (bx - minx)*scale + ox
        sy1 = (by - miny)*scale + oy
        draw.line([(sx0, sy0), (sx1, sy1)], width=thickness, fill="black")
    return img.convert("1")


def normalize_and_rasterize(args):
    seq, sd, sz = args
    try:
        norm = seq.copy()
        norm[:, :2] /= sd
        pil  = rasterize_sequence_to_pil_image(norm, sz)
        return norm, pil
    except:
        return None, None


def download_and_load(cat, raw_dir, max_items, skip_download, position):
    t0 = time.time()
    path, ok = download_category_file(cat, raw_dir, skip_download)
    dt_dl = time.time() - t0

    draws = []
    dt_ld = 0.0
    if ok:
        t1 = time.time()
        draws = load_simplified_drawings(
            path,
            max_items=max_items,
            desc=f"Loading {cat:20}",
            position=position
        )
        dt_ld = time.time() - t1

    return cat, ok, draws, dt_dl, dt_ld


def main(categories_initial, raw_dir, proc_dir,
         train_r, val_r, max_items, skip_download, single_proc):

    use_parallel = not single_proc
    print(f"=== QuickDraw Preproc | parallel={'ON' if use_parallel else 'OFF'} | skip_download={skip_download} ===")

    vec_base   = os.path.join(proc_dir, "quickdraw_vector")
    ras_base   = os.path.join(proc_dir, "quickdraw_raster")
    raw_ndjson = os.path.join(raw_dir, "quickdraw_raw")
    for d in (vec_base, ras_base):
        os.makedirs(d, exist_ok=True)
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(vec_base, sp), exist_ok=True)
        os.makedirs(os.path.join(ras_base, sp), exist_ok=True)

    # pick categories
    if skip_download:
        existing = {f for f in os.listdir(raw_ndjson) if f.endswith(".ndjson")} if os.path.isdir(raw_ndjson) else set()
        cats     = [c for c in categories_initial if f"{c}.ndjson" in existing]
        if not cats:
            print("No local .ndjson found. Exiting."); return
    else:
        cats = categories_initial

    if not cats:
        print("No categories to process. Exiting."); return

    print(f"Processing {len(cats)} categories: {cats}")

    # ensure per-category raster dirs
    for sp in ("train", "val", "test"):
        for c in cats:
            os.makedirs(os.path.join(ras_base, sp, c), exist_ok=True)

    category_map = {c: i for i, c in enumerate(cats)}

    # --- Download + Load in parallel ---
    all_sketches = {}
    if use_parallel:
        nd_threads = 8
        print(f"[Download+Load] using {nd_threads} threads")
        with ThreadPoolExecutor(max_workers=nd_threads) as ex:
            futures = {
                ex.submit(download_and_load, c, raw_dir, max_items, skip_download, idx+1): c
                for idx, c in enumerate(cats)
            }
            for fut in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Download+Load",
                            position=0):
                cat, ok, draws, dt_dl, dt_ld = fut.result()
                if not ok:
                    tqdm.write(f"→ skip '{cat}' (download fail) in {dt_dl:.2f}s")
                    continue
                tqdm.write(f"★ '{cat}': downloaded in {dt_dl:.2f}s, loaded {len(draws)} sketches in {dt_ld:.2f}s")
                if draws:
                    all_sketches[cat] = draws
    else:
        print("[Download+Load] single-process mode")
        for idx, cat in enumerate(tqdm(cats, desc="Download+Load", position=0)):
            t0 = time.time()
            path, ok = download_category_file(cat, raw_dir, skip_download)
            dt_dl = time.time() - t0
            if not ok:
                tqdm.write(f"→ skip '{cat}' (download fail) in {dt_dl:.2f}s")
                continue
            t1 = time.time()
            draws = load_simplified_drawings(
                path,
                max_items=max_items,
                desc=f"Loading {cat:20}",
                position=idx+1
            )
            dt_ld = time.time() - t1
            tqdm.write(f"★ '{cat}': downloaded in {dt_dl:.2f}s, loaded {len(draws)} sketches in {dt_ld:.2f}s")
            if draws:
                all_sketches[cat] = draws

    if not all_sketches:
        print("No sketches loaded. Aborting."); return

    # --- Pass 1: δ-conversion ---
    num_w      = max(1, cpu_count() // 2)
    print(f"[Pass 1] δ-conversion with {num_w} processes")
    all_seq    = {}
    train_dels = []
    for cat, draws in tqdm(all_sketches.items(),
                            total=len(all_sketches),
                            desc="Pass 1: δ-conversion",
                            unit="cat"):
        tqdm.write(f"[Pass1][{cat}] {len(draws)} drawings")
        t0 = time.time()
        if use_parallel:
            with Pool(num_w) as p:
                seqs = list(tqdm(
                    p.imap(convert_raw_sketch_to_delta_sequence, draws),
                    total=len(draws),
                    desc=f"Δ→seq {cat}",
                    leave=False,
                    position=0
                ))
        else:
            seqs = [
                convert_raw_sketch_to_delta_sequence(d)
                for d in tqdm(draws, desc=f"Δ→seq {cat}", leave=False, position=0)
            ]
        dt    = time.time() - t0
        seqs  = [s for s in seqs if s is not None and s.size > 0]
        tqdm.write(f"[Pass1][{cat}] {len(seqs)} valid in {dt:.2f}s")
        all_seq[cat] = seqs
        ntr = int(train_r * len(seqs))
        for s in seqs[:ntr]:
            train_dels.extend(s[:,0]); train_dels.extend(s[:,1])

    std_dev = float(np.std(train_dels)) if train_dels else 1.0
    std_dev = max(std_dev, 1e-6)
    print(f"[Pass1] global std_dev = {std_dev:.6f}")

    # --- Pass 2: normalize + rasterize ---
    print(f"[Pass 2] normalize+raster with {num_w} processes")
    for cat, seqs in tqdm(all_seq.items(),
                          total=len(all_seq),
                          desc="Pass 2: normalize+raster",
                          unit="cat"):
        tqdm.write(f"[Pass2][{cat}] {len(seqs)} sequences")
        args = [(s, std_dev, RASTER_IMG_SIZE) for s in seqs]
        t0   = time.time()
        if use_parallel:
            with Pool(num_w) as p:
                results = list(tqdm(
                    p.imap(normalize_and_rasterize, args),
                    total=len(args),
                    desc=f"NR {cat}",
                    leave=False,
                    position=0
                ))
        else:
            results = [
                normalize_and_rasterize(a)
                for a in tqdm(args, desc=f"NR {cat}", leave=False, position=0)
            ]
        dt    = time.time() - t0
        valid = [(n,i) for n,i in results if i is not None]
        tqdm.write(f"[Pass2][{cat}] {len(valid)} rasters in {dt:.2f}s")
        if not valid:
            continue

        vecs, imgs = zip(*valid)
        combo      = list(zip(vecs, imgs))
        random.shuffle(combo)
        vecs, imgs = zip(*combo)
        total      = len(vecs)
        ntr        = int(train_r * total)
        nvl        = int(val_r   * total)
        splits     = {
            "train": (vecs[:ntr], imgs[:ntr]),
            "val":   (vecs[ntr:ntr+nvl], imgs[ntr:ntr+nvl]),
            "test":  (vecs[ntr+nvl:], imgs[ntr+nvl:])
        }
        for sp, (vseqs, rimgs) in splits.items():
            if not vseqs:
                continue
            np.save(os.path.join(vec_base, sp, f"{cat}.npy"),
                    np.array(vseqs, dtype=object))
            outd = os.path.join(ras_base, sp, cat)
            for i, im in enumerate(rimgs):
                im.save(os.path.join(outd, f"sketch_{i:05d}.png"))
        tqdm.write(f"[Pass2][{cat}] split → train={ntr}, val={nvl}, test={total-ntr-nvl}")

    # --- Write config ---
    print("[Final] writing config")
    cfg = {
        "dataset_name": "quickdraw",
        "quickdraw_std_dev": std_dev,
        "category_map": category_map,
        "categories_processed": cats,
        "train_ratio": train_r,
        "val_ratio": val_r,
        "test_ratio": round(1-train_r-val_r, 2),
        "max_items_per_category_processed": max_items or "all",
        "vector_data_path_relative": "quickdraw_vector",
        "raster_data_path_relative": "quickdraw_raster",
        "raster_image_size": RASTER_IMG_SIZE,
    }
    with open(os.path.join(vec_base, CONFIG_FILENAME), "w") as f:
        json.dump(cfg, f, indent=4)

    print(f"Done! Vectors @ {vec_base}, Rasters @ {ras_base}, Config → {CONFIG_FILENAME}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="QuickDraw Preproc (Vector & Raster)")
    p.add_argument("--categories", nargs="+", default=None)
    p.add_argument("--all_categories", action="store_true")
    p.add_argument("--max_categories", type=int, default=None,
                   help="Process only the first N categories")
    p.add_argument("--skip_raw_download", action="store_true",
                   help="Do not fetch new .ndjson; only use local files")
    p.add_argument("--single_process", action="store_true",
                   help="Disable all parallelism; run sequentially")
    p.add_argument("--raw_dir", default=RAW_DOWNLOAD_DIR)
    p.add_argument("--processed_dir", default=PROCESSED_DATA_DIR)
    p.add_argument("--train_ratio", type=float, default=TRAIN_RATIO)
    p.add_argument("--val_ratio", type=float,   default=VAL_RATIO)
    p.add_argument("--max_items_per_category", type=int, default=None)

    args = p.parse_args()
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    if args.all_categories:
        cats = download_all_categories_list() or FALLBACK_CATEGORIES
    elif args.categories:
        cats = sorted(set(args.categories))
    else:
        cats = sorted(DEFAULT_CATEGORIES)

    if args.max_categories is not None:
        cats = cats[: args.max_categories]
        print(f"Limiting to first {len(cats)} categories: {cats}")

    print(f"Raw dir:       {os.path.abspath(args.raw_dir)}")
    print(f"Processed dir: {os.path.abspath(args.processed_dir)}")
    print(f"Using categories: {cats}\n")

    main(
        categories_initial = cats,
        raw_dir            = args.raw_dir,
        proc_dir           = args.processed_dir,
        train_r            = args.train_ratio,
        val_r              = args.val_ratio,
        max_items          = args.max_items_per_category,
        skip_download      = args.skip_raw_download,
        single_proc        = args.single_process
    )
