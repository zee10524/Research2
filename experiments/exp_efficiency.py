"""Computational efficiency (reviewer point 4): training time, inference time, memory.

All numbers are CPU-only on the machine this script runs on. LightGCN epoch time comes from
`train_lightgcn.py --time-only N` (results/lightgcn_timing.json); BERT encoding time is
printed by encode_fields.py. Run this script while nothing else is using the CPU.
"""
import json
import os
import platform
import time

import numpy as np
import psutil
import torch

from common import OUT, l2n, load_base, load_data, save_json, set_threads, topk, train_projection


def timeit(fn, repeat=5):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    set_threads()
    proc = psutil.Process(os.getpid())
    d = load_data()
    U, I, B = load_base()
    res = {"machine": {"cpu": platform.processor(), "logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                       "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "torch": torch.__version__, "device": "cpu"}}

    # --- training cost of the fusion stage ------------------------------------------------
    runs = []
    rss0 = proc.memory_info().rss
    for s in (42, 1, 2):
        H, st = train_projection(U, I, B, d, s, return_stats=True)
        runs.append(st["train_seconds"])
    res["projection_training"] = {"trainable_params": 64 * 64, "steps": 10_000, "batch": 1024,
                                  "seconds_mean": float(np.mean(runs)), "seconds_std": float(np.std(runs, ddof=1)),
                                  "rss_increase_mb": (proc.memory_info().rss - rss0) / 2**20}
    Bt, It = torch.tensor(B), torch.tensor(I)
    W = torch.randn(64, 64)
    res["hybrid_item_matrix_build_ms"] = timeit(lambda: (It + Bt @ W.T), 50)[0] * 1e3
    res["static_item_matrix_build_ms"] = timeit(lambda: 0.7 * l2n(I) + 0.3 * l2n(B), 50)[0] * 1e3

    # --- inference: full-catalog scoring and top-100 for every user -----------------------
    n_test_users = int(d.test_gt.any(1).sum())
    inf = {}
    for name, items in (("LightGCN", I), ("LightGCN + L2", l2n(I)), ("Static Hybrid", 0.7 * l2n(I) + 0.3 * l2n(B)), ("Projection Hybrid", l2n(H))):
        m, sd = timeit(lambda items=items: topk(U, items.astype(np.float32), d.train_mask))
        inf[name] = {"all_users_seconds": m, "std": sd, "ms_per_user": m / d.n_users * 1e3}
    res["inference_full_ranking_top100"] = inf
    res["n_users_ranked"] = d.n_users
    res["n_test_users"] = n_test_users

    # --- memory --------------------------------------------------------------------------
    mb = lambda a: a.nbytes / 2**20
    res["memory_mb"] = {
        "user_embeddings_6040x64_f32": mb(U), "item_embeddings_3706x64_f32": mb(I),
        "bert_items_3706x64_f32 (training of W_P only)": mb(B), "W_P_64x64_f32": 64 * 64 * 4 / 2**20,
        "serving_total (users + one fused item matrix)": mb(U) + mb(I),
        "bert_items_3706x768_f32 (before random projection)": 3706 * 768 * 4 / 2**20,
        "bert_base_weights_f32 (offline encoding only)": 109_482_240 * 4 / 2**20,
        "process_peak_rss": getattr(proc.memory_info(), "peak_wset", proc.memory_info().rss) / 2**20,
    }
    res["parameters"] = {"LightGCN (trainable)": (d.n_users + d.n_items) * 64, "Static Hybrid (trainable)": 0,
                         "Projection Hybrid (trainable, on top of frozen LightGCN)": 64 * 64, "BERT-base (frozen)": 109_482_240}
    p = OUT / "lightgcn_timing.json"
    if p.exists():
        res["lightgcn_training"] = json.load(open(p))
    save_json("efficiency.json", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
