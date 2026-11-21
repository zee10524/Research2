"""Verify the LightGCN seed files downloaded from Colab and the rebuilt validation split.

Usage: python check_seeds.py [seed_dir] [val_csv]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, evaluate, load_data

seed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research2_seeds"
val_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "research2_seeds_val_split.csv"

d = load_data()
print("files:", sorted(p.name for p in seed_dir.iterdir()))

v = pd.read_csv(val_csv)
tr = pd.read_csv(ROOT / "train_split.csv")
te = pd.read_csv(ROOT / "test_split.csv")
key = ["user", "item"]
print(f"val rows {len(v):,} | users {v.user.nunique():,} | columns {list(v.columns)}")
print("val/train overlap:", len(v.merge(tr[key], on=key)), "| val/test overlap:", len(v.merge(te[key], on=key)),
      "| duplicates:", int(v.duplicated(key).sum()), "| total:", len(v) + len(tr) + len(te))

ok = True
for jf in sorted(seed_dir.glob("lightgcn_seed*.json")):
    s = json.load(open(jf))
    ue = np.load(seed_dir / f"gcn_user_embeddings_seed{s['seed']}.npy")
    ie = np.load(seed_dir / f"gcn_item_embeddings_seed{s['seed']}.npy")
    r = evaluate(ue, ie, d)
    diff = max(abs(r[k] - s[k]) for k in ("recall@20", "ndcg@20", "tail_recall@20", "tail_ndcg@20", "coverage@20"))
    finite = bool(np.isfinite(ue).all() and np.isfinite(ie).all())
    ok &= diff < 1e-4 and finite and ue.shape == (d.n_users, 64) and ie.shape == (d.n_items, 64)
    print(f"seed {s['seed']}: user {ue.shape} item {ie.shape} finite {finite} best_epoch {s['best_epoch']} "
          f"R@20 {r['recall@20']:.4f} tailR@20 {r['tail_recall@20']:.4f} | max diff vs json {diff:.2e}")
print("ALL OK" if ok else "MISMATCH")
