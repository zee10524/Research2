"""Shared data loading, evaluation and training utilities for the revision experiments.

Protocol mirrors Evaluation.ipynb: train items are masked, the full 3,706-item catalog is
ranked, long-tail = bottom 80% of items by training frequency, and the long-tail evaluation
restricts only the ground truth (users without long-tail test items are skipped).
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EMB = ROOT / "Embeddings"
OUT = ROOT / "experiments" / "results"
OUT.mkdir(parents=True, exist_ok=True)

KS = (10, 20, 50, 100)
MAXK = max(KS)
_DISC = 1.0 / np.log2(np.arange(MAXK) + 2)
_CDISC = np.cumsum(_DISC)


@dataclass
class Data:
    n_users: int
    n_items: int
    train: pd.DataFrame
    test: pd.DataFrame
    train_mask: np.ndarray  # (U, I) bool, items seen in training
    test_gt: np.ndarray  # (U, I) bool
    item_count: np.ndarray  # (I,) training frequency
    tail: np.ndarray  # (I,) bool, bottom 80% by training frequency


def load_data() -> Data:
    tr = pd.read_csv(ROOT / "train_split.csv")
    te = pd.read_csv(ROOT / "test_split.csv")
    n_users, n_items = int(tr.user.max()) + 1, int(tr.item.max()) + 1
    m = np.zeros((n_users, n_items), bool)
    m[tr.user.values, tr.item.values] = True
    g = np.zeros((n_users, n_items), bool)
    g[te.user.values, te.item.values] = True
    cnt = np.bincount(tr.item.values, minlength=n_items)
    # Same tie-breaking as Evaluation.ipynb: stable sort of Counter (first-appearance order).
    ordered = sorted(Counter(tr.item.values.tolist()).items(), key=lambda x: x[1])
    tail = np.zeros(n_items, bool)
    tail[[i for i, _ in ordered[: int(0.8 * n_items)]]] = True
    return Data(n_users, n_items, tr, te, m, g, cnt, tail)


def load_base():
    """Frozen LightGCN user/item embeddings and the 64-d random-projected BERT item matrix."""
    u = np.load(EMB / "gcn_user_embeddings (1).npy").astype(np.float32)
    i = np.load(EMB / "gcn_item_embeddings (1).npy").astype(np.float32)
    b = np.load(EMB / "bert_item_embeddings_final (1).npy").astype(np.float32)
    return u, i, b


def l2n(x: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Divide every row by its L2 norm raised to gamma (gamma=1: unit norm, gamma=0: unchanged)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12) ** gamma


def topk(user_emb: np.ndarray, item_emb: np.ndarray, mask: np.ndarray, k: int = MAXK) -> np.ndarray:
    s = user_emb @ item_emb.T
    s[mask] = -np.inf
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def per_user(top: np.ndarray, gt: np.ndarray, k: int):
    """Per-user Recall@k and NDCG@k over users with at least one ground-truth item."""
    users = np.flatnonzero(gt.any(1))
    hit = np.take_along_axis(gt[users], top[users, :k], 1)
    ng = gt[users].sum(1)
    rec = hit.sum(1) / ng
    ndcg = (hit * _DISC[:k]).sum(1) / _CDISC[np.minimum(ng, k) - 1]
    return users, rec, ndcg


def evaluate_top(top: np.ndarray, d: Data, gt: np.ndarray | None = None) -> dict:
    """All-item and long-tail Recall/NDCG at every K plus exposure statistics."""
    gt = d.test_gt if gt is None else gt
    users = np.flatnonzero(gt.any(1))
    res: dict = {}
    for k in KS:
        _, r, n = per_user(top, gt, k)
        _, rt, nt = per_user(top, gt & d.tail[None, :], k)
        rec_items = top[users, :k]
        res[f"recall@{k}"] = float(r.mean())
        res[f"ndcg@{k}"] = float(n.mean())
        res[f"tail_recall@{k}"] = float(rt.mean())
        res[f"tail_ndcg@{k}"] = float(nt.mean())
        res[f"coverage@{k}"] = float(len(np.unique(rec_items)) / d.n_items)
        # coverage of the long-tail catalog itself (2,964-item denominator, as used for SGL-ED)
        res[f"tail_coverage@{k}"] = float(d.tail[np.unique(rec_items)].sum() / d.tail.sum())
        res[f"tail_share@{k}"] = float(d.tail[rec_items].mean())
        res[f"avg_rec_pop@{k}"] = float(d.item_count[rec_items].mean())
        expo = np.sort(np.bincount(rec_items.ravel(), minlength=d.n_items).astype(float))
        n_i = d.n_items
        res[f"gini@{k}"] = float(((2 * np.arange(1, n_i + 1) - n_i - 1) * expo).sum() / (n_i * expo.sum()))
    return res


def evaluate(user_emb, item_emb, d: Data, gt=None) -> dict:
    return evaluate_top(topk(user_emb, item_emb, d.train_mask), d, gt)


# Popularity buckets by training frequency (the long tail is count <= 292).
BUCKETS = (("1-5", 1, 5), ("6-20", 6, 20), ("21-100", 21, 100), ("101-292", 101, 292), ("head >292", 293, 10**9))


def bucket_recall(top: np.ndarray, d: Data, k: int = 20) -> dict:
    out = {}
    for name, lo, hi in BUCKETS:
        sel = (d.item_count >= lo) & (d.item_count <= hi)
        _, r, _ = per_user(top, d.test_gt & sel[None, :], k)
        out[name] = float(r.mean())
    return out


def mean_std(rows: list[dict]) -> dict:
    keys = rows[0].keys()
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0} for k in keys}


def save_json(name: str, obj) -> Path:
    p = OUT / name
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)
    return p


def set_threads(n: int = 2):
    """The fusion stage only does 1024x64 operations, where extra threads cost more than they give."""
    import torch

    torch.set_num_threads(n)


class TripletSampler:
    """Vectorised version of sample_batch() in Evaluation.ipynb: a uniformly drawn user,
    one uniformly drawn training item of that user, one uniform negative outside the
    user's training items."""

    def __init__(self, d: Data, seed: int):
        self.rng = np.random.default_rng(seed)
        self.mask = d.train_mask
        tr = d.train.sort_values("user")
        self.items = tr.item.values
        deg = np.bincount(tr.user.values, minlength=d.n_users)
        self.start = np.concatenate([[0], np.cumsum(deg)[:-1]])
        self.deg = deg
        self.users = np.flatnonzero(deg > 0)
        self.n_items = d.n_items

    def sample(self, batch: int = 1024):
        u = self.rng.choice(self.users, batch)
        pos = self.items[self.start[u] + (self.rng.random(batch) * self.deg[u]).astype(np.int64)]
        neg = self.rng.integers(0, self.n_items, batch)
        bad = self.mask[u, neg]
        while bad.any():
            neg[bad] = self.rng.integers(0, self.n_items, int(bad.sum()))
            bad = self.mask[u, neg]
        return u, pos, neg


def train_projection(user_emb, item_gcn, item_sem, d: Data, seed: int, epochs: int = 10, steps: int = 1000,
                     batch: int = 1024, lr: float = 1e-3, normalize_in_loss: bool = False, return_stats: bool = False):
    """Projection Hybrid exactly as in the paper: e_i = e_i^GCN + W e_i^sem with a bias-free
    linear W trained by BPR on frozen embeddings. Returns the hybrid item matrix."""
    import time

    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    ut = torch.tensor(user_emb)
    gt_ = torch.tensor(item_gcn)
    st = torch.tensor(item_sem)
    lin = torch.nn.Linear(item_sem.shape[1], item_gcn.shape[1], bias=False)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    sampler = TripletSampler(d, seed)
    t0 = time.perf_counter()
    losses = []
    for _ in range(epochs):
        tot = 0.0
        for _ in range(steps):
            u, p, n = (torch.from_numpy(x) for x in sampler.sample(batch))
            pe = gt_[p] + lin(st[p])
            ne = gt_[n] + lin(st[n])
            if normalize_in_loss:
                pe, ne = F.normalize(pe, dim=1), F.normalize(ne, dim=1)
            loss = -F.logsigmoid((ut[u] * pe).sum(1) - (ut[u] * ne).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        losses.append(tot / steps)
    secs = time.perf_counter() - t0
    with torch.no_grad():
        hybrid = (gt_ + lin(st)).numpy()
    if return_stats:
        return hybrid, {"train_seconds": secs, "epoch_losses": losses, "n_params": sum(p.numel() for p in lin.parameters())}
    return hybrid
