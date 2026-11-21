"""LightGCN re-training for additional seeds and for the efficiency table.

Same model and hyper-parameters as LightGcn_implementation.ipynb (d=64, 3 layers, Adam
lr 1e-3, batch 4096, L2 1e-5 on ego embeddings, 5 uniform negatives per positive, one full
graph propagation per batch). Negative sampling is vectorised.

val_split.csv is not in the repository, so the notebook's early stopping on validation
HR@20 cannot be replayed. If ../val_split.csv exists it is used (best epoch by HR@20,
checked every 5 epochs); otherwise the model after the last epoch is kept.

Usage: python train_lightgcn.py --seed 1 --epochs 200 [--time-only 3]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from common import EMB, ROOT, evaluate, load_data, per_user, save_json, set_threads, topk


def norm_adj(d):
    r = sp.coo_matrix((np.ones(len(d.train), np.float32), (d.train.user.values, d.train.item.values)),
                      shape=(d.n_users, d.n_items)).tocsr()
    a = sp.bmat([[None, r], [r.T, None]], format="csr")
    deg = np.asarray(a.sum(1)).ravel()
    dinv = np.where(deg > 0, np.power(np.maximum(deg, 1e-12), -0.5), 0.0)
    a = (sp.diags(dinv) @ a @ sp.diags(dinv)).tocoo()
    return torch.sparse_coo_tensor(np.vstack([a.row, a.col]), a.data.astype(np.float32), a.shape).coalesce()


def propagate(adj, emb, n_layers):
    out, cur = [emb], emb
    for _ in range(n_layers):
        cur = adj @ cur
        out.append(cur)
    return torch.stack(out, 1).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--time-only", type=int, default=0, help="run N epochs, report timing, save nothing")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--trace-test", action="store_true", help="print test metrics at every validation check")
    ap.add_argument("--out", default=None, help="directory for embeddings and metrics (e.g. a Google Drive folder)")
    args = ap.parse_args()
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not args.time_only and (out_dir / f"lightgcn_seed{args.seed}.json").exists():
            print(f"seed {args.seed} already finished in {out_dir}, skipping")
            return
    set_threads(args.threads)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        print("device:", torch.cuda.get_device_name(0), flush=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dim, layers, lr, bs, reg, n_neg = 64, 3, 1e-3, 4096, 1e-5, 5

    d = load_data()
    adj = norm_adj(d)
    adj = adj.to(dev) if dev.type == "cuda" else adj.to_sparse_csr()
    emb = torch.nn.Parameter(torch.empty(d.n_users + d.n_items, dim, device=dev))
    torch.nn.init.xavier_uniform_(emb[: d.n_users])
    torch.nn.init.xavier_uniform_(emb[d.n_users:])
    opt = torch.optim.Adam([emb], lr=lr)
    pairs = d.train[["user", "item"]].values.astype(np.int64)

    val_gt = None
    if (ROOT / "val_split.csv").exists() and not args.time_only:
        v = pd.read_csv(ROOT / "val_split.csv")
        val_gt = np.zeros_like(d.train_mask)
        val_gt[v.user.values, v.item.values] = True
    best, best_state, best_epoch, stale = -1.0, None, 0, 0

    epochs = args.time_only or args.epochs
    times, t_start, first_ep = [], time.perf_counter(), 1
    # Checkpoint so that a runtime that dies mid-seed (free Colab) resumes instead of restarting.
    ckpt = out_dir / f"ckpt_seed{args.seed}.pt" if out_dir is not None and not args.time_only else None
    if ckpt is not None and ckpt.exists():
        c = torch.load(ckpt, map_location=dev, weights_only=False)
        with torch.no_grad():
            emb.copy_(c["emb"])
        opt.load_state_dict(c["opt"])
        rng.bit_generator.state = c["rng"]
        torch.set_rng_state(c["torch_rng"])
        best, best_state, best_epoch, stale, times = c["best"], c["best_state"], c["best_epoch"], c["stale"], c["times"]
        first_ep = c["epoch"] + 1
        print(f"seed {args.seed}: resuming at epoch {first_ep} (best val HR@20 so far {best:.4f} at epoch {best_epoch})", flush=True)
    for ep in range(first_ep, epochs + 1):
        t0 = time.perf_counter()
        perm = rng.permutation(len(pairs))
        tot, nb = 0.0, max(1, len(pairs) // bs)
        for b in range(nb):
            u, p = pairs[perm[b * bs:(b + 1) * bs]].T
            neg = rng.integers(0, d.n_items, (len(u), n_neg))
            bad = d.train_mask[u[:, None], neg]
            while bad.any():
                neg[bad] = rng.integers(0, d.n_items, int(bad.sum()))
                bad = d.train_mask[u[:, None], neg]
            u_t, p_t, n_t = (torch.from_numpy(x).to(dev) for x in (u, p, neg))
            full = propagate(adj, emb, layers)
            ue, ie = full[: d.n_users], full[d.n_users:]
            pos = (ue[u_t] * ie[p_t]).sum(1, keepdim=True)
            negs = (ue[u_t].unsqueeze(1) * ie[n_t]).sum(2)
            bpr = -torch.log(torch.sigmoid(pos - negs) + 1e-10).mean()
            ego_u, ego_i = emb[: d.n_users], emb[d.n_users:]
            l2 = reg * (ego_u[u_t].pow(2).sum() + ego_i[p_t].pow(2).sum() + ego_i[n_t].pow(2).sum()) / len(u)
            loss = bpr + l2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        times.append(time.perf_counter() - t0)
        msg = f"seed {args.seed} epoch {ep}/{epochs} loss {tot/nb:.4f} {times[-1]:.1f}s"
        if val_gt is not None and (ep % 5 == 0 or ep == 1):
            with torch.no_grad():
                full = propagate(adj, emb, layers).cpu().numpy()
            top = topk(full[: d.n_users], full[d.n_users:], d.train_mask, 20)
            users = np.flatnonzero(val_gt.any(1))
            hr = float(np.take_along_axis(val_gt[users], top[users], 1).any(1).mean())
            msg += f" val HR@20 {hr:.4f}"
            if args.trace_test:  # diagnostic only, never used for model selection
                r = evaluate(full[: d.n_users], full[d.n_users:], d)
                msg += f" | test R@20 {r['recall@20']:.4f} tailR@20 {r['tail_recall@20']:.4f} cov@20 {r['coverage@20']:.3f}"
            if hr > best:
                best, best_state, best_epoch, stale = hr, emb.detach().clone(), ep, 0
            else:
                stale += 1
            if stale >= 30:  # notebook: patience of 30 validation checks
                print(msg + " | early stop", flush=True)
                break
        if ep % 5 == 0 or ep == 1 or args.time_only:
            print(msg, flush=True)
        if ckpt is not None and ep % args.ckpt_every == 0:
            tmp = ckpt.with_suffix(".tmp")
            torch.save({"epoch": ep, "emb": emb.detach(), "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(), "best": best, "best_state": best_state,
                        "best_epoch": best_epoch, "stale": stale, "times": times}, tmp)
            tmp.replace(ckpt)

    stats = {"seed": args.seed, "epochs_run": len(times), "sec_per_epoch": float(np.mean(times)),
             "total_train_seconds": float(np.sum(times)), "n_params": int(emb.numel()),
             "threads": torch.get_num_threads(), "best_epoch": best_epoch or epochs,
             "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
             "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None}
    if args.time_only:
        print(stats)
        save_json("lightgcn_timing.json", stats)
        return
    with torch.no_grad():
        full = propagate(adj, best_state if best_state is not None else emb, layers).cpu().numpy()
    ue, ie = full[: d.n_users], full[d.n_users:]
    r = evaluate(ue, ie, d)
    stats.update({k: r[k] for k in ("recall@20", "ndcg@20", "tail_recall@20", "tail_ndcg@20", "coverage@20")})
    stats["early_stopping"] = val_gt is not None
    stats["best_val_hr@20"] = best if val_gt is not None else None
    print(stats)
    emb_dir = out_dir or EMB
    np.save(emb_dir / f"gcn_user_embeddings_seed{args.seed}.npy", ue)
    np.save(emb_dir / f"gcn_item_embeddings_seed{args.seed}.npy", ie)
    if out_dir is not None:  # written last: its presence marks the seed as finished
        with open(out_dir / f"lightgcn_seed{args.seed}.json", "w") as f:
            json.dump(stats, f, indent=2)
        if ckpt is not None and ckpt.exists():
            ckpt.unlink()
    else:
        save_json(f"lightgcn_seed{args.seed}.json", stats)


if __name__ == "__main__":
    main()
