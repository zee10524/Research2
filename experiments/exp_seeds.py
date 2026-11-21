"""Fusion ablation and trade-off over 5 independently trained LightGCN backbones.

Unlike exp_components.py / exp_tradeoff.py (one frozen backbone, 5 fusion seeds), every seed
here has its own LightGCN (research2_seeds/, trained by train_lightgcn.py with model selection
on validation HR@20). The operating point of each family (gamma for LightGCN and Projection,
alpha for Static) is selected per seed on the validation split and reported on test:

  hm : maximise the harmonic mean of all-item and long-tail NDCG@20 on validation
  c5 : maximise long-tail NDCG@20 on validation subject to all-item NDCG@20 staying within
       5% of the same seed's raw LightGCN on validation

Usage: python exp_seeds.py [seed_dir] [val_csv]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import EMB, ROOT, evaluate_top, l2n, load_data, mean_std, per_user, save_json, set_threads, topk, train_projection

SEEDS = (42, 1, 2, 3, 4)
GAMMAS = [round(g, 1) for g in np.arange(0.0, 1.0001, 0.1)]
ALPHAS = [round(a, 1) for a in np.arange(0.0, 1.0001, 0.1)]
ALPHA = 0.7
KEYS = ("recall@20", "ndcg@20", "tail_recall@20", "tail_ndcg@20", "coverage@20", "tail_share@20", "gini@20")
FAMILIES = {"lightgcn": [f"lgcn_g{g}" for g in GAMMAS], "projection": [f"proj_g{g}" for g in GAMMAS],
            "static": [f"static_a{a}" for a in ALPHAS]}
PAIRS = (("lgcn_g1.0", f"static_a{ALPHA}"), ("lgcn_g1.0", "proj_g1.0"), ("lgcn_g0.0", "proj_g0.0"), ("lgcn_g0.0", "lgcn_g1.0"))


def hm(r):
    a, t = r["ndcg@20"], r["tail_ndcg@20"]
    return 2 * a * t / (a + t) if a + t > 0 else 0.0


def main():
    seed_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research2_seeds"
    val_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "research2_seeds_val_split.csv"
    set_threads()
    d = load_data()
    v = pd.read_csv(val_csv)
    val_gt = np.zeros_like(d.train_mask)
    val_gt[v.user.values, v.item.values] = True
    tail_gt = d.test_gt & d.tail[None, :]
    B = np.load(EMB / "bert_item_embeddings_final (1).npy").astype(np.float32)

    test, val, tests, chosen = {}, {}, {}, {}
    for s in SEEDS:
        U = np.load(seed_dir / f"gcn_user_embeddings_seed{s}.npy").astype(np.float32)
        I = np.load(seed_dir / f"gcn_item_embeddings_seed{s}.npy").astype(np.float32)
        rng = np.random.default_rng(s)
        Bshuf = B[rng.permutation(len(B))]
        Bnoise = l2n(rng.standard_normal(B.shape).astype(np.float32))
        H = train_projection(U, I, B, d, s)
        Hshuf = train_projection(U, I, Bshuf, d, s)
        Hnoise = train_projection(U, I, Bnoise, d, s)

        items = {f"lgcn_g{g}": l2n(I, g) for g in GAMMAS}
        items.update({f"proj_g{g}": l2n(H, g) for g in GAMMAS})
        items.update({f"static_a{a}": a * l2n(I) + (1 - a) * l2n(B) for a in ALPHAS})
        items.update({"static_shuf": ALPHA * l2n(I) + (1 - ALPHA) * l2n(Bshuf), "static_noise": ALPHA * l2n(I) + (1 - ALPHA) * Bnoise,
                      "static_raw": ALPHA * I + (1 - ALPHA) * B, "proj_shuf_raw": Hshuf, "proj_shuf_l2": l2n(Hshuf),
                      "proj_noise_raw": Hnoise, "proj_noise_l2": l2n(Hnoise)})
        tops = {}
        t_s, v_s = {}, {}
        for name, item in items.items():
            top = topk(U, item, d.train_mask)
            rt, rv = evaluate_top(top, d), evaluate_top(top, d, val_gt)
            t_s[name], v_s[name] = {k: rt[k] for k in KEYS}, {k: rv[k] for k in KEYS}
            if any(name in p for p in PAIRS):
                tops[name] = top
        for name in t_s:
            test.setdefault(name, []).append(t_s[name])
            val.setdefault(name, []).append(v_s[name])

        floor = 0.95 * v_s["lgcn_g0.0"]["ndcg@20"]
        for fam, names in FAMILIES.items():
            best_hm = max(names, key=lambda n: hm(v_s[n]))
            ok = [n for n in names if v_s[n]["ndcg@20"] >= floor]
            best_c5 = max(ok, key=lambda n: v_s[n]["tail_ndcg@20"]) if ok else None
            chosen.setdefault(f"{fam} [hm]", []).append(best_hm)
            chosen.setdefault(f"{fam} [c5]", []).append(best_c5)

        for a, b in PAIRS:
            for gname, gt in (("tail", tail_gt), ("all", d.test_gt)):
                _, ra, na = per_user(tops[a], gt, 20)
                _, rb, nb = per_user(tops[b], gt, 20)
                tests.setdefault(f"{a} vs {b} [{gname}]", []).append({
                    "seed": s, "recall@20_diff(b-a)": float((rb - ra).mean()), "recall@20_p": float(stats.wilcoxon(ra, rb).pvalue),
                    "ndcg@20_diff(b-a)": float((nb - na).mean()), "ndcg@20_p": float(stats.wilcoxon(na, nb).pvalue)})
        print(f"seed {s} done | selected:", {k: c[-1] for k, c in chosen.items()}, flush=True)

    def show(label, m):
        print(f"{label:34s} all R {m['recall@20']['mean']:.4f} N {m['ndcg@20']['mean']:.4f}+-{m['ndcg@20']['std']:.4f} | "
              f"tail R {m['tail_recall@20']['mean']:.4f}+-{m['tail_recall@20']['std']:.4f} N {m['tail_ndcg@20']['mean']:.4f}+-{m['tail_ndcg@20']['std']:.4f} | "
              f"cov {m['coverage@20']['mean']:.3f}")

    res_test = {n: mean_std(rows) for n, rows in test.items()}
    res_val = {n: mean_std(rows) for n, rows in val.items()}
    print("\n--- test, fixed settings (mean +- std over 5 LightGCN seeds) ---")
    for n, m in res_test.items():
        show(n, m)

    selected = {}
    print("\n--- test, operating point selected on validation ---")
    for key, names in chosen.items():
        if any(n is None for n in names):
            selected[key] = {"selected": names, "note": "no setting satisfied the constraint for at least one seed"}
            print(f"{key:34s} constraint not satisfiable: {names}")
            continue
        selected[key] = {"selected": names, **mean_std([test[n][i] for i, n in enumerate(names)])}
        show(f"{key} {sorted(set(names))}", selected[key])

    print("\n--- paired Wilcoxon per seed ---")
    for k, rows in tests.items():
        print(f"{k:38s} dR {np.mean([r['recall@20_diff(b-a)'] for r in rows]):+.4f} (max p {max(r['recall@20_p'] for r in rows):.2g}) "
              f"dN {np.mean([r['ndcg@20_diff(b-a)'] for r in rows]):+.4f} (max p {max(r['ndcg@20_p'] for r in rows):.2g})")

    save_json("seeds.json", {"seeds": SEEDS, "gammas": GAMMAS, "alphas": ALPHAS, "test": res_test, "validation": res_val,
                             "selected_on_validation": selected, "paired_tests": tests})


if __name__ == "__main__":
    main()
