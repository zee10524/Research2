"""Component ablation on the frozen paper embeddings (reviewer point: which component is
responsible for the long-tail gain?).

Three factors are separated:
  1. inference-time L2 normalisation of the item matrix (used for both hybrids in
     Evaluation.ipynb, but not for the LightGCN baseline),
  2. the semantic (BERT) signal, checked against shuffled-row and pure-noise controls,
  3. the learned projection W_P, over 5 seeds.
"""
import numpy as np
from scipy import stats

from common import (KS, bucket_recall, evaluate_top, l2n, load_base, load_data, mean_std, per_user, save_json,
                    set_threads, topk, train_projection)

SEEDS = (42, 1, 2, 3, 4)
ALPHA = 0.7


def main():
    set_threads()
    d = load_data()
    U, I, B = load_base()
    results, tops = {}, {}

    def run(name, item_emb, user_emb=U, keep=False):
        top = topk(user_emb, item_emb, d.train_mask)
        r = evaluate_top(top, d)
        r["bucket_recall@20"] = bucket_recall(top, d)
        if keep:
            tops[name] = top
        return r

    def add(name, rows):
        buckets = [r.pop("bucket_recall@20") for r in rows]
        results[name] = mean_std(rows)
        results[name]["bucket_recall@20"] = {k: float(np.mean([b[k] for b in buckets])) for k in buckets[0]}
        results[name]["n_seeds"] = len(rows)
        m = results[name]
        print(f"{name:44s} all R@20 {m['recall@20']['mean']:.4f} N@20 {m['ndcg@20']['mean']:.4f} | "
              f"tail R@20 {m['tail_recall@20']['mean']:.4f}+-{m['tail_recall@20']['std']:.4f} "
              f"N@20 {m['tail_ndcg@20']['mean']:.4f}+-{m['tail_ndcg@20']['std']:.4f} | cov@20 {m['coverage@20']['mean']:.4f}", flush=True)

    # --- deterministic variants -------------------------------------------------------
    pop = d.item_count.astype(np.float32)[:, None]
    add("MostPop", [run("MostPop", pop, np.ones((d.n_users, 1), np.float32))])
    add("LightGCN (raw dot product)", [run("lgcn_raw", I, keep=True)])
    add("LightGCN + L2-norm items", [run("lgcn_l2", l2n(I), keep=True)])

    # content-only baseline: user profile = mean of centred, unit-norm BERT vectors of training items
    Bc = l2n(B - B.mean(0, keepdims=True))
    prof = d.train_mask.astype(np.float32) @ Bc
    add("BERT-only (content profile, centred)", [run("bert_only", Bc, l2n(prof))])
    Bn = l2n(B)
    add("BERT-only (content profile, uncentred)", [run("bert_only_unc", Bn, l2n(d.train_mask.astype(np.float32) @ Bn))])

    add(f"Static a={ALPHA} (paper: L2-normed parts)", [run("static_l2", ALPHA * l2n(I) + (1 - ALPHA) * l2n(B), keep=True)])
    add(f"Static a={ALPHA} (raw parts)", [run("static_raw", ALPHA * I + (1 - ALPHA) * B)])

    # --- seeded variants ----------------------------------------------------------------
    rows = {k: [] for k in ("static_shuf", "static_noise", "proj_raw", "proj_l2", "proj_shuf_raw", "proj_shuf_l2",
                            "proj_noise_raw", "proj_noise_l2", "projN_l2", "projN_shuf_l2")}
    for s in SEEDS:
        rng = np.random.default_rng(s)
        Bshuf = B[rng.permutation(len(B))]
        Bnoise = l2n(rng.standard_normal(B.shape).astype(np.float32))
        rows["static_shuf"].append(run("", ALPHA * l2n(I) + (1 - ALPHA) * l2n(Bshuf)))
        rows["static_noise"].append(run("", ALPHA * l2n(I) + (1 - ALPHA) * Bnoise))

        H = train_projection(U, I, B, d, s)
        rows["proj_raw"].append(run("proj_raw", H, keep=(s == SEEDS[0])))
        rows["proj_l2"].append(run("proj_l2", l2n(H), keep=(s == SEEDS[0])))
        H = train_projection(U, I, Bshuf, d, s)
        rows["proj_shuf_raw"].append(run("", H))
        rows["proj_shuf_l2"].append(run("", l2n(H)))
        H = train_projection(U, I, Bnoise, d, s)
        rows["proj_noise_raw"].append(run("", H))
        rows["proj_noise_l2"].append(run("", l2n(H)))
        # projection trained with the same L2 normalisation that is applied at inference
        rows["projN_l2"].append(run("", l2n(train_projection(U, I, B, d, s, normalize_in_loss=True))))
        rows["projN_shuf_l2"].append(run("", l2n(train_projection(U, I, Bshuf, d, s, normalize_in_loss=True))))
        print(f"seed {s} done", flush=True)

    add(f"Static a={ALPHA}, BERT rows shuffled (L2)", rows["static_shuf"])
    add(f"Static a={ALPHA}, Gaussian noise for BERT (L2)", rows["static_noise"])
    add("Projection (raw dot product)", rows["proj_raw"])
    add("Projection + L2-norm (paper)", rows["proj_l2"])
    add("Projection, BERT rows shuffled (raw)", rows["proj_shuf_raw"])
    add("Projection, BERT rows shuffled + L2", rows["proj_shuf_l2"])
    add("Projection, Gaussian noise for BERT (raw)", rows["proj_noise_raw"])
    add("Projection, Gaussian noise for BERT + L2", rows["proj_noise_l2"])
    add("Projection trained with L2-norm + L2", rows["projN_l2"])
    add("Projection trained with L2-norm, shuffled + L2", rows["projN_shuf_l2"])

    # --- paired significance tests on per-user long-tail metrics (seed 42) ---------------
    tail_gt = d.test_gt & d.tail[None, :]
    tests = {}
    for a, b in (("lgcn_l2", "static_l2"), ("lgcn_l2", "proj_l2"), ("static_l2", "proj_l2"), ("lgcn_raw", "lgcn_l2"),
                 ("lgcn_raw", "proj_raw")):
        for gname, gt in (("tail", tail_gt), ("all", d.test_gt)):
            _, ra, na = per_user(tops[a], gt, 20)
            _, rb, nb = per_user(tops[b], gt, 20)
            tests[f"{a} vs {b} [{gname}]"] = {
                "n_users": int(len(ra)),
                "recall@20_diff(b-a)": float((rb - ra).mean()), "recall@20_p": float(stats.wilcoxon(ra, rb).pvalue),
                "ndcg@20_diff(b-a)": float((nb - na).mean()), "ndcg@20_p": float(stats.wilcoxon(na, nb).pvalue),
            }
    for k, v in tests.items():
        print(k, v)

    save_json("components.json", {"seeds": SEEDS, "ks": KS, "results": results, "paired_tests": tests})


if __name__ == "__main__":
    main()
