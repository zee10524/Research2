"""Metadata-field and dimensionality-reduction ablation (reviewers 1 and 2).

Uses the per-field chunk sums written by encode_fields.py. A field subset is pooled as
sum(chunk [CLS]) / n_chunks over the selected fields, then reduced to 64-d with the paper's
Gaussian random projection (std 0.01). Each variant is scored three ways:
  * content-only: user profile = mean of centred unit item vectors (isolates the semantic signal)
  * Static Hybrid (alpha = 0.7, L2-normalised parts)
  * Projection Hybrid (+ L2 normalisation at inference, as in Evaluation.ipynb)
"""
import sys

import numpy as np

from common import EMB, evaluate, l2n, load_base, load_data, mean_std, save_json, set_threads, train_projection

SEEDS = (42, 1, 2)
KEYS = ("recall@20", "ndcg@20", "tail_recall@20", "tail_ndcg@20", "coverage@20")


def pool(sums, counts, idx):
    c = counts[:, idx].sum(1, keepdims=True)
    return (sums[:, idx].sum(1) / np.maximum(c, 1)).astype(np.float32)


def content_only(X, d):
    Xc = l2n(X - X.mean(0, keepdims=True))
    return evaluate(l2n(d.train_mask.astype(np.float32) @ Xc), Xc, d)


def main():
    set_threads()
    d = load_data()
    U, I, B_saved = load_base()
    z = np.load(EMB / "bert_fields_768.npz")
    sums, counts, fields = z["sums"], z["counts"], [str(f) for f in z["fields"]]
    nf = len(fields)
    full = pool(sums, counts, list(range(nf)))

    # sanity check: the saved 64-d matrix should be (close to) a linear image of the re-encoded full document
    Wfit, *_ = np.linalg.lstsq(full, B_saved, rcond=None)
    rel = float(np.linalg.norm(full @ Wfit - B_saved) / np.linalg.norm(B_saved))
    print(f"relative residual of saved BERT-64 as a linear map of re-encoded full doc: {rel:.4f}", flush=True)

    variants = {"all fields": list(range(nf))}
    for i, f in enumerate(fields):
        variants[f"only {f}"] = [i]
    for i, f in enumerate(fields):
        variants[f"without {f}"] = [j for j in range(nf) if j != i]

    out = {"fields": fields, "seeds": SEEDS, "saved_vs_reencoded_rel_residual": rel, "variants": {},
           "coverage_items_with_field": {f: int((counts[:, i] > 0).sum()) for i, f in enumerate(fields)}}

    def run_variant(name, X768):
        rows = {"content": [], "static": [], "projection": []}
        for s in SEEDS:
            Wb = np.random.default_rng(s).normal(scale=0.01, size=(768, 64)).astype(np.float32)
            X = X768 @ Wb
            rows["content"].append({k: content_only(X, d)[k] for k in KEYS})
            rows["static"].append({k: evaluate(U, 0.7 * l2n(I) + 0.3 * l2n(X), d)[k] for k in KEYS})
            rows["projection"].append({k: evaluate(U, l2n(train_projection(U, I, X, d, s)), d)[k] for k in KEYS})
        out["variants"][name] = {m: mean_std(r) for m, r in rows.items()}
        v = out["variants"][name]
        print(f"{name:22s} content tailR@20 {v['content']['tail_recall@20']['mean']:.4f} allR@20 {v['content']['recall@20']['mean']:.4f} | "
              f"static tailR@20 {v['static']['tail_recall@20']['mean']:.4f} tailN@20 {v['static']['tail_ndcg@20']['mean']:.4f} | "
              f"proj tailR@20 {v['projection']['tail_recall@20']['mean']:.4f}+-{v['projection']['tail_recall@20']['std']:.4f} "
              f"tailN@20 {v['projection']['tail_ndcg@20']['mean']:.4f} allN@20 {v['projection']['ndcg@20']['mean']:.4f}", flush=True)
        save_json("fields.json", out)

    for name, idx in variants.items():
        run_variant(name, pool(sums, counts, idx))
    rng = np.random.default_rng(0)
    run_variant("all fields, shuffled", full[rng.permutation(len(full))])

    # --- dimensionality reduction: random projection vs PCA vs no reduction (content-only) ---
    dr = {}
    c768 = content_only(full, d)
    dr["content-only 768-d (no reduction)"] = {k: c768[k] for k in KEYS}
    Xc = full - full.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca = Xc @ Vt[:64].T
    dr["explained_variance_pca64"] = float((S[:64] ** 2).sum() / (S ** 2).sum())
    r = content_only(pca, d)
    dr["content-only PCA-64"] = {k: r[k] for k in KEYS}
    pca_s = pca / pca.std() * B_saved.std()  # match the scale of the random-projected features
    r = evaluate(U, 0.7 * l2n(I) + 0.3 * l2n(pca), d)
    dr["static PCA-64"] = {k: r[k] for k in KEYS}
    rows = [{k: evaluate(U, l2n(train_projection(U, I, pca_s, d, s)), d)[k] for k in KEYS} for s in SEEDS]
    dr["projection PCA-64"] = mean_std(rows)
    out["dim_reduction"] = dr
    for k, v in dr.items():
        print(k, v)
    save_json("fields.json", out)


if __name__ == "__main__":
    sys.exit(main())
