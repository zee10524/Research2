"""Accuracy vs long-tail trade-off (reviewer point: how can both objectives be balanced?).

Item vectors are divided by ||e_i||^gamma. gamma=0 is the plain dot product (LightGCN's
operating point), gamma=1 is the full L2 normalisation used for the hybrids in the paper.
Sweeping gamma traces the whole frontier for every model, so models are compared at matched
operating points instead of at two different normalisation settings.

NOTE: the curves are computed on the test split and are a sensitivity analysis. A single
operating point must be selected on validation data before it is reported as a result.
"""
import numpy as np

from common import OUT, evaluate, l2n, load_base, load_data, mean_std, save_json, set_threads, train_projection

GAMMAS = [round(g, 2) for g in np.arange(0.0, 1.0001, 0.1)]
ALPHAS = [round(a, 1) for a in np.arange(0.0, 1.0001, 0.1)]
SEEDS = (42, 1, 2, 3, 4)
KEYS = ("recall@20", "ndcg@20", "tail_recall@20", "tail_ndcg@20", "coverage@20", "tail_share@20", "gini@20")


def main():
    set_threads()
    d = load_data()
    U, I, B = load_base()
    hybrids = [train_projection(U, I, B, d, s) for s in SEEDS]
    out = {"gammas": GAMMAS, "alphas": ALPHAS, "lightgcn": [], "projection": [], "static_alpha": []}
    for g in GAMMAS:
        r = evaluate(U, l2n(I, g), d)
        out["lightgcn"].append({k: r[k] for k in KEYS})
        rows = [evaluate(U, l2n(H, g), d) for H in hybrids]
        out["projection"].append({k: v["mean"] for k, v in mean_std([{k: r[k] for k in KEYS} for r in rows]).items()})
        print(f"gamma={g:.1f} LightGCN N@20 {out['lightgcn'][-1]['ndcg@20']:.4f} tailN@20 {out['lightgcn'][-1]['tail_ndcg@20']:.4f} | "
              f"Projection N@20 {out['projection'][-1]['ndcg@20']:.4f} tailN@20 {out['projection'][-1]['tail_ndcg@20']:.4f}", flush=True)
    for a in ALPHAS:  # static fusion weight on L2-normalised parts, as in the notebook sweep
        r = evaluate(U, a * l2n(I) + (1 - a) * l2n(B), d)
        out["static_alpha"].append({k: r[k] for k in KEYS})
        print(f"alpha={a:.1f} Static N@20 {r['ndcg@20']:.4f} tailN@20 {r['tail_ndcg@20']:.4f} cov {r['coverage@20']:.4f}", flush=True)
    save_json("tradeoff.json", out)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, key, c in (("LightGCN", "lightgcn", "tab:blue"), ("Projection Hybrid (5 seeds)", "projection", "tab:orange")):
        x = [p["ndcg@20"] for p in out[key]]
        y = [p["tail_ndcg@20"] for p in out[key]]
        ax[0].plot(x, y, "-o", color=c, label=f"{name}, $\\gamma$ 0 to 1", ms=4)
        for g, xi, yi in zip(GAMMAS, x, y):
            if g in (0.0, 0.5, 1.0):
                ax[0].annotate(f"$\\gamma$={g:g}", (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=8, color=c)
    x = [p["ndcg@20"] for p in out["static_alpha"]]
    y = [p["tail_ndcg@20"] for p in out["static_alpha"]]
    ax[0].plot(x, y, "-s", color="tab:green", label="Static Hybrid, $\\alpha$ 0 to 1 ($\\gamma$=1)", ms=4)
    ax[0].set_xlabel("All-item NDCG@20")
    ax[0].set_ylabel("Long-tail NDCG@20")
    ax[0].set_title("Accuracy vs long-tail frontier")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    for name, key, c in (("LightGCN", "lightgcn", "tab:blue"), ("Projection Hybrid", "projection", "tab:orange")):
        ax[1].plot(GAMMAS, [p["ndcg@20"] for p in out[key]], "-o", color=c, ms=4, label=f"{name}: all-item")
        ax[1].plot(GAMMAS, [p["tail_ndcg@20"] for p in out[key]], "--o", color=c, ms=4, label=f"{name}: long-tail")
    ax[1].set_xlabel("Normalisation exponent $\\gamma$")
    ax[1].set_ylabel("NDCG@20")
    ax[1].set_title("Effect of item-norm exponent")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "tradeoff.png", dpi=200)
    print("saved", OUT / "tradeoff.png")


if __name__ == "__main__":
    main()
