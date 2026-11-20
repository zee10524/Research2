"""Field-wise frozen BERT encoding for the metadata ablation.

MLM_implementation.ipynb only kept the 64-d random projection of the full-document
embedding, so the raw 768-d vectors have to be regenerated. Every metadata field is encoded
separately with the notebook's recipe (510-token chunks, [CLS] of the last layer). We store
the per-field SUM of chunk [CLS] vectors and the chunk COUNT, so that any subset of fields
can be pooled afterwards as sum(sums) / sum(counts), i.e. the same chunk-level mean pooling
the paper applies to the concatenated document.

Usage: python encode_fields.py [--stats] [--threads N]
Resumable: progress is checkpointed to results/bert_fields_partial.npz.
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import EMB, OUT, ROOT

FIELDS = ("title", "genres", "plot_summary", "tagline", "cast_bios", "crew_bios", "reviews")
PREFIX = {"title": "Title", "genres": "Genres", "plot_summary": "Plot", "tagline": "Tagline",
          "cast_bios": "Cast", "crew_bios": "Crew", "reviews": "Reviews"}
MAX_TOK = 510


def load_movies() -> pd.DataFrame:
    enr = pd.read_csv(ROOT / "Dataset" / "movies_enriched.csv")
    rev = pd.read_csv(ROOT / "Dataset" / "movies_with_reviews_fixed.csv")
    movies = enr.merge(rev[["movie_id", "reviews"]], on="movie_id", how="left")
    for c in FIELDS:
        movies[c] = movies[c].fillna("").astype(str)
    # align to the LightGCN item index (LabelEncoder order == sorted movieId)
    tr = pd.read_csv(ROOT / "train_split.csv", usecols=["movieId", "item"]).drop_duplicates()
    idx = tr.sort_values("item").merge(movies, left_on="movieId", right_on="movie_id", how="left")
    assert idx.movie_id.notna().all() and (idx.item.values == np.arange(len(idx))).all()
    return idx.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    from transformers import BertModel, BertTokenizerFast

    tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
    movies = load_movies()
    n = len(movies)

    # Build every (item, field, chunk) unit. Empty fields contribute no chunk.
    units = []  # (item, field_idx, token ids)
    non_empty = np.zeros((n, len(FIELDS)), bool)
    for fi, f in enumerate(FIELDS):
        texts = [f"{PREFIX[f]}: {t}." if t.strip() else "" for t in movies[f]]
        enc = tok(texts, add_special_tokens=False, verbose=False)["input_ids"]
        for it, ids in enumerate(enc):
            non_empty[it, fi] = len(ids) > 0
            for s in range(0, len(ids), MAX_TOK):
                units.append((it, fi, ids[s:s + MAX_TOK]))
    lens = np.array([len(u[2]) for u in units])
    print(f"items={n} chunks={len(units)} tokens={lens.sum():,}")
    for fi, f in enumerate(FIELDS):
        sel = np.array([u[1] == fi for u in units])
        print(f"  {f:13s} non-empty={non_empty[:, fi].sum():5d} chunks={sel.sum():6d} tokens={lens[sel].sum():,}")
    if args.stats:
        return

    bert = BertModel.from_pretrained("bert-base-uncased").eval()
    part = OUT / "bert_fields_partial.npz"
    sums = np.zeros((n, len(FIELDS), 768), np.float32)
    counts = np.zeros((n, len(FIELDS)), np.int32)
    done = 0
    order = np.argsort(-lens, kind="stable")  # longest first: similar lengths share a batch
    if part.exists():
        z = np.load(part)
        sums, counts, done = z["sums"], z["counts"], int(z["done"])
        print(f"resuming at {done}/{len(units)}")

    t0, last = time.perf_counter(), time.perf_counter()
    start = done
    with torch.inference_mode():
        while done < len(units):
            idx = order[done:done + args.batch]
            batch = [[tok.cls_token_id] + units[j][2] + [tok.sep_token_id] for j in idx]
            m = max(map(len, batch))
            ids = torch.tensor([b + [tok.pad_token_id] * (m - len(b)) for b in batch])
            att = torch.tensor([[1] * len(b) + [0] * (m - len(b)) for b in batch])
            cls = bert(input_ids=ids, attention_mask=att).last_hidden_state[:, 0].numpy()
            for j, v in zip(idx, cls):
                sums[units[j][0], units[j][1]] += v
                counts[units[j][0], units[j][1]] += 1
            done += len(idx)
            if time.perf_counter() - last > 300 or done >= len(units):
                np.savez(part, sums=sums, counts=counts, done=done)
                el = time.perf_counter() - t0
                tok_done = lens[order[start:done]].sum()
                eta = el / max(tok_done, 1) * lens[order[done:]].sum()
                print(f"{done}/{len(units)} chunks | {el/60:.1f} min | eta {eta/60:.1f} min", flush=True)
                last = time.perf_counter()

    np.savez_compressed(EMB / "bert_fields_768.npz", sums=sums, counts=counts, fields=np.array(FIELDS))
    print(f"saved {EMB / 'bert_fields_768.npz'} in {(time.perf_counter() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
