"""Rebuild val_split.csv, which was written by LightGcn_implementation.ipynb but never committed.

The notebook partitions ratings.dat into train2 / val / test, so the validation set is exactly
the MovieLens-1M interactions that are in neither train_split.csv nor test_split.csv.

Usage: python build_val_split.py path/to/ratings.dat
"""
import sys

import pandas as pd

from common import ROOT


def main():
    ratings = pd.read_csv(sys.argv[1], sep="::", engine="python", names=["userId", "movieId", "rating", "timestamp"])
    train = pd.read_csv(ROOT / "train_split.csv")
    test = pd.read_csv(ROOT / "test_split.csv")
    key = ["userId", "movieId"]
    assert not ratings.duplicated(key).any()

    # integrity: every train/test row must be a MovieLens-1M interaction with the same timestamp
    for name, df in (("train", train), ("test", test)):
        m = df.merge(ratings, on=key, how="left", suffixes=("", "_ml"))
        assert m.timestamp_ml.notna().all(), f"{name} rows missing from ratings.dat"
        assert (m.timestamp == m.timestamp_ml).all(), f"{name} timestamps differ from ratings.dat"

    used = pd.concat([train[key], test[key]]).assign(used=True)
    val = ratings.merge(used, on=key, how="left")
    val = val[val.used.isna()].drop(columns="used")
    val["rating"] = 1
    val["user"] = val.userId.map(dict(zip(train.userId, train.user)))
    val["item"] = val.movieId.map(dict(zip(train.movieId, train.item)))
    assert val.user.notna().all() and val.item.notna().all()
    val = val.astype({"user": int, "item": int}).sort_values(["user", "timestamp"])[list(train.columns)]

    total = len(train) + len(val) + len(test)
    # the notebook holds out each user's most recent training interactions
    last_train = train.groupby("user").timestamp.max()
    chrono = (val.timestamp >= val.user.map(last_train)).mean()
    print(f"ratings {len(ratings):,} | train {len(train):,} | val {len(val):,} | test {len(test):,} | sum {total:,}")
    print(f"val users {val.user.nunique():,} | share of val rows not older than the user's last train row: {chrono:.4f}")
    assert total == len(ratings) == 1_000_209
    if len(val) != 82_316:
        print("WARNING: the paper reports 82,316 validation interactions")
    val.to_csv(ROOT / "val_split.csv", index=False)
    print("saved", ROOT / "val_split.csv")


if __name__ == "__main__":
    main()
