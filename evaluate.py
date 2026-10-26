"""Compare popularity, content, SVD, hybrid and hybrid+MMR over several random splits."""
import json
import sys

import numpy as np
import pandas as pd

from recsys import *

SEEDS = [42, 43, 44]
K = 10


def prep(train, test, ids, X):
    users, items, n_cat = ids
    nu, ni = len(users), len(items)
    R = to_matrix(train, users, items, nu, ni)
    seen = R.toarray() > 0
    rel = to_matrix(test[test["rating"] >= LIKE_THRESHOLD], users, items, nu, ni, binary=True).toarray() > 0
    pop = np.maximum(np.asarray((R > 0).sum(axis=0)).ravel(), 1) / nu
    return R, seen, rel, pop


def metrics_for(S, seen, rel, pop, X, n_cat):
    return evaluate(top_k(S, seen, K), rel, X, pop, n_cat)


def run(seed, ratings, movies, X, ids):
    users, items, n_cat = ids
    train, test = split_ratings(ratings, 0.2, seed)

    # ---- tune k and the blend weight on a validation split carved out of TRAIN only
    tr_sub, val = split_ratings(train, 0.2, seed + 100)
    R, seen, rel, pop = prep(tr_sub, val, ids, X)
    C = zscore(content_scores(R, X), seen)
    best_k = max([20, 50, 100], key=lambda k: metrics_for(svd_scores(R, k), seen, rel, pop, X, n_cat)["ndcg@10"])
    Z = zscore(svd_scores(R, best_k), seen)
    best_a = max([0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                 key=lambda a: metrics_for(a * Z + (1 - a) * C, seen, rel, pop, X, n_cat)["ndcg@10"])

    # ---- final evaluation on the untouched test split
    R, seen, rel, pop = prep(train, test, ids, X)
    Zs = zscore(svd_scores(R, best_k), seen)
    Cs = zscore(content_scores(R, X), seen)
    H = best_a * Zs + (1 - best_a) * Cs
    res = {
        "Popularity": metrics_for(popularity_scores(R), seen, rel, pop, X, n_cat),
        "Content (TF-IDF)": metrics_for(Cs, seen, rel, pop, X, n_cat),
        "SVD": metrics_for(Zs, seen, rel, pop, X, n_cat),
        "Hybrid": metrics_for(H, seen, rel, pop, X, n_cat),
    }
    for lam in [0.9, 0.8, 0.7, 0.6]:
        res[f"Hybrid + MMR (lambda={lam})"] = evaluate(mmr_rerank(H, seen, X, K, 100, lam), rel, X, pop, n_cat)
    return res, {"k": best_k, "alpha": best_a}


def main():
    ratings, movies, tags = load_data()
    X, _ = item_features(movies, tags)
    users = {u: i for i, u in enumerate(sorted(ratings["userId"].unique()))}
    items = {m: i for i, m in enumerate(movies["movieId"])}
    ids = (users, items, len(movies))

    runs, chosen = [], []
    for s in SEEDS:
        r, c = run(s, ratings, movies, X, ids)
        runs.append(r); chosen.append(c)
        print(f"seed {s} done, tuned: {c}", flush=True)

    names = list(runs[0].keys())
    cols = list(runs[0][names[0]].keys())
    mean = {n: {c: float(np.mean([r[n][c] for r in runs])) for c in cols} for n in names}
    std = {n: {c: float(np.std([r[n][c] for r in runs])) for c in cols} for n in names}
    json.dump({"mean": mean, "std": std, "tuned": chosen, "seeds": SEEDS}, open("results.json", "w"), indent=2)
    print("\nMean over", len(SEEDS), "splits")
    print(pd.DataFrame(mean).T.round(4).to_string())
    print("\nStd")
    print(pd.DataFrame(std).T.round(4).to_string())


if __name__ == "__main__":
    main()
