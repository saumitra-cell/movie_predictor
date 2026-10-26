"""Core library: features, models, re-ranking and metrics for the hybrid movie recommender."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA = Path("data")
LIKE_THRESHOLD = 4.0  # a rating >= 4 counts as "liked"


# ---------------------------------------------------------------- data
def load_data():
    ratings = pd.read_csv(DATA / "ratings.csv")
    movies = pd.read_csv(DATA / "movies.csv")
    tags = pd.read_csv(DATA / "tags.csv")
    return ratings, movies, tags


def item_features(movies, tags, return_vectorizer=False):
    """TF-IDF over genres, user tags and release decade -> (n_items x n_terms), L2-normalised."""
    genres = (movies["genres"].str.lower().str.replace("(no genres listed)", "", regex=False)
              .str.replace("-", "_", regex=False).str.replace("|", " ", regex=False))
    year = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    decade = year.apply(lambda y: "" if np.isnan(y) else f"decade_{int(y // 10 * 10)}")
    t = tags.assign(tag=tags["tag"].astype(str).str.lower().str.strip().str.replace(r"[ -]", "_", regex=True))
    tag_text = t.groupby("movieId")["tag"].apply(" ".join)
    tag_col = movies["movieId"].map(tag_text).fillna("")
    docs = (genres + " ") * 2 + decade + " " + tag_col
    vec = TfidfVectorizer(sublinear_tf=True, token_pattern=r"[A-Za-z0-9_]{2,}")
    X = normalize(vec.fit_transform(docs))
    if return_vectorizer:
        return X.tocsr(), vec
    return X.tocsr(), vec.get_feature_names_out()


def split_ratings(df, test_frac=0.2, seed=42):
    """Per-user random hold-out: every user keeps ~80% of ratings for training."""
    df = df.sample(frac=1, random_state=seed)
    n = df.groupby("userId")["movieId"].transform("size")
    rank = df.groupby("userId").cumcount()
    is_test = rank < (n * test_frac).round()
    return df[~is_test], df[is_test]


def to_matrix(df, user_idx, item_idx, n_users, n_items, binary=False):
    rows = df["userId"].map(user_idx).to_numpy()
    cols = df["movieId"].map(item_idx).to_numpy()
    vals = np.ones(len(df)) if binary else df["rating"].to_numpy()
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))


# ---------------------------------------------------------------- models
def popularity_scores(R):
    counts = np.asarray((R > 0).sum(axis=0)).ravel().astype(np.float32)
    return np.tile(counts, (R.shape[0], 1))


def svd_scores(R, k=50, seed=0):
    """PureSVD: truncated SVD of the user-item rating matrix (missing = 0)."""
    svd = TruncatedSVD(n_components=k, random_state=seed)
    U = svd.fit_transform(R)
    return (U @ svd.components_).astype(np.float32)


def content_scores(R, X):
    """Profile = sum of the item vectors of movies the user liked; score = cosine to every item."""
    liked = (R >= LIKE_THRESHOLD).astype(float)
    profile = normalize(liked @ X)
    return (profile @ X.T).toarray().astype(np.float32)


def zscore(S, seen):
    """Standardise each user's scores over the items they have not seen yet."""
    ok = ~seen
    n = ok.sum(axis=1, keepdims=True)
    mu = (S * ok).sum(axis=1, keepdims=True) / n
    var = (((S - mu) ** 2) * ok).sum(axis=1, keepdims=True) / n
    return (S - mu) / np.where(var > 0, np.sqrt(var), 1.0)


# ---------------------------------------------------------------- ranking
def top_k(S, seen, k):
    S2 = np.where(seen, -np.inf, S)
    idx = np.argpartition(-S2, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(S2, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def mmr_rerank(S, seen, X, n=10, pool=100, lam=0.8):
    """Maximal Marginal Relevance: trade a little relevance for a more varied list.
    lam=1 -> pure relevance, lower lam -> more diversity."""
    pools = top_k(S, seen, pool)
    out = np.zeros((S.shape[0], n), dtype=int)
    for u in range(S.shape[0]):
        cand = pools[u]
        sc = S[u, cand]
        sc = (sc - sc.min()) / (sc.max() - sc.min() + 1e-9)
        sim = (X[cand] @ X[cand].T).toarray()
        chosen = [0]
        while len(chosen) < n:
            redundancy = sim[:, chosen].max(axis=1)
            mmr = lam * sc - (1 - lam) * redundancy
            mmr[chosen] = -np.inf
            chosen.append(int(mmr.argmax()))
        out[u] = cand[chosen]
    return out


# ---------------------------------------------------------------- metrics
def evaluate(rec, test_rel, X, item_pop, n_catalog):
    """rec: (users x n) item indices. test_rel: dense bool (users x items) of held-out liked items."""
    n = rec.shape[1]
    n_rel = test_rel.sum(axis=1)
    valid = n_rel > 0
    hits = np.take_along_axis(test_rel, rec, axis=1)
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = (hits * disc).sum(axis=1)
    idcg = np.array([disc[: min(int(r), n)].sum() for r in n_rel])
    ild = []
    for u in np.where(valid)[0]:
        sim = (X[rec[u]] @ X[rec[u]].T).toarray()
        ild.append(1 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))
    return {
        "precision@10": float((hits.sum(axis=1) / n)[valid].mean()),
        "recall@10": float((hits.sum(axis=1) / np.maximum(n_rel, 1))[valid].mean()),
        "ndcg@10": float((dcg / np.maximum(idcg, 1e-9))[valid].mean()),
        "coverage": float(len(np.unique(rec[valid])) / n_catalog),
        "novelty": float((-np.log2(item_pop[rec[valid]])).mean()),
        "diversity": float(np.mean(ild)),
    }
