"""Recommendation engine for a NEW user: fold a handful of liked movies into the trained model."""
import re

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from recsys import item_features, load_data, mmr_rerank, to_matrix

SYNONYMS = {
    r"sci[- ]?fi|science fiction": "sci_fi", r"film[- ]noir|film noir": "film_noir",
    r"rom[- ]?com": "romance comedy", r"scary": "horror", r"funny": "comedy",
    r"animated|cartoon": "animation", r"kids|family": "children",
}


def clean_query(text):
    t = text.lower()
    for pat, rep in SYNONYMS.items():
        t = re.sub(pat, rep, t)
    return t.replace("-", "_")


def pretty(term):
    return term.replace("_", " ")


class Engine:
    def __init__(self, k=50, alpha=0.6, min_ratings=5):
        ratings, movies, tags = load_data()
        self.movies = movies.reset_index(drop=True)
        self.X, self.vec = item_features(self.movies, tags, return_vectorizer=True)
        self.terms = self.vec.get_feature_names_out()
        users = {u: i for i, u in enumerate(sorted(ratings["userId"].unique()))}
        self.item_idx = {m: i for i, m in enumerate(self.movies["movieId"])}
        self.n_items = len(self.movies)
        R = to_matrix(ratings, users, self.item_idx, len(users), self.n_items)
        self.counts = np.asarray((R > 0).sum(axis=0)).ravel()
        svd = TruncatedSVD(n_components=k, random_state=0).fit(R)
        self.V = svd.components_.astype(np.float32)          # k x items
        self.Vn = normalize(self.V.T)                         # items x k (for cosine)
        self.titles = self.movies["title"].to_numpy()
        self.genres = self.movies["genres"].to_numpy()
        self.alpha, self.min_ratings = alpha, min_ratings

    def pickable(self):
        """Movie ids a user can choose from, most-rated first."""
        ok = np.where(self.counts >= self.min_ratings)[0]
        ok = ok[np.argsort(-self.counts[ok])]
        return [int(self.movies["movieId"].iloc[i]) for i in ok]

    def title_of(self, movie_id):
        return self.titles[self.item_idx[movie_id]]

    def recommend(self, liked_ids, query="", n=10, lam=0.8, gems=0.0, query_strength=3.0):
        liked = [self.item_idx[m] for m in liked_ids]
        q = None
        if query.strip():
            q = self.vec.transform([clean_query(query)]).toarray().ravel()
            q = q / np.linalg.norm(q) if np.linalg.norm(q) > 0 else None
        if not liked and q is None:
            return []

        seen = np.zeros(self.n_items, bool)
        seen[liked] = True
        cand = (self.counts >= self.min_ratings) & ~seen

        def z(s):
            return (s - s[cand].mean()) / (s[cand].std() + 1e-9)

        # content part: liked-movie profile + optional text query
        c = np.zeros(self.X.shape[1])
        if liked:
            prof = np.asarray(self.X[liked].sum(axis=0)).ravel()
            c += prof / (np.linalg.norm(prof) + 1e-9)
        content = self.X @ c

        if liked:
            r = np.zeros(self.n_items, np.float32)
            r[liked] = 5.0
            cf = (r @ self.V.T) @ self.V                      # fold-in for a new user
            a = self.alpha
            score = a * z(cf) + (1 - a) * z(content)
        else:
            score = z(content)
        if q is not None:
            score = score + query_strength * z(self.X @ q)
        if gems:
            score = score - gems * z(np.log1p(self.counts))
        score = np.where(cand, score, score[cand].min() - 1)

        rec = mmr_rerank(score[None, :], (~cand)[None, :], self.X, n=n, pool=100, lam=lam)[0]
        return [self._explain(int(i), liked, q) for i in rec]

    def _explain(self, i, liked, q):
        info = {"title": self.titles[i], "genres": self.genres[i].replace("|", " · "),
                "n_ratings": int(self.counts[i]), "because": None, "shared": []}
        xi = self.X[i]
        if liked:
            lat = self.Vn[liked] @ self.Vn[i]
            con = (self.X[liked] @ xi.T).toarray().ravel()
            j = liked[int(np.argmax(0.5 * lat + 0.5 * con))]
            info["because"] = self.titles[j]
            overlap = xi.multiply(self.X[j]).toarray().ravel()
        else:
            overlap = xi.toarray().ravel() * q
        for t in np.argsort(-overlap)[:6]:
            if overlap[t] > 0 and not self.terms[t].startswith("decade_"):
                info["shared"].append(pretty(self.terms[t]))
            if len(info["shared"]) == 3:
                break
        return info
