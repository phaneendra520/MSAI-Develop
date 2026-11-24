"""
CineMatch: an explainable, user-steerable hybrid movie recommender.

FOUNDATION / CREDIT
-------------------
This system is built on the design of Rounak Banik's Kaggle notebook
"Movie Recommender Systems" (2017), https://www.kaggle.com/code/rounakbanik/movie-recommender-systems,
also published in Banik (2018), Hands-On Recommendation Systems with Python (Packt).
From that foundation we keep three ideas, re-implemented here for MovieLens 1M:
  1. The IMDB-style Bayesian weighted rating  WR = v/(v+m)*R + m/(v+m)*C
     (the "Simple Recommender").
  2. A content-based filter: vectorise item metadata into a "soup" and rank by
     cosine similarity to a seed title.
  3. The idea of a hybrid that mixes content similarity with collaborative signal.

EXTENSIONS ADDED IN THIS PROJECT (see the APA report for discussion)
  E1. Multi-item user profiles (likes AND dislikes) instead of a single seed title,
      so a brand-new user can get personal recommendations without retraining.
  E2. Latent-factor collaborative filtering (PureSVD; Cremonesi et al., 2010): truncated
      SVD of the user-item rating matrix; new users are "folded in" from the items
      they name, so no retraining is needed.
  E3. A user-tunable hybrid: the person chooses how much weight goes to
      content, collaborative and popularity signals.
  E4. Maximal Marginal Relevance (MMR) re-ranking with a diversity slider.
  E5. Natural-language explanations for every recommendation.
  E6. Hard filters (genre, year range) and fuzzy title matching.
  E7. An offline evaluation harness (evaluate.py) comparing the foundation
      against the extended system.

Data: MovieLens 1M (Harper & Konstan, 2015), GroupLens Research.
"""
from __future__ import annotations

import difflib
import os
import re
import urllib.request
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# Mirror of MovieLens 1M in CSV form (used only if the local data/ folder is empty).
MIRROR = "https://raw.githubusercontent.com/khanhnamle1994/movielens/master/"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _ensure_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name in ("movies.csv", "ratings.csv"):
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(MIRROR + name, path)


def load_movielens():
    _ensure_data()
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"), sep="\t", encoding="latin-1",
                         usecols=["movie_id", "title", "genres"])
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"), sep="\t", encoding="latin-1",
                          usecols=["user_id", "movie_id", "rating", "timestamp"])
    return movies, ratings


def pretty_title(raw: str) -> str:
    """'Matrix, The (1999)' -> 'The Matrix (1999)'."""
    m = re.match(r"^(.*?), (The|A|An|Les|La|Le|Il|El|Das|Der|Die)( \(.*\))? \((\d{4})\)$", raw)
    if m:
        base, art, alt, year = m.groups()
        return f"{art} {base}{alt or ''} ({year})"
    return raw


def _norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(\d{4}\)", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"^(the|a|an) ", "", s.strip())
    return re.sub(r"\s+", " ", s).strip()


# ----------------------------------------------------------------------------
# The recommender
# ----------------------------------------------------------------------------
@dataclass
class Profile:
    """A user's stated tastes. Weights > 0 are likes, < 0 are dislikes."""
    items: dict = field(default_factory=dict)   # movie index -> weight

    def like(self, idx, w=1.0):
        self.items[idx] = w

    def dislike(self, idx, w=-1.0):
        self.items[idx] = w

    def clear(self):
        self.items.clear()


class CineMatch:
    def __init__(self, movies: pd.DataFrame, ratings: pd.DataFrame, n_factors: int = 32,
                 min_ratings: int = 20, seed: int = 42):
        self.n_factors = n_factors
        self.min_ratings = min_ratings
        self.seed = seed
        self.fit(movies, ratings)

    # ---------------------------------------------------------------- fitting
    def fit(self, movies, ratings):
        rated = ratings["movie_id"].unique()
        m = movies[movies["movie_id"].isin(rated)].copy().reset_index(drop=True)
        m["year"] = m["title"].str.extract(r"\((\d{4})\)\s*$").astype(float)
        m["display"] = m["title"].map(pretty_title)
        m["genre_list"] = m["genres"].str.split("|")
        self.movies = m
        self.id2idx = {mid: i for i, mid in enumerate(m["movie_id"])}
        self._norm_titles = [_norm(t) for t in m["display"]]

        # --- Foundation (Banik): Bayesian weighted rating ------------------
        stats = ratings.groupby("movie_id")["rating"].agg(["count", "mean"])
        v = m["movie_id"].map(stats["count"]).fillna(0).to_numpy()
        R = m["movie_id"].map(stats["mean"]).fillna(0).to_numpy()
        C = ratings["rating"].mean()
        mq = np.percentile(v, 90)          # Banik uses the 90th percentile as m
        self.vote_count, self.vote_mean = v, R
        self.wr = (v / (v + mq)) * R + (mq / (v + mq)) * C
        self.pop_score = (self.wr - self.wr.min()) / (self.wr.max() - self.wr.min())

        # --- Foundation (Banik): metadata "soup" + vectoriser --------------
        # ML-1M has no cast/crew/keywords, so the soup is genres + decade + title words.
        def soup(row):
            g = " ".join("g_" + x.replace("'", "").replace("-", "") for x in row.genre_list)
            dec = f"dec_{int(row.year // 10 * 10)}" if not np.isnan(row.year) else ""
            words = _norm(row.display)
            return f"{g} {g} {dec} {words}"   # genres double-weighted
        tfidf = TfidfVectorizer(token_pattern=r"[a-z0-9_]+", min_df=2)
        self.content = normalize(tfidf.fit_transform(m.apply(soup, axis=1)))

        # --- Extension E2: PureSVD latent factors --------------------------
        ratings = ratings[ratings["movie_id"].isin(self.id2idx)]
        users = ratings["user_id"].unique()
        self.uid2row = {u: i for i, u in enumerate(users)}
        rows = ratings["user_id"].map(self.uid2row).to_numpy()
        cols = ratings["movie_id"].map(self.id2idx).to_numpy()
        vals = ratings["rating"].to_numpy().astype(np.float32)
        X = csr_matrix((vals, (rows, cols)), shape=(len(users), len(m)))
        k = min(self.n_factors, min(X.shape) - 1)
        rng = np.random.default_rng(self.seed)
        v0 = rng.standard_normal(min(X.shape))
        _, s, vt = svds(X, k=k, v0=v0)
        self.V = vt.T                                   # items x k, used for scoring
        self.item_factors = normalize(vt.T * s)         # unit length, used for similarity
        self.cf_reliable = self.vote_count >= self.min_ratings
        self.genres = sorted({g for gl in m["genre_list"] for g in gl})

    # ---------------------------------------------------------------- lookup
    def find(self, text: str, n: int = 1, cutoff: float = 0.6):
        """Fuzzy title search. Returns a list of movie indices (best first)."""
        q = _norm(text)
        if not q:
            return []
        year = re.search(r"(19|20)\d{2}", text)
        exact = [i for i, t in enumerate(self._norm_titles) if t == q]
        if exact:
            if year:
                exact = [i for i in exact if str(int(self.movies.year[i])) == year.group(0)] or exact
            return sorted(exact, key=lambda i: -self.vote_count[i])[:n]
        cands = difflib.get_close_matches(q, self._norm_titles, n=max(n, 8), cutoff=cutoff)
        idxs = []
        for c in cands:
            idxs += [i for i, t in enumerate(self._norm_titles) if t == c]
        # substring matches (e.g. "star wars" -> the Star Wars films)
        if len(q) >= 4:
            idxs += [i for i, t in enumerate(self._norm_titles) if q in t and i not in idxs]
        seen, out = set(), []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                out.append(i)
        ratio = lambda i: difflib.SequenceMatcher(None, q, self._norm_titles[i]).ratio()
        starts = lambda i: self._norm_titles[i].startswith(q) and len(q) >= 4
        # titles that start with the query win (e.g. "star wars"), then closeness, then popularity
        out.sort(key=lambda i: (not starts(i), -self.vote_count[i] if starts(i) else 0,
                                -round(ratio(i), 1), -self.vote_count[i]))
        return out[:n]

    def title(self, idx):
        return self.movies.display[idx]

    # ---------------------------------------------------------------- foundation
    def foundation_recommend(self, seed_idx: int, k: int = 10, qualify: bool = True):
        """Re-implementation of Banik's improved content recommender: top-25 cosine
        neighbours of one seed title, then keep the qualified ones (votes >= 60th pct)
        and sort by weighted rating."""
        sims = (self.content @ self.content[seed_idx].T).toarray().ravel()
        sims[seed_idx] = -1
        top = np.argsort(-sims)[:25]
        if qualify:
            thresh = np.percentile(self.vote_count, 60)
            top = [i for i in top if self.vote_count[i] >= thresh]
            top = sorted(top, key=lambda i: -self.wr[i])
        return list(top[:k])

    # ---------------------------------------------------------------- scoring
    def _profile_vectors(self, profile: Profile):
        idx = np.array(list(profile.items.keys()), dtype=int)
        w = np.array(list(profile.items.values()), dtype=float)
        c_vec = np.asarray(self.content[idx].T @ w).ravel()
        f_vec = self.V[idx].T @ w                        # fold-in: u = r V
        return idx, w, c_vec, f_vec

    @staticmethod
    def _minmax(x):
        lo, hi = np.min(x), np.max(x)
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    def score(self, profile: Profile, w_content=0.1, w_cf=0.8, w_pop=0.1):
        """Extension E3: tunable hybrid relevance for every item."""
        if not profile.items:
            return self.pop_score.copy(), {}
        idx, w, c_vec, f_vec = self._profile_vectors(profile)
        content = np.asarray(self.content @ c_vec).ravel()
        cf = self.V @ f_vec                              # PureSVD score: r V V^T
        total = w_content + w_cf + w_pop or 1.0
        rel = (w_content * self._minmax(content) + w_cf * self._minmax(cf)
               + w_pop * self.pop_score) / total
        return rel, {"content": content, "cf": cf}

    def recommend(self, profile: Profile, k=10, w_content=0.1, w_cf=0.8, w_pop=0.1,
                  diversity=0.3, genres=None, year_range=None, exclude=None,
                  min_votes=None, max_votes=None):
        """Full pipeline: score -> filter -> MMR re-rank -> explain."""
        rel, parts = self.score(profile, w_content, w_cf, w_pop)
        mask = np.ones(len(rel), dtype=bool)
        mask[list(profile.items.keys())] = False
        if exclude:
            mask[list(exclude)] = False
        if genres:
            gs = set(genres)
            mask &= self.movies.genre_list.map(lambda gl: bool(gs & set(gl))).to_numpy()
        if year_range:
            lo, hi = year_range
            mask &= self.movies.year.between(lo, hi).to_numpy()
        mv = self.min_ratings if min_votes is None else min_votes
        mask &= self.vote_count >= mv
        if max_votes is not None:                 # "hidden gems": cap popularity
            mask &= self.vote_count <= max_votes
        cand = np.where(mask)[0]
        if len(cand) == 0:
            return []
        cand = cand[np.argsort(-rel[cand])][: max(200, k * 10)]
        chosen = self._mmr(cand, rel, k, diversity)
        return [self.explain(i, profile, rel[i], parts) for i in chosen]

    def _mmr(self, cand, rel, k, diversity):
        """Extension E4: Maximal Marginal Relevance (Carbonell & Goldstein, 1998)."""
        lam = 1.0 - float(diversity)
        if diversity <= 0:
            return list(cand[:k])
        r = self._minmax(rel[cand])
        F = self.item_factors[cand]
        G = self.content[cand]
        sim = 0.5 * (F @ F.T) + 0.5 * (G @ G.T).toarray()   # blended item-item similarity
        chosen, left = [], list(range(len(cand)))
        while left and len(chosen) < k:
            if chosen:
                penalty = sim[np.ix_(left, chosen)].max(axis=1)
            else:
                penalty = np.zeros(len(left))
            mmr = lam * r[left] - (1 - lam) * penalty
            best = left[int(np.argmax(mmr))]
            chosen.append(best)
            left.remove(best)
        return [cand[i] for i in chosen]

    # ---------------------------------------------------------------- explain
    def explain(self, i, profile: Profile, relevance, parts):
        """Extension E5: human-readable reasons."""
        row = self.movies.iloc[i]
        reasons = []
        likes = [j for j, w in profile.items.items() if w > 0]
        if likes:
            cf_sims = self.item_factors[likes] @ self.item_factors[i]
            c_sims = (self.content[likes] @ self.content[i].T).toarray().ravel()
            j_pos = int(np.argmax(cf_sims + 0.5 * c_sims))   # the "because you liked" anchor
            j = likes[j_pos]
            if cf_sims[j_pos] > 0.15:
                reasons.append(f"people who liked {self.title(j)} also rated this highly")
            shared = set(row.genre_list) & set().union(*[set(self.movies.genre_list[l]) for l in likes])
            if shared:
                reasons.append("shares " + ", ".join(sorted(shared)) + " with movies you like")
        dislikes = [j for j, w in profile.items.items() if w < 0]
        if dislikes:
            d = self.item_factors[dislikes] @ self.item_factors[i]
            if d.max() < 0.05:
                reasons.append("it is unlike the titles you disliked")
        if self.wr[i] >= np.percentile(self.wr, 90):
            reasons.append(f"well rated overall ({self.vote_mean[i]:.1f}/5 from {int(self.vote_count[i])} ratings)")
        if not reasons:
            reasons.append("a strong overall match for your profile")
        return {
            "idx": int(i),
            "title": row.display,
            "genres": ", ".join(row.genre_list),
            "year": int(row.year) if not np.isnan(row.year) else None,
            "avg_rating": round(float(self.vote_mean[i]), 2),
            "num_ratings": int(self.vote_count[i]),
            "match": round(float(relevance) * 100),
            "why": (lambda t: t[0].upper() + t[1:])("; ".join(reasons[:3])) + ".",
        }


_ENGINE = None


def get_engine() -> CineMatch:
    """Load data and fit once per process (about 10-20 seconds)."""
    global _ENGINE
    if _ENGINE is None:
        movies, ratings = load_movielens()
        _ENGINE = CineMatch(movies, ratings)
    return _ENGINE
