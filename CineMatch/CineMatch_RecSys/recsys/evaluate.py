"""
Offline evaluation: foundation (Banik-style) vs. the extended CineMatch system.

Protocol
  * Users with >= 30 ratings are eligible. A tuning sample (300 users, seed 7) was used
    only to choose the default hybrid weights; the evaluation sample (600 users,
    seed 2026) is drawn from the remaining users, so tuning and testing do not overlap.
  * Per-user temporal split: each evaluation user's earliest 80% of ratings are
    training, the latest 20% are test. All other users' ratings stay in training.
  * The model is fitted on training ratings only (no leakage).
  * Relevant items = test items the user rated 4 or 5.
  * Two scenarios:
      - Warm start: the profile is the user's full training history.
      - Cold start: the profile is only the user's 3 most recent liked (>= 4) titles,
        which mimics a new person typing three favourites into the GUI or chatbot.

Metrics: Precision@10, Recall@10, NDCG@10, HitRate@10, intra-list diversity (ILD),
catalog coverage, and novelty (mean self-information of recommended items).

Run:  python evaluate.py            (about 2-3 minutes)
"""
import json
import time

import numpy as np
import pandas as pd

from recommender import CineMatch, Profile, load_movielens

K = 10


def split(ratings, n_users=600, seed=2026, tuning_seed=7, n_tuning=300):
    counts = ratings.groupby("user_id").size()
    eligible = counts[counts >= 30].index.to_numpy()
    tuning = np.random.default_rng(tuning_seed).choice(eligible, size=n_tuning, replace=False)
    pool = np.setdiff1d(eligible, tuning)
    users = np.random.default_rng(seed).choice(pool, size=n_users, replace=False)
    r = ratings.sort_values(["user_id", "timestamp"])
    test_parts = []
    for _, g in r[r.user_id.isin(users)].groupby("user_id"):
        n_test = max(1, int(round(len(g) * 0.2)))
        test_parts.append(g.iloc[-n_test:])
    test = pd.concat(test_parts)
    train = r.drop(test.index)
    return train, test, users


def ndcg(rec, rel):
    dcg = sum(1 / np.log2(i + 2) for i, x in enumerate(rec) if x in rel)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), len(rec))))
    return dcg / idcg if idcg else 0.0


def ild(eng, rec):
    """Intra-list diversity = 1 - mean pairwise content (genre/decade) similarity."""
    if len(rec) < 2:
        return 0.0
    G = eng.content[rec]
    S = (G @ G.T).toarray()
    n = len(rec)
    return float(1 - (S.sum() - n) / (n * (n - 1)))


HYBRID = dict(w_content=0.1, w_cf=0.8, w_pop=0.1)
SYSTEMS = {
    "Popularity (weighted rating)": "pop",
    "Foundation: Banik content, single seed": "foundation",
    "E1 Content profile": dict(w_content=1, w_cf=0, w_pop=0, diversity=0),
    "E2 PureSVD CF profile": dict(w_content=0, w_cf=1, w_pop=0, diversity=0),
    "E3 Hybrid (0.1/0.8/0.1)": dict(**HYBRID, diversity=0),
    "E3+E4 Hybrid + MMR (div=0.15)": dict(**HYBRID, diversity=0.15),
    "E3+E4 Hybrid + MMR (div=0.30)": dict(**HYBRID, diversity=0.30),
    "E3+E4 Hybrid + MMR (div=0.50)": dict(**HYBRID, diversity=0.50),
}


def run(eng, users, train_by_user, test_by_user, scenario, pop_prob):
    pop_rank = np.argsort(-eng.wr)
    thresh = np.percentile(eng.vote_count, 60)
    res = {n: {"p": [], "r": [], "ndcg": [], "hit": [], "ild": [], "nov": [], "items": set()}
           for n in SYSTEMS}
    for u in users:
        tr, te = train_by_user[u], test_by_user[u]
        rel = {eng.id2idx[m] for m, s in zip(te.movie_id, te.rating) if s >= 4 and m in eng.id2idx}
        if not rel:
            continue
        seen = {eng.id2idx[m] for m in tr.movie_id if m in eng.id2idx}
        if scenario == "cold":
            liked = tr[tr.rating >= 4]
            if len(liked) < 3:
                continue
            liked = liked.iloc[-3:]
            prof_items = {eng.id2idx[m]: 1.0 for m in liked.movie_id if m in eng.id2idx}
            seed_movie = liked.iloc[-1].movie_id
        else:
            prof_items = {eng.id2idx[m]: float(s) / 5.0 for m, s in zip(tr.movie_id, tr.rating)
                          if m in eng.id2idx}
            seed_movie = tr[tr.rating == tr.rating.max()].iloc[-1].movie_id
        prof = Profile(items=prof_items)

        for name, cfg in SYSTEMS.items():
            if cfg == "pop":
                rec = [i for i in pop_rank if i not in seen][:K]
            elif cfg == "foundation":
                s_idx = eng.id2idx[seed_movie]
                sims = (eng.content @ eng.content[s_idx].T).toarray().ravel()
                sims[list(seen)] = -1
                top = np.argsort(-sims)[:25]
                top = sorted([i for i in top if eng.vote_count[i] >= thresh], key=lambda i: -eng.wr[i])
                rec = top[:K]
            else:
                rec = [o["idx"] for o in eng.recommend(prof, k=K, exclude=seen, **cfg)]
            r = res[name]
            hits = len(set(rec) & rel)
            r["p"].append(hits / K)
            r["r"].append(hits / len(rel))
            r["ndcg"].append(ndcg(rec, rel))
            r["hit"].append(1.0 if hits else 0.0)
            r["ild"].append(ild(eng, rec))
            if len(rec):
                r["nov"].append(float(np.mean(-np.log2(np.clip(pop_prob[list(rec)], 1e-6, 1)))))
            r["items"].update(rec)
    rows = []
    for name, r in res.items():
        rows.append({
            "Scenario": scenario, "System": name, "Users": len(r["p"]),
            "Precision@10": np.mean(r["p"]), "Recall@10": np.mean(r["r"]),
            "NDCG@10": np.mean(r["ndcg"]), "HitRate@10": np.mean(r["hit"]),
            "ILD": np.mean(r["ild"]), "Coverage": len(r["items"]) / len(eng.movies),
            "Novelty (bits)": np.mean(r["nov"]),
        })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    movies, ratings = load_movielens()
    train, test, users = split(ratings)
    eng = CineMatch(movies, train)
    pop_prob = eng.vote_count / train.user_id.nunique()
    train_by_user = {u: g for u, g in train[train.user_id.isin(users)].groupby("user_id")}
    test_by_user = {u: g for u, g in test.groupby("user_id")}
    print(f"Fitted on {len(train):,} training ratings; evaluating {len(users)} users")

    frames = []
    for scenario in ("warm", "cold"):
        df = run(eng, users, train_by_user, test_by_user, scenario, pop_prob)
        frames.append(df)
        print(f"\n=== {scenario.upper()} START ===")
        print(df.drop(columns="Scenario").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    out = pd.concat(frames)
    out.to_csv("evaluation_results.csv", index=False)
    with open("evaluation_meta.json", "w") as f:
        json.dump({"train_ratings": int(len(train)), "test_ratings": int(len(test)),
                   "catalog": int(len(eng.movies)), "seconds": round(time.time() - t0, 1)}, f, indent=2)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    main()
