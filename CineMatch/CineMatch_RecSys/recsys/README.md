# CineMatch: an explainable, user-steerable hybrid movie recommender

MSAI-631 Artificial Intelligence for Human-Computer Interaction, recommendation system project.

## Credit: the foundation this project builds on
This project is built on **Rounak Banik's Kaggle notebook "Movie Recommender Systems" (2017)**,
https://www.kaggle.com/code/rounakbanik/movie-recommender-systems, also published in
Banik, R. (2018), *Hands-On Recommendation Systems with Python*, Packt.
Kept from the foundation (re-implemented for MovieLens 1M):
* the IMDB-style Bayesian weighted rating (the "Simple Recommender"),
* the metadata "soup" + vectoriser + cosine-similarity content filter seeded by one title,
* the idea of combining content and collaborative signals in a hybrid.

## What was changed / extended
| # | Extension | Where |
|---|-----------|-------|
| E1 | Multi-title profiles with likes **and** dislikes (not one seed title) | `recommender.py` `Profile`, `score()` |
| E2 | PureSVD latent-factor collaborative filtering with fold-in for brand-new users (no retraining) | `recommender.py` `fit()` |
| E3 | User-adjustable hybrid weights (collaborative / content / popularity) | `recommender.py` `score()` |
| E4 | Diversity slider via Maximal Marginal Relevance re-ranking | `recommender.py` `_mmr()` |
| E5 | A plain-language "why" for every recommendation | `recommender.py` `explain()` |
| E6 | Conversational agent + Gradio GUI, genre/year filters, fuzzy title search | `agent.py`, `app.py` |
| E7 | Offline evaluation harness (warm and cold start) | `evaluate.py` |

## Data
MovieLens 1M (Harper & Konstan, 2015): 1,000,209 ratings, 3,706 rated movies, 6,040 users.
The CSVs are in `data/`. If that folder is empty the code downloads them automatically.

## Run it
```bash
pip install -r requirements.txt
python app.py            # GUI at http://127.0.0.1:7860  (Chat + Explore & Control tabs)
python agent.py          # text-prompt chatbot in the terminal
python evaluate.py       # offline evaluation (about 2 minutes) -> evaluation_results.csv
python make_figure.py    # figures/results.png
```
Google Colab: upload the folder, `!pip install gradio`, then `!python app.py --share`.

## Example chat
```
You: I loved Toy Story and The Matrix
You: recommend 3 comedies from the 90s
You: I didn't like Titanic
You: more variety
You: hidden gems
You: something like Alien
You: why Aliens?
```

## Headline results (600 held-out users, temporal split, top-10)
| System | Warm P@10 | Cold P@10 | Cold ILD |
|---|---|---|---|
| Popularity | 0.079 | 0.079 | 0.86 |
| Foundation (Banik, single seed) | 0.038 | 0.039 | 0.32 |
| E2 PureSVD CF | 0.113 | 0.127 | 0.74 |
| E3 Hybrid | 0.113 | 0.122 | 0.71 |
| E3+E4 Hybrid + MMR 0.30 | 0.114 | 0.123 | 0.76 |
