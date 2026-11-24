"""
CineMatch GUI (Gradio). Two ways to interact with the same recommender:

  1. "Chat" tab: the conversational agent (agent.py).
  2. "Explore & Control" tab: direct-manipulation controls (pick liked/disliked
     titles, move sliders for the hybrid weights and diversity, set filters) with
     explained result cards.

Run:  python app.py       then open http://127.0.0.1:7860
In Google Colab:  !python app.py --share   (gives a public link)
"""
import argparse
import html

import gradio as gr

from agent import HELP, CineMatchAgent
from recommender import Profile, get_engine

ENGINE = get_engine()
TITLES = sorted(ENGINE.movies.display.tolist())
TITLE2IDX = {t: i for i, t in enumerate(ENGINE.movies.display)}

CREDIT = """
### About CineMatch
**Foundation.** CineMatch is built on Rounak Banik's Kaggle notebook *Movie Recommender Systems* (2017)
and his book *Hands-On Recommendation Systems with Python* (2018). From that project it keeps the
IMDB-style weighted rating, the metadata-"soup" content filter with cosine similarity, and the idea of
a hybrid recommender.

**What this version adds.** Multi-title profiles with likes *and* dislikes; PureSVD collaborative
filtering with fold-in for brand-new users; user-adjustable hybrid weights; a diversity slider (MMR
re-ranking); a plain-language reason for every suggestion; genre and year filters; a conversational agent;
and an offline evaluation harness.

**Data.** MovieLens 1M (Harper & Konstan, 2015): 1,000,209 ratings of 3,706 movies by 6,040 users.
Titles run from 1919 to 2000.
"""


def cards(recs):
    if not recs:
        return "<p>No matches. Loosen the filters or add a movie you like.</p>"
    out = []
    for n, r in enumerate(recs, 1):
        out.append(f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:10px 14px;margin:8px 0">
  <div style="display:flex;justify-content:space-between;gap:12px">
    <b>{n}. {html.escape(r['title'])}</b>
    <span style="white-space:nowrap">{r['match']}% match</span>
  </div>
  <div style="background:#eee;border-radius:4px;height:6px;margin:6px 0">
    <div style="background:#4f7cff;width:{max(3, r['match'])}%;height:6px;border-radius:4px"></div>
  </div>
  <div style="font-size:0.9em;opacity:0.8">{html.escape(r['genres'])} · avg {r['avg_rating']}/5 from {r['num_ratings']:,} ratings</div>
  <div style="margin-top:4px"><i>Why:</i> {html.escape(r['why'])}</div>
</div>""")
    return "\n".join(out)


def run_controls(likes, dislikes, w_content, w_cf, w_pop, diversity, genres, y0, y1, k):
    p = Profile()
    for t in likes or []:
        p.like(TITLE2IDX[t])
    for t in dislikes or []:
        p.dislike(TITLE2IDX[t])
    if y0 > y1:
        y0, y1 = y1, y0
    recs = ENGINE.recommend(p, k=int(k), w_content=w_content, w_cf=w_cf, w_pop=w_pop,
                            diversity=diversity, genres=genres or None, year_range=(y0, y1))
    note = ("<p><i>No likes yet, so these are the top-rated titles. Add a few movies you enjoy.</i></p>"
            if not likes else "")
    return note + cards(recs)


def chat(message, history, agent):
    if agent is None:
        agent = CineMatchAgent(ENGINE)
    reply = agent.respond(message)
    history = (history or []) + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": reply}]
    return "", history, agent


def build():
    with gr.Blocks(title="CineMatch") as demo:
        gr.Markdown("# 🎬 CineMatch: an explainable movie recommender\n"
                    "Tell it what you like by chatting, or steer it directly with the controls.")
        with gr.Tab("Chat"):
            agent_state = gr.State(None)
            bot = gr.Chatbot(value=[{"role": "assistant", "content": HELP}], height=480, label="CineMatch")
            box = gr.Textbox(placeholder="e.g. I loved Toy Story and The Matrix", label="Your message")
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Reset conversation")
            gr.Examples(["I loved Toy Story and The Matrix", "something like Alien",
                         "recommend 5 comedies from the 90s", "I didn't like Titanic",
                         "more variety", "hidden gems", "why Aliens?", "profile"], inputs=box)
            box.submit(chat, [box, bot, agent_state], [box, bot, agent_state])
            send.click(chat, [box, bot, agent_state], [box, bot, agent_state])
            clear.click(lambda: ("", [{"role": "assistant", "content": HELP}], None),
                        None, [box, bot, agent_state])

        with gr.Tab("Explore & Control"):
            with gr.Row():
                with gr.Column(scale=1):
                    likes = gr.Dropdown(TITLES, multiselect=True, label="Movies I like",
                                        value=["Toy Story (1995)", "The Matrix (1999)"])
                    dislikes = gr.Dropdown(TITLES, multiselect=True, label="Movies I didn't like")
                    gr.Markdown("**How should CineMatch decide?** (weights are normalised)")
                    w_cf = gr.Slider(0, 1, 0.8, step=0.05, label="People with similar taste (collaborative)")
                    w_content = gr.Slider(0, 1, 0.1, step=0.05, label="Similar genre & era (content)")
                    w_pop = gr.Slider(0, 1, 0.1, step=0.05, label="Overall popularity & rating")
                    diversity = gr.Slider(0, 0.9, 0.3, step=0.05, label="Variety (0 = most similar, higher = more varied)")
                    genres = gr.CheckboxGroup(ENGINE.genres, label="Only these genres (optional)")
                    with gr.Row():
                        y0 = gr.Slider(1919, 2000, 1919, step=1, label="From year")
                        y1 = gr.Slider(1919, 2000, 2000, step=1, label="To year")
                    k = gr.Slider(3, 20, 10, step=1, label="How many")
                    go = gr.Button("Recommend", variant="primary")
                with gr.Column(scale=1):
                    out = gr.HTML()
            inputs = [likes, dislikes, w_content, w_cf, w_pop, diversity, genres, y0, y1, k]
            go.click(run_controls, inputs, out)
            for c in inputs:   # live updates keep the feedback loop tight
                c.change(run_controls, inputs, out)
            demo.load(run_controls, inputs, out)

        with gr.Tab("About & Credits"):
            gr.Markdown(CREDIT)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public Gradio link (Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build().launch(share=args.share, server_port=args.port)
