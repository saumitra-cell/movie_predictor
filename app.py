import json
from pathlib import Path

import pandas as pd
import streamlit as st

from download_data import download
from engine import Engine

st.set_page_config(page_title="Movie Taste Explorer", page_icon="🎬", layout="centered")


@st.cache_resource(show_spinner="Loading data and training the model (first run only)...")
def get_engine():
    download()
    return Engine()


eng = get_engine()
titles = {m: eng.title_of(m) for m in eng.pickable()}

st.title("🎬 Movie Taste Explorer")
st.caption("Pick movies you love. Get recommendations that explain themselves, and tune how adventurous they are.")

liked = st.multiselect("Movies you like (pick 3 or more)", options=list(titles),
                       format_func=lambda m: titles[m], placeholder="Start typing a title...")
query = st.text_input("Optional: describe the mood or genre you want",
                      placeholder="e.g. dark sci-fi, feel-good animation, crime thriller")

with st.sidebar:
    st.header("Tune the list")
    variety = st.slider("Variety", 0.0, 1.0, 0.4,
                        help="Low = the closest matches. High = a more varied mix of genres and styles "
                             "(re-ranks with MMR, evaluated in the project).")
    gems = st.slider("Mainstream ↔ hidden gems", 0.0, 1.5, 0.0, 0.1,
                     help="Pushes down very popular titles. Experimental: not part of the offline evaluation.")
    n = st.slider("How many results", 5, 20, 10)

if len(liked) == 0 and not query.strip():
    st.info("Choose a few movies above (or describe what you want) to get recommendations.")
else:
    if 0 < len(liked) < 3 and not query.strip():
        st.warning("Results get much better with 3 or more movies.")
    recs = eng.recommend(liked, query, n=n, lam=1 - 0.5 * variety, gems=gems)
    if not recs:
        st.error("I couldn't match that description. Try genre words like 'comedy' or 'sci-fi'.")
    for i, r in enumerate(recs, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {r['title']}**")
            st.caption(r["genres"])
            why = []
            if r["because"]:
                why.append(f"Because you liked **{r['because']}**")
            if r["shared"]:
                why.append("shared themes: " + ", ".join(r["shared"]))
            if why:
                st.write(" · ".join(why))

with st.expander("How it works and how well it performs"):
    st.markdown(
        "- **Collaborative part:** an SVD model learned from 100,000 ratings by 610 MovieLens users. "
        "Your picks are folded into it as a new user.\n"
        "- **Content part:** TF-IDF over genres, user tags and release decade, plus your text description.\n"
        "- **Variety slider:** MMR re-ranking trades a little relevance for a more varied list.\n"
        "- **Explanations:** the liked movie closest to each result, and the genres/tags they share."
    )
    f = Path("results.json")
    if f.exists():
        m = json.loads(f.read_text())["mean"]
        keep = ["Popularity", "SVD", "Hybrid", "Hybrid + MMR (lambda=0.8)"]
        df = pd.DataFrame({k: m[k] for k in keep}).T[["ndcg@10", "recall@10", "diversity"]].round(3)
        st.caption("Offline test on held-out ratings (mean of 3 splits):")
        st.table(df)

st.caption("Data: MovieLens ml-latest-small (GroupLens). Harper & Konstan, ACM TiiS 2015.")
