# Movie Taste Explorer: a hybrid recommender with explanations

**Live app:** _add your Streamlit link here_

Pick movies you like (and optionally describe a mood). The app recommends films, explains each one
("Because you liked X · shared themes: ..."), and lets you tune variety and popularity.

## How it works
- **Collaborative filtering:** PureSVD on 100k MovieLens ratings. A new user's picks are folded into the model.
- **Content model:** TF-IDF over genres, user tags and release decade; also powers the text description box.
- **Hybrid:** per-user z-scored blend of the two (weight tuned on a validation split).
- **Variety slider:** MMR re-ranking trades a little relevance for a more varied list.
- **Explanations:** nearest liked movie (latent + content similarity) and the terms they share.

## Run
```bash
pip install -r requirements.txt
python evaluate.py          # optional: reproduces results.json (run `python download_data.py` first)
streamlit run app.py        # downloads MovieLens on first run
```
(`python download_data.py` fetches the data manually.)

## Offline evaluation (mean of 3 random per-user 80/20 splits, top-10, relevant = held-out rating >= 4)

| Model | NDCG@10 | Recall@10 | Coverage | Novelty | Diversity |
|---|---|---|---|---|---|
| Popularity | 0.161 | 0.103 | 0.6% | 1.67 | 0.932* |
| Content (TF-IDF) | 0.008 | 0.008 | 18.6% | 7.25 | 0.141 |
| SVD | 0.273 | 0.202 | 5.1% | 2.40 | 0.893 |
| Hybrid | 0.275 | 0.202 | 5.9% | 2.51 | 0.852 |
| Hybrid + MMR (0.8) | 0.278 | 0.205 | 5.7% | 2.47 | 0.880 |
| Hybrid + MMR (0.6) | 0.276 | 0.201 | 5.6% | 2.44 | 0.907 |

*Popularity gives every user the same list, so its diversity is not meaningful.

## Honest findings and limits
- SVD beats the popularity baseline by about 70% in NDCG@10.
- The hybrid is not significantly better than SVD alone; the difference is within split-to-split noise.
- MMR raises list diversity (0.852 -> 0.907) with no meaningful accuracy loss. Diversity is measured with the same
  genre/tag features MMR uses, so part of that gain is by construction.
- The content model only knows genres, tags (1,572 of 9,742 movies) and decade, so text search works best on genre/mood words.
- The text-query boost and the "hidden gems" slider are not part of the offline evaluation.
- Small dataset (610 users); popularity bias remains: only ~6% of the catalog ever appears in a top-10.

## Data
MovieLens ml-latest-small (GroupLens), downloaded at runtime and not redistributed here.
Harper & Konstan (2015), The MovieLens Datasets: History and Context, ACM TiiS 5(4).
