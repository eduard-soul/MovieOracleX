
#!/usr/bin/env python3
"""-------------------------------------------------------------------
Simple command‑line movie recommender using pre‑trained embeddings
--------------------------------------------------------------------

USAGE
-----
python movie_recommender_cli.py --user alice \
       --embeddings embeddings.npz --mapping movie_mapping.csv

FILES CREATED
-------------
* <user>.csv         : the user’s ratings (movie_id,rating)
* <user>_embed.npz   : the current embedding vector for this user
"""

import argparse
import os
import sys
import readline     # for nicer input()
import numpy as np
import pandas as pd
from difflib import get_close_matches

try:
    import faiss                 # fast ANN search if available
except ImportError:
    faiss = None                 # fallback to numpy brute‑force

# ──────────────────────────────────────────────────────────────────────
def load_embeddings(path):
    data = np.load(path, allow_pickle=True)
    E = data['embeddings'].astype(np.float32)
    movie_ids = data['movie_ids'].tolist()
    id2idx = {m: i for i, m in enumerate(movie_ids)}
    # pre‑normalise for cosine sim
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1
    E = E / norms
    return E, movie_ids, id2idx

def build_index(E):
    if faiss is None:
        return None
    dim = E.shape[1]
    idx = faiss.IndexFlatIP(dim)        # inner‑product (cosine)
    idx.add(E)
    return idx

# ── À la place de load_mapping() ───────────────────────────────────
def load_mapping(path):
    # tout en chaînes, puis on vire les lignes incomplètes
    df = pd.read_csv(path, dtype=str).dropna(subset=['movie_id', 'film_id'])
    df = df[df['film_id'].str.strip() != '']
    id2title = dict(zip(df['movie_id'], df['film_id']))
    title2id = dict(zip(df['film_id'], df['movie_id']))
    return id2title, title2id

def load_user_ratings(user, mapping_inv):
    csv_path = f"{user}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # ensure movie_id canonical
        df['movie_id'] = df['movie_id'].apply(lambda x: mapping_inv.get(x, x))
    else:
        df = pd.DataFrame(columns=['movie_id', 'rating'])
    return df

def save_user_ratings(user, df, mapping):
    csv_path = f"{user}.csv"
    # store human‑readable film_id for convenience
    tmp = df.copy()
    tmp['film_id'] = tmp['movie_id'].map(mapping)
    tmp[['film_id', 'rating']].to_csv(csv_path, index=False)
    print(f"Saved ratings to {csv_path}")

def compute_user_vector(df_ratings, id2idx, E):
    if df_ratings.empty:
        return None
    vec = np.zeros(E.shape[1], dtype=np.float32)
    total = 0.0
    for _, row in df_ratings.iterrows():
        mid = row['movie_id']
        if mid not in id2idx:
            continue
        weight = row['rating'] - 5.0     # center around neutral 5
        vec += weight * E[id2idx[mid]]
        total += abs(weight)
    if total == 0:
        return None
    # normalise
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm

def recommend(user_vec, E, movie_ids, id2idx, rated_set, topk=10, index=None):
    if user_vec is None:
        print("No ratings yet – impossible de recommander.")
        return []
    if index is not None:
        D, I = index.search(user_vec.reshape(1, -1).astype(np.float32), topk + len(rated_set) + 20)
        cand_idx = I[0]
    else:  # brute‑force
        sims = E @ user_vec
        cand_idx = np.argsort(sims)[::-1]
    recs = []
    for idx in cand_idx:
        mid = movie_ids[idx]
        if mid in rated_set:
            continue
        recs.append(mid)
        if len(recs) == topk:
            break
    return recs

# ── À la place de fuzzy_search() ──────────────────────────────────
def fuzzy_search(term, film_titles):
    term_low = term.lower()
    # on garde seulement les vraies chaînes
    titles = [t for t in film_titles if isinstance(t, str)]
    exact = [t for t in titles if term_low in t.lower()]
    if exact:
        return exact[:20]
    return get_close_matches(term, titles, n=20, cutoff=0.4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--user', required=True, help='username / nickname')
    ap.add_argument('--embeddings', default='embeddings.npz')
    ap.add_argument('--mapping', default='movie_mapping.csv')
    args = ap.parse_args()

    E, movie_ids, id2idx = load_embeddings(args.embeddings)
    index = build_index(E)
    map_id2title, map_title2id = load_mapping(args.mapping)
    film_titles = list(map_title2id.keys())

    df_ratings = load_user_ratings(args.user, map_title2id)
    rated_set = set(df_ratings['movie_id'])

    print("\nBienvenue, {} !".format(args.user))
    while True:
        print("\nMenu : [s]earch & rate  |  [r]ecommend  |  [q]uit")
        choice = input("Votre choix ? ").strip().lower()[:1]
        if choice == 'q':
            break

        # ───────────────────────────────────────── Search
        if choice == 's':
            term = input("Tape le titre (ou morceau) du film : ").strip()
            if not term:
                continue
            matches = fuzzy_search(term, film_titles)
            if not matches:
                print("Aucun résultat.")
                continue
            print("\nVoici les titres trouvés :")
            for i, t in enumerate(matches, 1):
                print(f" {i:2d}. {t}")
            sel = input("Numéro du film à noter (ou Enter pour annuler) : ").strip()
            if not sel.isdigit():
                continue
            idx = int(sel) - 1
            if idx < 0 or idx >= len(matches):
                continue
            film_title = matches[idx]
            mid = map_title2id[film_title]
            rating = input("Note de 1 à 10 ? ").strip()
            try:
                rating = float(rating)
            except ValueError:
                continue
            # update df
            df_ratings = df_ratings[df_ratings['movie_id'] != mid]  # overwrite if exists
            df_ratings = pd.concat([df_ratings, pd.DataFrame({'movie_id':[mid],'rating':[rating]})],
                                   ignore_index=True)
            rated_set.add(mid)
            save_user_ratings(args.user, df_ratings, map_id2title)

        # ───────────────────────────────────────── Recommend
        elif choice == 'r':
            user_vec = compute_user_vector(df_ratings, id2idx, E)
            recs = recommend(user_vec, E, movie_ids, id2idx, rated_set, topk=10, index=index)
            if not recs:
                continue
            print("\nJe te conseille :")
            for i, mid in enumerate(recs, 1):
                print(f" {i:2d}. {map_id2title[mid]}")
            # Actions après rec
            print("\n[1‑10] pour noter un film recommandé | [s] chercher autre film | Enter pour menu")
            action = input("> ").strip().lower()
            if action.isdigit():
                idx_choice = int(action) - 1
                if 0 <= idx_choice < len(recs):
                    film_title = map_id2title[recs[idx_choice]]
                    mid = recs[idx_choice]
                    rating = input(f"Ta note pour '{film_title}' ? ").strip()
                    try:
                        rating = float(rating)
                    except ValueError:
                        continue
                    df_ratings = df_ratings[df_ratings['movie_id'] != mid]
                    df_ratings = pd.concat([df_ratings, pd.DataFrame({'movie_id':[mid],'rating':[rating]})],
                                           ignore_index=True)
                    rated_set.add(mid)
                    save_user_ratings(args.user, df_ratings, map_id2title)
            elif action == 's':
                continue

    # sortie – sauvegarde finale + embedding user
    save_user_ratings(args.user, df_ratings, map_id2title)
    user_vec = compute_user_vector(df_ratings, id2idx, E)
    if user_vec is not None:
        np.savez_compressed(f"{args.user}_embed.npz", user_vec=user_vec.astype(np.float32))
        print(f"Embedding utilisateur sauvegardé dans {args.user}_embed.npz")
    print("Au revoir !")

if __name__ == '__main__':
    main()
