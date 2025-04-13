import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# Étape 1 : Charger les données nécessaires
print("Chargement des données...")

# Charger la matrice des facteurs latents des films
item_latent_matrix = np.loadtxt('item_latent_matrix.csv', delimiter=',')
print(f"Matrice latente chargée : {item_latent_matrix.shape[0]} films avec {item_latent_matrix.shape[1]} facteurs")

# Charger le mapping des IDs de films
movie_id_to_index = {}
with open('movie_id_mapping.txt', 'r') as f:
    for line in f:
        movie_id, idx = line.strip().split(',')
        movie_id_to_index[int(movie_id)] = int(idx)
print(f"Mapping chargé : {len(movie_id_to_index)} films")

# Charger les informations sur les films
movies_df = pd.read_csv('movies.csv')
print(f"Fichier movies.csv chargé : {len(movies_df)} films")

# Étape 2 : Fonction pour rechercher un film
def search_movie(search_term):
    matches = movies_df[movies_df['title'].str.contains(search_term, case=False, na=False)]
    if len(matches) == 0:
        print("Aucun film trouvé.")
        return None
    elif len(matches) > 5:
        print(f"Trop de résultats ({len(matches)}), voici les 5 premiers :")
        return matches.head(5)
    else:
        print("Films trouvés :")
        return matches

# Étape 3 : Fonction pour générer des recommandations
def generate_recommendations(user_ratings, num_recommendations=5):
    n_factors = item_latent_matrix.shape[1]
    user_factors = np.zeros(n_factors)
    for movie_id, rating in user_ratings.items():
        movie_idx = movie_id_to_index[movie_id]
        user_factors += (rating - 3) * item_latent_matrix[movie_idx]  # Centrage autour de 3
    if np.linalg.norm(user_factors) == 0:
        return []  # Pas assez de données
    user_factors /= np.linalg.norm(user_factors)  # Normalisation

    predictions = np.dot(item_latent_matrix, user_factors)
    sorted_indices = np.argsort(predictions)[::-1]

    recommended_movies = []
    for idx in sorted_indices:
        movie_id = [k for k, v in movie_id_to_index.items() if v == idx][0]
        if movie_id not in user_ratings:
            movie_info = movies_df[movies_df['movieId'] == movie_id]
            if not movie_info.empty:
                confidence = min(1.0, max(0.0, (predictions[idx] + 1) / 2))
                recommended_movies.append((movie_id, movie_info.iloc[0], confidence))
        if len(recommended_movies) >= num_recommendations:
            break
    return recommended_movies

# Étape 4 : Fonction pour demander une note
def rate_movie(movie_id, movie_title):
    while True:
        rating = input(f"Notez '{movie_title}' (1 à 5) : ").strip()
        try:
            rating = float(rating)
            if 1 <= rating <= 5:
                print(f"Film '{movie_title}' noté {rating}")
                return rating
            else:
                print("La note doit être entre 1 et 5.")
        except ValueError:
            print("Veuillez entrer un nombre valide.")
    return None

# Étape 5 : Boucle principale
user_ratings = {}
print("Commencez par rechercher un film pour noter.")
while True:
    # Recherche initiale ou arrêt
    search_term = input("\nEntrez le titre d’un film (ou 'stop' pour terminer) : ").strip()
    if search_term.lower() == 'stop':
        if len(user_ratings) == 0:
            print("Vous devez noter au moins un film avant de terminer.")
            continue
        break
    
    matches = search_movie(search_term)
    if matches is None or len(matches) == 0:
        continue
    
    # Afficher les résultats de recherche avec numéros
    for i, (idx, row) in enumerate(matches.iterrows(), 1):
        print(f"{i}. {row['title']} (ID: {row['movieId']})")
    
    # Sélectionner un film parmi les résultats de recherche
    while True:
        choice = input("Entrez le numéro du film choisi (ou 'r' pour rechercher à nouveau) : ").strip()
        if choice.lower() == 'r':
            break
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(matches):
                selected_movie = matches.iloc[choice_idx]
                movie_id = selected_movie['movieId']
                if movie_id not in movie_id_to_index:
                    print("Ce film n’est pas dans le modèle entraîné.")
                    break
                
                # Noter le film recherché
                rating = rate_movie(movie_id, selected_movie['title'])
                if rating is not None:
                    user_ratings[movie_id] = rating
                
                # Boucle de recommandations
                while True:
                    recommendations = generate_recommendations(user_ratings, 5)
                    if not recommendations:
                        print("Pas assez de données pour des recommandations pour l'instant.")
                        break
                    
                    print("\nVoici 5 recommandations :")
                    for i, (rec_movie_id, rec_movie_info, confidence) in enumerate(recommendations, 1):
                        print(f"{i}. {rec_movie_info['title']} (Genres: {rec_movie_info['genres']}) - Confiance : {confidence:.2f}")
                    
                    # Choix parmi les recommandations
                    while True:
                        rec_choice = input("Sélectionnez une recommandation à noter (1-5) ou 'n' pour une nouvelle recherche : ").strip()
                        if rec_choice.lower() == 'n':
                            break
                        try:
                            rec_idx = int(rec_choice) - 1
                            if 0 <= rec_idx < len(recommendations):
                                rec_movie_id, rec_movie_info, _ = recommendations[rec_idx]
                                if rec_movie_id not in user_ratings:  # Vérifier si pas déjà noté
                                    rating = rate_movie(rec_movie_id, rec_movie_info['title'])
                                    if rating is not None:
                                        user_ratings[rec_movie_id] = rating
                                    break  # Retourner à la boucle de recommandations
                                else:
                                    print("Ce film a déjà été noté.")
                            else:
                                print("Numéro invalide.")
                        except ValueError:
                            print("Veuillez entrer un numéro valide ou 'n'.")
                    if rec_choice.lower() == 'n':
                        break  # Retour à la recherche
                break
            else:
                print("Numéro invalide.")
        except ValueError:
            print("Veuillez entrer un numéro valide.")
        if choice.lower() == 'r':
            break

# Étape 6 : Afficher les recommandations finales
print("\nVoici vos 10 recommandations finales :")
final_recommendations = generate_recommendations(user_ratings, 10)
for i, (movie_id, movie_info, confidence) in enumerate(final_recommendations, 1):
    print(f"{i}. {movie_info['title']} (Genres: {movie_info['genres']}) - Confiance : {confidence:.2f}")

print("\nRecommandations générées avec succès !")
