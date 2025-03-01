from fastapi import FastAPI
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les fichiers JSON
with open("movies.json", "r") as f:
    df = pd.read_json(f)

with open("cosine_sim.json", "r") as f:
    cosine_sim = json.load(f)

# Créer l'API FastAPI
app = FastAPI()

# Transformer les titres en vecteurs TF-IDF pour la recherche de similarité
tfidf = TfidfVectorizer()
title_matrix = tfidf.fit_transform(df['title'])

def find_closest_title(user_title):
    """Trouve le titre de film le plus proche en utilisant TF-IDF + Similarité cosinus."""
    user_title_vector = tfidf.transform([user_title])
    similarities = cosine_similarity(user_title_vector, title_matrix).flatten()

    best_match_index = similarities.argmax()
    best_match_score = similarities[best_match_index]

    if best_match_score > 0.3:
        return df['title'].iloc[best_match_index]
    return None

def recommend_movies(title):
    """Retourne les films vraiment similaires avec un seuil de similarité."""
    if title not in df['title'].values:
        closest_title = find_closest_title(title)
        if closest_title:
            title = closest_title
        else:
            return {"error": "Aucun film similaire trouvé"}

    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[str(idx)]))

    # Trier par score de similarité décroissant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Exclure le film lui-même et garder ceux avec similarité > 0.2
    filtered_scores = [x for x in sim_scores if x[0] != idx and x[1] > 0.2]

    # Prendre les 3 meilleurs
    movie_indices = [i[0] for i in filtered_scores[:3]]

    return {
        "searched_title": title,
        "recommendations": df['title'].iloc[movie_indices].tolist()
    }

@app.get("/recommend/")
def get_recommendations(title: str):
    return recommend_movies(title)
