import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords français (si non téléchargés)
nltk.download('stopwords', quiet=True)
french_stop_words = stopwords.words('french')

# Données des films
data = {
    'title': ['Film A', 'Film B', 'Film C', 'Film D'],
    'description': [
        'Un film d’action avec des courses-poursuites.',
        'Une comédie romantique drôle.',
        'Un thriller psychologique plein de suspense.',
        'Un film d’aventure avec des voyages.'
    ]
}
df = pd.DataFrame(data)

# Transformer en TF-IDF
tfidf = TfidfVectorizer(stop_words=french_stop_words)
tfidf_matrix = tfidf.fit_transform(df['description'])

# Calculer la similarité cosinus
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Sauvegarde des données et de la matrice
df.to_json("movies.json", orient="records")

cosine_sim_dict = {
    str(i): [round(float(v), 4) for v in cosine_sim[i]]  # Arrondi pour alléger
    for i in range(len(df))
}

with open("cosine_sim.json", "w") as f:
    json.dump(cosine_sim_dict, f, indent=4)

print("Modèle entraîné et sauvegardé !")
