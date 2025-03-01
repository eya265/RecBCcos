{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "d2cd0d31-8ecb-4d63-9840-d432ed5dfc36",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Modèle entraîné et sauvegardé !\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\User\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import json\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# Données des films\n",
    "data = {\n",
    "    'title': ['Film A', 'Film B', 'Film C', 'Film D'],\n",
    "    'description': [\n",
    "        'Un film d’action avec des courses-poursuites.',\n",
    "        'Une comédie romantique drôle.',\n",
    "        'Un thriller psychologique plein de suspense.',\n",
    "        'Un film d’aventure avec des voyages.'\n",
    "    ]\n",
    "}\n",
    "df = pd.DataFrame(data)\n",
    "\n",
    "# Importation des stop words en français\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "\n",
    "# Télécharger les stop words français\n",
    "nltk.download('stopwords')\n",
    "french_stop_words = stopwords.words('french')\n",
    "\n",
    "# Appliquer les stop words français\n",
    "tfidf = TfidfVectorizer(stop_words=french_stop_words)\n",
    "\n",
    "tfidf_matrix = tfidf.fit_transform(df['description'])\n",
    "\n",
    "# Calculer la similarité cosinus\n",
    "cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)\n",
    "\n",
    "# Sauvegarde en JSON\n",
    "df.to_json(\"movies.json\", orient=\"records\")\n",
    "cosine_sim_dict = {str(i): list(cosine_sim[i]) for i in range(len(df))}\n",
    "with open(\"cosine_sim.json\", \"w\") as f:\n",
    "    json.dump(cosine_sim_dict, f)\n",
    "\n",
    "print(\"Modèle entraîné et sauvegardé !\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f48d56d-4199-405e-b46b-5dbc2074b9d4",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
