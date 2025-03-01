# Utilisation de Python 3.9
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Démarrer l’API FastAPI
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
