
# Utilise une image officielle Python
FROM python:3.11-slim

# Crée un dossier de travail dans le conteneur
WORKDIR /app

# Copie les fichiers nécessaires
COPY . /app

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Expose le port
EXPOSE 8080

# Commande de lancement
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
