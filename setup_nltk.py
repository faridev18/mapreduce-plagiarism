import nltk
import os

def setup_nltk():
    # Créer le répertoire pour les données tabulaires si nécessaire
    nltk_dir = os.path.join(nltk.data.path[0], 'tab')
    os.makedirs(nltk_dir, exist_ok=True)
    
    # Télécharger toutes les ressources nécessaires
    nltk.download('punkt', download_dir=os.path.join(nltk.data.path[0], 'tab'))
    nltk.download('stopwords')
    nltk.download('punkt')  # Version standard

if __name__ == '__main__':
    setup_nltk()
    print("Configuration NLTK terminée avec succès!")