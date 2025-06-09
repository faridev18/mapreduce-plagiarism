import os
import re
from docx import Document
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

def setup_nltk():
    """Configure toutes les ressources NLTK nécessaires de manière robuste"""
    try:
        # Télécharger les ressources standard
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Vérifier et configurer punkt_tab si nécessaire
        punkt_tab_path = os.path.join(nltk.data.path[0], 'tokenizers', 'punkt_tab')
        if not os.path.exists(punkt_tab_path):
            os.makedirs(punkt_tab_path, exist_ok=True)
            # Copier les données de punkt vers punkt_tab
            import shutil
            src_path = os.path.join(nltk.data.path[0], 'tokenizers', 'punkt')
            if os.path.exists(src_path):
                shutil.copytree(src_path, os.path.join(punkt_tab_path, 'english'))
    except Exception as e:
        print(f"Warning: NLTK setup encountered an error: {e}")

# Configurer NLTK au chargement du module
setup_nltk()

def preprocess_text(text):
    """Prétraitement robuste du texte avec gestion des erreurs"""
    if not text or not isinstance(text, str):
        return []
    
    try:
        # Convertir en minuscules
        text = text.lower()
        # Supprimer la ponctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenisation avec fallback
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()  # Fallback simple si word_tokenize échoue
        
        # Supprimer les stopwords avec fallback
        try:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        except:
            pass  # Continuer sans filtre stopwords si échec
        
        # Stemming avec gestion d'erreur
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens if word]
        
        return tokens
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return []

def read_file(file_path):
    """Lecture robuste de fichiers avec gestion des erreurs"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif file_path.endswith('.docx'):
            doc = Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs if para.text])
        elif file_path.endswith('.pdf'):
            text = []
            reader = PdfReader(file_path)
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                except:
                    continue
            return '\n'.join(text)
        else:
            raise ValueError(f"Unsupported file format: {os.path.splitext(file_path)[1]}")
    except Exception as e:
        raise IOError(f"Failed to read file {file_path}: {str(e)}")

def jaccard_similarity(set1, set2):
    """Calcul de similarité Jaccard robuste"""
    if not isinstance(set1, set) or not isinstance(set2, set):
        set1 = set(set1) if hasattr(set1, '__iter__') else set()
        set2 = set(set2) if hasattr(set2, '__iter__') else set()
    
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0.0

def split_documents_for_distributed(docs, n_servers):
    """Répartition des documents avec validation des entrées"""
    if not docs or n_servers < 1:
        return [[]]
    
    n_servers = max(1, min(n_servers, len(docs)))
    chunks = [[] for _ in range(n_servers)]
    for i, doc in enumerate(docs):
        chunks[i % n_servers].append(doc)
    return chunks