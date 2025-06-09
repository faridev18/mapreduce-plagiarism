from collections import defaultdict
from utils import preprocess_text, jaccard_similarity
import os
import logging

class PlagiarismDetector:
    def __init__(self, reference_doc, threshold=0.7):
        """
        Initialise le détecteur de plagiat avec un document de référence.
        
        Args:
            reference_doc (str): Contenu du document de référence
            threshold (float): Seuil de similarité (0-1)
        """
        try:
            self.reference_tokens = set(preprocess_text(reference_doc))
            self.threshold = max(0.0, min(1.0, threshold))  # Garantir que le seuil est entre 0 et 1
            self.logger = self._setup_logger()
        except Exception as e:
            raise ValueError(f"Erreur lors de l'initialisation: {str(e)}")

    def _setup_logger(self):
        """Configure le logger pour le suivi des opérations"""
        logger = logging.getLogger('PlagiarismDetector')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        return logger

    def mapper(self, document):
        """
        Calcule la similarité entre un document et la référence.
        
        Args:
            document (dict): Doit contenir 'id' et 'content'
            
        Returns:
            tuple: (document_id, similarité)
        """
        try:
            if not isinstance(document, dict) or 'content' not in document:
                raise ValueError("Document doit être un dict avec 'content'")
                
            doc_id = document.get('id', 'unknown')
            doc_tokens = set(preprocess_text(document['content']))
            similarity = jaccard_similarity(self.reference_tokens, doc_tokens)
            
            self.logger.info(f"MAP: Document {doc_id} - Similarité: {similarity:.2f}")
            return (doc_id, similarity)
            
        except Exception as e:
            self.logger.error(f"Erreur dans mapper: {str(e)}")
            return (document.get('id', 'error'), 0.0)

    def reducer(self, mapped_values):
        """
        Filtre les documents dont la similarité dépasse le seuil.
        
        Args:
            mapped_values (list): Liste de tuples (doc_id, similarité)
            
        Returns:
            list: Documents plagiés avec leurs similarités
        """
        try:
            plagiarized = []
            for doc_id, similarity in mapped_values:
                if similarity >= self.threshold:
                    self.logger.info(f"REDUCE: Document {doc_id} dépasse le seuil ({similarity:.2f} >= {self.threshold})")
                    plagiarized.append((doc_id, round(similarity, 4)))
            
            self.logger.info(f"REDUCE: {len(plagiarized)} documents plagiés trouvés")
            return sorted(plagiarized, key=lambda x: x[1], reverse=True)  # Tri par similarité décroissante
            
        except Exception as e:
            self.logger.error(f"Erreur dans reducer: {str(e)}")
            return []

    def run_local(self, documents):
        """
        Exécute MapReduce localement (scénario 1 - tous les documents sur un serveur).
        
        Args:
            documents (list): Liste de documents à analyser
            
        Returns:
            list: Documents plagiés
        """
        self.logger.info("Début du traitement LOCAL (scénario 1)")
        try:
            mapped_values = [self.mapper(doc) for doc in documents]
            return self.reducer(mapped_values)
        except Exception as e:
            self.logger.error(f"Erreur dans run_local: {str(e)}")
            return []

    def run_distributed(self, document_chunks):
        """
        Exécute MapReduce de manière distribuée (scénario 2 - documents répartis).
        
        Args:
            document_chunks (list): Liste de listes, chaque sous-liste représente un serveur
            
        Returns:
            list: Documents plagiés
        """
        self.logger.info(f"Début du traitement DISTRIBUÉ (scénario 2) sur {len(document_chunks)} serveurs")
        try:
            # Phase Map parallèle (simulée)
            all_mapped = []
            for server_id, chunk in enumerate(document_chunks, 1):
                self.logger.info(f"MAP sur serveur {server_id} - {len(chunk)} documents")
                server_mapped = [self.mapper(doc) for doc in chunk]
                all_mapped.extend(server_mapped)
            
            # Phase Reduce centralisée
            return self.reducer(all_mapped)
            
        except Exception as e:
            self.logger.error(f"Erreur dans run_distributed: {str(e)}")
            return []

    def analyze_results(self, results):
        """
        Analyse et formate les résultats pour affichage.
        
        Args:
            results (list): Résultats de reducer
            
        Returns:
            dict: Statistiques et données formatées
        """
        stats = {
            'total_docs': len(results),
            'threshold': self.threshold,
            'plagiarized': [],
            'max_similarity': 0.0,
            'min_similarity': 0.0,
            'average': 0.0
        }
        
        if results:
            similarities = [sim for _, sim in results]
            stats.update({
                'plagiarized': results,
                'max_similarity': max(similarities),
                'min_similarity': min(similarities),
                'average': sum(similarities) / len(similarities)
            })
            
        return stats