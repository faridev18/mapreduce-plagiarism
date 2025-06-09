import streamlit as st
import os
from mapreduce_plagiarism import PlagiarismDetector
from utils import read_file, jaccard_similarity
import tempfile
from collections import defaultdict

st.set_page_config(layout="wide")
st.title("Détection de plagiat distribué avec MapReduce")

# Configuration
st.sidebar.header("Configuration")
threshold = st.sidebar.slider("Seuil de similarité", 0.0, 1.0, 0.7, 0.05)

# Téléchargement du document de référence
st.header("1. Document de référence")
ref_file = st.file_uploader("Télécharger le document de référence", 
                          type=['txt', 'docx', 'pdf'], 
                          key="ref_file")

# Téléchargement des documents à comparer
st.header("2. Documents à vérifier")
uploaded_files = st.file_uploader("Télécharger les documents à vérifier", 
                                type=['txt', 'docx', 'pdf'], 
                                accept_multiple_files=True,
                                key="doc_files")

# Configuration des serveurs
if uploaded_files:
    st.header("3. Configuration des serveurs")
    
    n_servers = st.number_input("Nombre de serveurs", 
                               min_value=1, 
                               max_value=len(uploaded_files), 
                               value=min(3, len(uploaded_files)))
    
    # Création des colonnes pour la répartition
    cols = st.columns(n_servers)
    server_files = defaultdict(list)
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Permettre à l'utilisateur de choisir le serveur pour chaque fichier
        selected_server = st.selectbox(
            f"Affecter '{uploaded_file.name}' à quel serveur?",
            options=list(range(1, n_servers+1)),
            index=i % n_servers,
            key=f"server_{i}"
        )
        server_files[selected_server-1].append(uploaded_file)
    
    # Afficher la répartition
    st.header("Répartition des fichiers")
    for server in range(n_servers):
        with st.expander(f"🗄️ Serveur {server+1} ({len(server_files[server])} fichiers)"):
            for file in server_files[server]:
                st.write(f"- {file.name}")

# Exécution
if ref_file and uploaded_files and n_servers:
    if st.button("Lancer la détection de plagiat distribué"):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Sauvegarder le fichier de référence
            ref_path = os.path.join(temp_dir, ref_file.name)
            with open(ref_path, "wb") as f:
                f.write(ref_file.getbuffer())
            
            # Lire le contenu du document de référence
            try:
                ref_content = read_file(ref_path)
                st.success("✅ Document de référence chargé avec succès")
                
                # Initialiser le détecteur de plagiat
                detector = PlagiarismDetector(ref_content, threshold)
                
                # Préparer les documents par serveur
                server_docs = []
                for server in range(n_servers):
                    server_documents = []
                    for uploaded_file in server_files[server]:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        content = read_file(file_path)
                        server_documents.append({
                            'id': uploaded_file.name,
                            'content': content,
                            'server': server+1
                        })
                    server_docs.append(server_documents)
                
                # Visualisation du processus MapReduce
                st.header("🔎 Processus de détection")
                
                # Phase Map sur chaque serveur
                st.subheader("Phase Map (sur chaque serveur)")
                map_results = []
                
                map_cols = st.columns(n_servers)
                for server in range(n_servers):
                    with map_cols[server]:
                        st.markdown(f"**Serveur {server+1}**")
                        for doc in server_docs[server]:
                            similarity = detector.mapper(doc)[1]
                            st.write(f"- {doc['id']}: {similarity:.1%}")
                            map_results.append((doc['id'], similarity, doc['server']))
                
                # Phase Reduce
                st.subheader("Phase Reduce (agrégation centrale)")
                plagiarized = detector.reducer([(x[0], x[1]) for x in map_results])
                
                # Résultats finaux
                st.header("📊 Résultats")
                if plagiarized:
                    st.subheader("Documents potentiellement plagiés:")
                    for doc_id, similarity in plagiarized:
                        server = next(x[2] for x in map_results if x[0] == doc_id)
                        st.write(f"- **{doc_id}** (similarité: {similarity:.2%}, Serveur: {server})")
                else:
                    st.info("Aucun document plagié détecté avec ce seuil.")
                
                # Statistiques
                st.subheader("📈 Statistiques de répartition")
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    st.metric("Total documents", len(uploaded_files))
                with stats_cols[1]:
                    st.metric("Serveurs utilisés", n_servers)
                with stats_cols[2]:
                    st.metric("Documents plagiés", len(plagiarized))
                
            except Exception as e:
                st.error(f"Erreur: {str(e)}")