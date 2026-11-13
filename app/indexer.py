from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS = Path("corpus")
DB_PATH = "chroma_db"

# Modèle d'embeddings qui comprend le français (téléchargé la première fois)
modele = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("grammaire")

ids, textes, metadonnees = [], [], []
for fichier in sorted(CORPUS.glob("*.md")):
    contenu = fichier.read_text(encoding="utf-8")
    ids.append(fichier.stem)
    textes.append(contenu)
    metadonnees.append({"source": fichier.name})

embeddings = modele.encode(textes).tolist()
collection.upsert(
    ids=ids, documents=textes, embeddings=embeddings, metadatas=metadonnees
)
print(f"{len(ids)} fiches indexées.")