import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.express as px
import numpy as np
import pandas as pd
from chromadb.api.models import Collection

from chromadb.api.types import IncludeEnum

def fetch_embeddings(collection: Collection):
    """Extrai os embeddings e metadados da coleção ChromaDB."""
    results = collection.get(include=[IncludeEnum.embeddings, IncludeEnum.metadatas])
    embeddings = results.get("embeddings", [])
    metadatas = results.get("metadatas", [])
    
    labels = [meta.get("document_type", "Unknown") for meta in metadatas]  # Extrair o rótulo
    return np.array(embeddings), labels

def reduce_dimensionality(embeddings, method='tsne', dimensions=3):
    """Reduz a dimensionalidade dos embeddings usando t-SNE ou PCA."""
    if method == 'tsne':
        reducer = TSNE(n_components=dimensions, random_state=42)
    elif method == 'pca':
        reducer = PCA(n_components=dimensions)
    else:
        raise ValueError("Método inválido! Use 'tsne' ou 'pca'.")
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    return embeddings_reduced
    
def plot_embeddings_3d(embeddings_3d, labels, method='t-SNE'):
    """Plota os embeddings em 3D de forma interativa com plotly."""
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'label': labels
    })
    
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label', title=f"Visualização dos Embeddings - {method}")
    fig.update_layout(scene=dict(
        xaxis_title="Componente 1",
        yaxis_title="Componente 2",
        zaxis_title="Componente 3"
    ))
    fig.show()

import chromadb


# Inicialize o cliente com as novas configurações
client = chromadb.PersistentClient(path='data/chroma')

# Lista todas as coleções
collections = client.list_collections()

# Obtenha a coleção desejada
collection = client.get_collection("langchain")

print(collection)

# Extrair embeddings e rótulos
embeddings, labels = fetch_embeddings(collection)

# Reduzir dimensionalidade para 3D
embeddings_3d = reduce_dimensionality(embeddings, method='tsne', dimensions=3)

# Plotar os embeddings em 3D
plot_embeddings_3d(embeddings_3d, labels, method='t-SNE')
