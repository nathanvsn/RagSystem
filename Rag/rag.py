import os
from langchain_huggingface import HuggingFaceEmbeddings
import fitz
from decouple import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from collections import Counter

os.environ['HUGGINGFACE_API_KEY'] = config('HUGGINGFACE_API_KEY')

# Função para gerar grupos de palavras
def generate_word_groups(text, group_size=5):
    words = text.split()
    return [' '.join(words[i:i + group_size]) for i in range(len(words) - group_size + 1)]

# Função para remover textos repetitivos
def remove_repetitive_text(pages, group_size=5, threshold=0.7):
    word_group_count = Counter()
    total_pages = len(pages)

    # Contar a frequência de grupos de palavras
    for doc in pages:
        word_groups = generate_word_groups(doc.page_content, group_size)
        word_group_count.update(word_groups)

    # Identificar quais grupos são repetitivos
    repetitive_groups = {group for group, count in word_group_count.items() if count / total_pages > threshold}

    # Remover os grupos repetitivos de cada página
    cleaned_pages = []
    for doc in pages:
        cleaned_content = doc.page_content
        for group in repetitive_groups:
            cleaned_content = cleaned_content.replace(group, '')
        
        # Adicionando metadados
        metadata = {
            'source': doc.metadata.get('source', 'unknown'),
            'page': doc.metadata.get('page', 'unknown'),
            'document_type': 'RegulamentoFundo',
            'regulator': 'Anbima',
            'document_number': 'Ghia-GhiaExtremosul250423',
            'CodAnbima': 'F0000708720',
            'subject': 'Constituição, funcionamento e divulgação de informações dos fundos de investimento',
            'publication_date': '2023-04-25',
            'keywords': ', '.join(['fundos de investimento', 'Ghia', 'Fundo de Investimento', 'Renda Fixa']),  # Converte lista para string
            'file_origin': pdf_path
        }
        
        cleaned_pages.append(Document(metadata=metadata, page_content=cleaned_content.strip()))

    return cleaned_pages

# Exemplo de uso principal
if __name__ == '__main__':
    pdf_path = 'RegulamentoGhiaSul.pdf'  # Substitua pelo caminho do seu PDF
    
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    
    # Fazer o embedding das páginas
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
    )
    
    # Aplicar a remoção de texto repetitivo e adicionar metadados
    cleaned_docs = remove_repetitive_text(docs)
    chunks = text_splitter.split_documents(documents=cleaned_docs)
        
    directory = 'data/chroma'

    embeddings = HuggingFaceEmbeddings(model_name=config('HUGGINGFACE_MODEL'))
    
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=directory,
    )
    
    vector_store.add_documents(documents=chunks, overwrite=True)
