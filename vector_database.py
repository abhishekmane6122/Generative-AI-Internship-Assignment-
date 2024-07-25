import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_texts():
    with open("pdf_text.txt", "r") as f:
        pdf_text = f.read()
    
    with open("website_text.txt", "r") as f:
        website_text = f.read()

    with open("youtube_text.txt", "r") as f:
        youtube_text = f.read()

    return [pdf_text, website_text, youtube_text]

def setup_faiss(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, model, texts

def search(index, model, texts, query, k=1):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Example usage
if __name__ == "__main__":
    texts = load_texts()
    index, model, texts = setup_faiss(texts)
    results = search(index, model, texts, "Tell me about Apple Vision Pro")
    print(results)
