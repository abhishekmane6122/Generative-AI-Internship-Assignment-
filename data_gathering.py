import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

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

def extract_text_from_pdf(pdf_path):
    """Extract text from a local PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_url(url):
    """Extract text from a PDF file located at a URL."""
    response = requests.get(url)
    pdf_bytes = response.content
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def scrape_website(url):
    """Scrape text content from a website."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def get_youtube_transcript(video_id):
    """Retrieve the transcript of a YouTube video by its ID."""
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([item['text'] for item in transcript])

# Example usage
if __name__ == "__main__":
    texts = load_texts()
    index, model, texts = setup_faiss(texts)
    results = search(index, model, texts, "Tell me about Apple Vision Pro")
    print(results)