"""
Download all required ML models for offline use in Aegis Guard.
"""
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def download_models():
    # Embedding models
    embedding_models = [
        "all-MiniLM-L6-v2",
    ]
    
    # Classification models
    classification_models = [
        "protectai/deberta-v3-base-prompt-injection-v2",
    ]
    
    print("Pre-downloading models for offline mode...")
    
    for model in embedding_models:
        print(f"Downloading embedding model: {model}")
        SentenceTransformer(model)
        
    for model in classification_models:
        print(f"Downloading classification model: {model}")
        AutoTokenizer.from_pretrained(model)
        AutoModelForSequenceClassification.from_pretrained(model)
        
    print("All models downloaded successfully.")

if __name__ == "__main__":
    download_models()
