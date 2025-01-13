"""This file includes the modules used to construct the multi-modality item embedding learning model."""
import torch
import torch.nn.functional as F
from ml_prototype.mm_embedding.module import TextEncoder


def compute_cosine_similarity(embedding1, embedding2):
    """Compute the cosine similarity between two embeddings."""
    return F.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()


def test_text_encoder_similarity():
    """Test script for evaluating embedding effectiveness across languages."""
    # Initialize the multilingual TextEncoder
    print("Initializing TextEncoder with multilingual BERT...")
    text_encoder = TextEncoder(model_name='bert-base-multilingual-cased', output_dim=512)
    text_encoder.eval()

    # Define semantically equivalent sentences in different languages
    multilingual_sentences = {
        "English": "This is a test sentence.",
        "Chinese": "这是一个测试句子。",
        "French": "C'est une phrase de test.",
        "Spanish": "Esta es una oración de prueba.",
        "Russian": "Это тестовое предложение.",
    }

    # Define irrelevant words in the same language
    irrelevant_words = {
        "English": ["apple", "banana", "cherry"],
        "Chinese": ["苹果", "香蕉", "樱桃"],
        "French": ["pomme", "banane", "cerise"],
        "Spanish": ["manzana", "plátano", "cereza"],
        "Russian": ["яблоко", "банан", "вишня"],
    }

    # Compute embeddings for each sentence
    print("\nGenerating embeddings for multilingual sentences...")
    embeddings = {}
    with torch.no_grad():
        for lang, sentence in multilingual_sentences.items():
            embeddings[lang] = text_encoder([sentence]).squeeze(0)  # Remove batch dimension
            print(f"{lang} Embedding Shape: {embeddings[lang].shape}")

    # Compute embeddings for irrelevant words
    print("\nGenerating embeddings for irrelevant words...")
    irrelevant_embeddings = {}
    with torch.no_grad():
        for lang, words in irrelevant_words.items():
            irrelevant_embeddings[lang] = [text_encoder([word]).squeeze(0) for word in words]
            for i, word in enumerate(words):
                print(f"{lang} '{word}' Embedding Shape: {irrelevant_embeddings[lang][i].shape}")

    # Compare embeddings across languages
    print("\nEvaluating similarity across languages:")
    languages = list(multilingual_sentences.keys())
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            lang1, lang2 = languages[i], languages[j]
            similarity = compute_cosine_similarity(embeddings[lang1], embeddings[lang2])
            print(f"Cosine Similarity between {lang1} and {lang2}: {similarity:.4f}")

    # Compare embeddings within the same language
    print("\nEvaluating dissimilarity within the same language:")
    for lang in languages:
        for i in range(len(irrelevant_words[lang])):
            similarity = compute_cosine_similarity(embeddings[lang], irrelevant_embeddings[lang][i])
            print(f"Cosine Similarity between '{multilingual_sentences[lang]}' and '{irrelevant_words[lang][i]}': {similarity:.4f}")


if __name__ == "__main__":
    test_text_encoder_similarity()
