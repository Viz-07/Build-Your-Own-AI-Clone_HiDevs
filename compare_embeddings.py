from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

def main():
    # Initialize local embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Define words to compare
    word1 = "apple"
    word2 = "iphone"

    # Get embeddings
    vec1 = embedding_function.embed_query(word1)
    vec2 = embedding_function.embed_query(word2)

    print(f"Vector length: {len(vec1)}")

    # Calculate cosine similarity
    sim_score = cosine_similarity(
        [vec1],
        [vec2]
    )[0][0]

    print(f"Cosine similarity between '{word1}' and '{word2}': {sim_score:.4f}")

if __name__ == "__main__":
    main()
