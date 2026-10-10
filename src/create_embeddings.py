from sentence_transformers import SentenceTransformer

chunks = [
    "Whisper is a speech recognition model developed by OpenAI.",
    "WavLM is a speech representation model for audio features.",
    "RAG retrieves relevant information before the LLM answers.",
    "Vector databases store embeddings for semantic search."
]

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst embedding:")
print(embeddings[0][:10])
from sentence_transformers import util

question = "Which model is used for speech recognition?"

question_embedding = model.encode(question)

scores = util.cos_sim(question_embedding, embeddings)[0]

best_index = scores.argmax().item()

print("\nQuestion:")
print(question)

print("\nSimilarity scores:")
print(scores)

print("\nBest matching chunk:")
print(chunks[best_index])