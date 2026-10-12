
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


chunks = [
    "Whisper is a speech recognition model developed by OpenAI.",
    "WavLM is a speech representation model for audio features.",
    "RAG retrieves relevant information before the LLM answers.",
    "Vector databases store embeddings for semantic search.",
]

print("Loading text embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nCreating embeddings...")
embeddings = model.encode(chunks, convert_to_tensor=True)

print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)


questions = [
    "Which model is used for speech recognition?",
    "Which model creates audio representations?",
    "What does RAG do?",
    "What stores embeddings for semantic search?",
]

for question in questions:
    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
    )

    scores = util.cos_sim(question_embedding, embeddings)[0]
    top_indices = scores.topk(k=min(3, len(chunks))).indices.tolist()

    candidates = [chunks[i] for i in top_indices]
    pairs = [[question, chunk] for chunk in candidates]
    rerank_scores = reranker.predict(pairs)

    best_index = int(rerank_scores.argmax())

    print("\nQuestion:", question)
    print("Best match:", candidates[best_index])
