from sentence_transformers import SentenceTransformer, CrossEncoder, util
from llm import answer_with_context


chunks = [
    "Whisper is a speech recognition model developed by OpenAI.",
    "WavLM is a speech representation model for audio features.",
    "RAG retrieves relevant information before the LLM answers.",
    "Vector databases store embeddings for semantic search."
]



embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

chunk_embeddings = embedding_model.encode(chunks)

def retrieve_chunks(question, top_k=2):
    question_embedding = embedding_model.encode(question)

    scores = util.cos_sim(
        question_embedding,
        chunk_embeddings
    )[0]

    top_results = scores.topk(k=top_k)

    retrieved_chunks = []

    for score, index in zip(top_results.values, top_results.indices):
        retrieved_chunks.append(
            {
                "text": chunks[index],
                "score": score.item()
            }
        )

    return retrieved_chunks
def rerank_chunks(question, retrieved_chunks):
    pairs = []

    for result in retrieved_chunks:
        pairs.append(
            [question, result["text"]]
        )

    rerank_scores = reranker.predict(pairs)

    for result, score in zip(retrieved_chunks, rerank_scores):
        result["rerank_score"] = float(score)

    reranked = sorted(
        retrieved_chunks,
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return reranked
def rag_answer(question):
    retrieved = retrieve_chunks(question, top_k=2)

    reranked = rerank_chunks(
        question,
        retrieved
    )

    best_context = reranked[0]["text"]

    answer = answer_with_context(
        question,
        best_context
    )

    return best_context, answer


question = "Which model is used for speech recognition?"

context, answer = rag_answer(question)

print("\nFinal context:")
print(context)

print("\nFinal RAG answer:")
print(answer)