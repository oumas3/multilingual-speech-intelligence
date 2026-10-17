from pathlib import Path
import re

from pypdf import PdfReader
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
    util,
)

from llm import answer_with_context


# =================================================
# MODELS
# =================================================

print("Loading RAG models...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("RAG models loaded!")


# =================================================
# PDF TEXT EXTRACTION
# =================================================

def extract_pdf_text(pdf_path):
    reader = PdfReader(str(pdf_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    document_text = " ".join(pages)

    document_text = re.sub(
        r"\s+",
        " ",
        document_text
    ).strip()

    if not document_text:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    return document_text


# =================================================
# CHUNKING
# =================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=100
):
    chunks = []

    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text)
        )

        # Try to finish at a sentence
        if end < len(text):
            sentence_end = text.rfind(
                ".",
                start,
                end
            )

            if sentence_end > start:
                end = sentence_end + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(
            end - overlap,
            start + 1
        )

    return chunks


# =================================================
# KNOWLEDGE BASE
# =================================================

def build_knowledge_base(pdf_path):
    text = extract_pdf_text(pdf_path)

    chunks = create_chunks(text)

    if not chunks:
        raise ValueError(
            "Could not create chunks from the PDF."
        )

    embeddings = embedding_model.encode(
        chunks,
        convert_to_tensor=True,
        show_progress_bar=False
    )

    return {
        "chunks": chunks,
        "embeddings": embeddings,
        "source": Path(pdf_path).name
    }


# =================================================
# RETRIEVAL
# =================================================

def retrieve_chunks(
    question,
    knowledge_base,
    top_k=5
):
    chunks = knowledge_base["chunks"]
    embeddings = knowledge_base["embeddings"]

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        question_embedding,
        embeddings
    )[0]

    k = min(
        top_k,
        len(chunks)
    )

    results = scores.topk(k=k)

    retrieved = []

    for score, index in zip(
        results.values,
        results.indices
    ):
        i = index.item()

        retrieved.append(
            {
                "text": chunks[i],
                "score": score.item()
            }
        )

    return retrieved


# =================================================
# RERANKING
# =================================================

def rerank_chunks(
    question,
    retrieved
):
    if not retrieved:
        return []

    pairs = [
        [question, item["text"]]
        for item in retrieved
    ]

    scores = reranker.predict(pairs)

    for item, score in zip(
        retrieved,
        scores
    ):
        item["rerank_score"] = float(score)

    return sorted(
        retrieved,
        key=lambda x: x["rerank_score"],
        reverse=True
    )


# =================================================
# RAG QUESTION ANSWERING
# =================================================

def rag_answer(
    question,
    knowledge_base
):
    if not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    # Step 1: Semantic retrieval
    retrieved = retrieve_chunks(
        question,
        knowledge_base
    )

    # Step 2: Relevance check based on
    # the strongest cosine similarity
    top_similarity = max(
        item["score"]
        for item in retrieved
    ) if retrieved else 0.0

    if (
        not retrieved
        or top_similarity < 0.25
    ):
        return {
            "question": question,
            "answer": (
                "I cannot answer this question "
                "based on the provided document."
            ),
            "context": "",
            "source": knowledge_base["source"],
            "results": retrieved[:2]
        }

    # Step 3: CrossEncoder reranking
    reranked = rerank_chunks(
        question,
        retrieved
    )

    # Step 4: Keep strongest evidence
    best_results = reranked[:2]

    # Step 5: Build grounded context
    context = "\n\n".join(
        item["text"]
        for item in best_results
    )

    # Step 6: Generate grounded answer
    answer = answer_with_context(
        question,
        context
    )

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "source": knowledge_base["source"],
        "results": best_results
    }

    # Step 5: Build grounded context
    context = "\n\n".join(
        item["text"]
        for item in best_results
    )

    # Step 6: Generate answer using Qwen
    answer = answer_with_context(
        question,
        context
    )

    return {
        "question": question,
        "answer": answer,
        "context": context,
        "source": knowledge_base["source"],
        "results": best_results
    }