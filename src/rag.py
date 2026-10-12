
from pathlib import Path
import re

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from llm import answer_with_context


# Find the project's data folder, regardless of the terminal's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Find PDF documents in the data folder.
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        f"No PDF files found in {DATA_DIR}. Add a research PDF to the data folder."
    )

# Extract text from every PDF.
chunks = []
chunk_sources = []

chunk_size = 500
overlap = 100

for pdf_path in pdf_files:
    reader = PdfReader(str(pdf_path))
    pages_text = []

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)

    # Clean the extracted text, not an empty string.
    document_text = re.sub(r"\s+", " ", " ".join(pages_text)).strip()

    if not document_text:
        print(f"Warning: no extractable text in {pdf_path.name}")
        continue

    # Split the document into overlapping chunks.
    start = 0

    while start < len(document_text):
        end = min(start + chunk_size, len(document_text))
        chunk = document_text[start:end].strip()

        if chunk:
            chunks.append(chunk)
            chunk_sources.append(pdf_path.name)

        if end == len(document_text):
            break

        start = end - overlap

if not chunks:
    raise ValueError("No text chunks could be extracted from the PDFs.")

print(f"Loaded {len(pdf_files)} PDF file(s).")
print(f"Created {len(chunks)} text chunks.")


# Load the models used for retrieval and reranking.
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Creating document embeddings...")
chunk_embeddings = embedding_model.encode(
    chunks,
    convert_to_tensor=True,
    show_progress_bar=True
)


def retrieve_chunks(question, top_k=3):
    """Retrieve chunks with high semantic similarity to the question."""
    if not question or not question.strip():
        raise ValueError("The question cannot be empty.")

    question_embedding = embedding_model.encode(
        question,
        convert_to_tensor=True
    )

    scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    number_to_retrieve = min(top_k, len(chunks))
    top_results = scores.topk(k=number_to_retrieve)

    results = []

    for score, index in zip(top_results.values, top_results.indices):
        i = index.item()
        results.append({
            "text": chunks[i],
            "source": chunk_sources[i],
            "score": score.item()
        })

    return results


def rerank_chunks(question, retrieved_chunks):
    """Reorder retrieved chunks by relevance to the question."""
    if not retrieved_chunks:
        return []

    pairs = [
        [question, result["text"]]
        for result in retrieved_chunks
    ]

    rerank_scores = reranker.predict(pairs)

    for result, score in zip(retrieved_chunks, rerank_scores):
        result["rerank_score"] = float(score)

    return sorted(
        retrieved_chunks,
        key=lambda item: item["rerank_score"],
        reverse=True
    )


def rag_answer(question):
    """Retrieve evidence from PDFs and generate a grounded answer."""
    retrieved = retrieve_chunks(question, top_k=3)
    reranked = rerank_chunks(question, retrieved)

    # Use the two highest-ranked chunks as answer context.
    best_results = reranked[:2]
    context = "\n\n".join(item["text"] for item in best_results)

    answer = answer_with_context(question, context)

    sources = list(dict.fromkeys(
        item["source"] for item in best_results
    ))

    return {
        "question": question,
        "context": context,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":
    question = "What is cloud computing?"
    result = rag_answer(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nRetrieved context:")
    print(result["context"])

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")
