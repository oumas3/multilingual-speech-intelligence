# Multilingual Speech Intelligence

An end-to-end AI application for **speech understanding, document-grounded question answering, and voice-based document interaction**.

The project combines speech foundation models, semantic retrieval, reranking, and a lightweight language model in an interactive Streamlit application.

## Core Features

- Audio transcription with **Whisper**
- Speech representation extraction with **WavLM**
- Transcript analysis with **Qwen2.5**
- Runtime PDF ingestion and chunking
- Semantic retrieval with **Sentence Transformers**
- Second-stage retrieval with **CrossEncoder reranking**
- Document-grounded question answering
- Rejection of low-relevance questions
- Voice-based interaction with uploaded PDFs
- Retrieval evaluation with in-domain and out-of-domain queries

---

## System Overview

The application contains three main workflows:

1. **Speech Intelligence**  
   Analyze an audio file using Whisper, WavLM, and Qwen.

2. **Document Intelligence**  
   Upload a PDF and ask questions grounded in its content.

3. **Voice + Document QA**  
   Ask questions about a PDF using speech instead of typed input.

---

# 1. Speech Intelligence

The speech pipeline processes the same audio input through two independent paths.

```text
                         ┌──────────────────────┐
                         │     Audio Input      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌─────────────────┐             ┌─────────────────┐
          │   Whisper ASR   │             │  WavLM Encoder  │
          └────────┬────────┘             └────────┬────────┘
                   │                               │
                   ▼                               ▼
             Transcription                  Speech Embedding
                   │                         [768 dimensions]
                   ▼
          ┌─────────────────┐
          │ Qwen2.5-0.5B    │
          └────────┬────────┘
                   │
                   ▼
          Transcript Analysis
```

**Whisper** handles speech-to-text transcription.

**WavLM** independently extracts a 768-dimensional representation of the acoustic signal.

The Whisper transcript is passed to **Qwen2.5** for language-level analysis.

The WavLM representation is currently kept as a separate speech feature and is not passed to Qwen or the RAG pipeline.

---

# 2. Document Intelligence

The document pipeline performs retrieval before sending evidence to the language model.

```text
                    ┌──────────────────────┐
                    │     Uploaded PDF     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Text Extraction    │
                    │       PyPDF          │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Chunking        │
                    │  500 chars / 100 ov. │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ MiniLM Embeddings    │
                    │   384 dimensions     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Cosine Retrieval     │
                    │       Top-K = 5      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ CrossEncoder         │
                    │ Reranking            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Relevance Check      │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
               Relevant              Low relevance
                    │                     │
                    ▼                     ▼
          ┌──────────────────┐      Refuse Question
          │ Qwen2.5 + Context│
          └────────┬─────────┘
                   │
                   ▼
          Grounded Answer
          + Retrieved Evidence
```

The language model does not receive the entire PDF. It receives selected evidence retrieved from the document.

---

## Retrieval Pipeline

### PDF Extraction

Text is extracted from the uploaded document using `pypdf`.

The current implementation is designed for text-based PDFs.

### Chunking

Extracted text is divided into overlapping chunks.

```text
Chunk size : 500 characters
Overlap    : 100 characters
```

When possible, chunk boundaries are adjusted to finish at sentence boundaries.

### Embeddings

Document chunks are encoded using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each chunk is represented by a **384-dimensional embedding**.

The user's question is encoded using the same model.

### Initial Retrieval

Cosine similarity is calculated between the question embedding and all document chunk embeddings.

The system retrieves:

```text
Top-K = 5
```

candidate passages.

### Reranking

The retrieved candidates are passed to:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The CrossEncoder evaluates each:

```text
(question, document chunk)
```

pair directly and reranks the initial retrieval results.

The highest-ranked passages are then used as evidence for generation.

---

# 3. Voice + Document QA

The voice interface connects speech recognition directly to the document retrieval pipeline.

```text
                    ┌──────────────────────┐
                    │   Spoken Question    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Whisper ASR      │
                    └──────────┬───────────┘
                               │
                               ▼
                    Transcribed Question
                               │
                               ▼
              ┌──────────────────────────────┐
              │      Document Retrieval      │
              │                              │
              │ MiniLM → Top-K → CrossEncoder│
              └──────────────┬───────────────┘
                             │
                             ▼
                    Retrieved Evidence
                             │
                             ▼
                    ┌──────────────────────┐
                    │      Qwen2.5         │
                    │ Question + Evidence  │
                    └──────────┬───────────┘
                               │
                               ▼
                      Grounded Answer
```

This workflow allows the user to interact with a document through speech while keeping the answer grounded in retrieved PDF content.

WavLM is not used in this workflow because document retrieval operates on the transcribed text query.

---

# Models

| Task | Model | Output / Role |
|---|---|---|
| Automatic Speech Recognition | `openai/whisper-tiny` | Transcript |
| Speech Representation | `microsoft/wavlm-base-plus` | 768-d embedding |
| Document Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-d embeddings |
| Retrieval Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Relevance score |
| Language Model | `Qwen/Qwen2.5-0.5B-Instruct` | Analysis and grounded generation |

---

# Relevance Filtering

Dense retrieval will always return the nearest document chunks, even when a question is unrelated to the document.

To reduce unsupported generation, a relevance check is applied before sending context to Qwen.

Current experimental threshold:

```text
Cosine similarity >= 0.25
```

If the retrieved evidence does not meet the relevance requirement, the system returns:

> I cannot answer this question based on the provided document.

The threshold is an experimental heuristic evaluated on the project's current benchmark. It is not assumed to be universally optimal across documents or domains.

---

# Evaluation

A small retrieval benchmark was created to evaluate whether the pipeline retrieves useful evidence and distinguishes document-related questions from unrelated questions.

## Evaluation Setup

The test corpus contains a cloud-computing survey paper.

The benchmark contains:

```text
15 total queries

10 in-domain queries
 5 out-of-domain queries
```

The evaluation measures:

- Recall@5
- CrossEncoder Top-1 accuracy
- relevance classification accuracy
- relevant-query similarity
- irrelevant-query similarity

## Results

| Metric | Result |
|---|---:|
| Recall@5 | **100%** |
| CrossEncoder Top-1 Accuracy | **100%** |
| Relevance Classification Accuracy | **100%** |
| Average Relevant Similarity | **0.6772** |
| Average Irrelevant Similarity | **0.1262** |
| Lowest Relevant Similarity | **0.4863** |
| Highest Irrelevant Similarity | **0.2138** |
| Experimental Threshold | **0.25** |

The evaluated similarity scores showed the following separation:

```text
              OUT-OF-DOMAIN                  IN-DOMAIN

     avg = 0.1262                                avg = 0.6772

              0.2138       0.2500       0.4863
                 │            │             │
                 │            │             │
        highest irrelevant  threshold   lowest relevant
```

All 15 queries were classified correctly using the current threshold in this benchmark.

These numbers come from a **small, curated, single-document engineering benchmark** and should not be interpreted as general RAG accuracy.

Run the evaluation with:

```bash
python evaluation/evaluate_rag.py
```

---

# Project Structure

```text
multilingual-speech-intelligence/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── paper.pdf
│   └── test_audio.mp3
│
├── src/
│   ├── transcribe.py
│   ├── audio_embeddings.py
│   ├── llm.py
│   └── rag.py
│
└── evaluation/
    └── evaluate_rag.py
```

---

# Installation

## Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd multilingual-speech-intelligence
```

## Create a Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

Start the Streamlit interface with:

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

The `fileWatcherType` option avoids conflicts between Streamlit's source watcher and optional modules exposed through the Transformers ecosystem.

---

# Running the Evaluation

```bash
python evaluation/evaluate_rag.py
```

Example evaluation output:

```text
FINAL EVALUATION RESULTS

Questions evaluated: 15
Relevant questions: 10
Irrelevant questions: 5

Recall@5: 100.00%
Reranking Top-1 Accuracy: 100.00%
Threshold Classification Accuracy: 100.00%

Average relevant similarity: 0.6772
Average irrelevant similarity: 0.1262
Lowest relevant similarity: 0.4863
Highest irrelevant similarity: 0.2138
Current relevance threshold: 0.25
```

---

# Tech Stack

### Machine Learning

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers

### Speech

- Whisper
- WavLM
- Librosa

### Retrieval & Generation

- Dense semantic retrieval
- Cosine similarity
- CrossEncoder reranking
- Retrieval-Augmented Generation
- Qwen2.5

### Document Processing

- PyPDF

### Interface

- Streamlit

---

# Engineering Decisions

## Two-Stage Retrieval

The RAG pipeline separates retrieval into two stages:

```text
Bi-Encoder Retrieval
        │
        ▼
   Top-K Candidates
        │
        ▼
CrossEncoder Reranking
        │
        ▼
  Selected Evidence
```

The bi-encoder provides efficient semantic search while the CrossEncoder performs more precise query-passage comparison on the smaller candidate set.

## Grounded Generation

Qwen receives the user's question together with selected document evidence.

The generation prompt explicitly instructs the model to answer using the provided context rather than relying on outside knowledge.

## Out-of-Domain Handling

A relevance threshold is applied before generation to reduce answers to unsupported questions.

This addresses a common retrieval problem: nearest-neighbor search will return a result even when none of the available passages is actually relevant.

## Separate Speech Representations

Whisper and WavLM serve different roles.

```text
Whisper → linguistic content
WavLM   → acoustic representation
```

They currently operate as parallel components.

The project therefore does not claim that these representations are fused into a single multimodal model.

---

# Current Limitations

- The relevance threshold was evaluated on a small single-document benchmark.
- PDF processing currently targets text-based documents rather than scanned PDFs requiring OCR.
- Embeddings are stored in memory rather than in a persistent vector database.
- Local model inference can be computationally expensive on CPU-only machines.
- The current benchmark is an engineering sanity test rather than a large-scale academic evaluation.
- Multilingual and code-switched speech performance still requires systematic evaluation.
- WavLM representations are extracted but are not currently used for downstream classification or retrieval.

---

# Future Work

Planned directions include:

- multilingual speech evaluation
- code-switched speech evaluation
- larger document collections
- multi-document retrieval
- improved relevance calibration
- automatic evaluation dataset generation
- persistent vector indexing for larger corpora
- inference and model-loading optimization
- richer visualization of retrieved evidence
- deployment-oriented model serving

---

# Author

**Oumaima Ouayres**

Data Science & AI  
Machine Learning · NLP · Speech AI · Retrieval-Augmented Generation