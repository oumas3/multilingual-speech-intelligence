import sys
from pathlib import Path

# =================================================
# PROJECT PATHS
# =================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag import (
    build_knowledge_base,
    retrieve_chunks,
    rerank_chunks,
)


# =================================================
# CONFIGURATION
# =================================================

PDF_PATH = PROJECT_ROOT / "data" / "paper.pdf"

TOP_K = 5
RELEVANCE_THRESHOLD = 0.25


# =================================================
# EVALUATION DATASET
# =================================================

test_questions = [

    # -------------------------------------------------
    # Relevant questions
    # -------------------------------------------------

    {
        "question": "What is cloud computing?",
        "expected_keywords": [
            "cloud computing",
            "computing",
            "resources",
        ],
        "relevant": True,
    },

    {
        "question": "What are the four architectural layers of cloud computing?",
        "expected_keywords": [
            "physical",
            "infrastructure",
            "platform",
            "application",
        ],
        "relevant": True,
    },

    {
        "question": "What is the infrastructure layer responsible for?",
        "expected_keywords": [
            "virtualization",
            "storage",
            "computing resources",
        ],
        "relevant": True,
    },

    {
        "question": "What are the three service models of cloud computing?",
        "expected_keywords": [
            "iaas",
            "paas",
            "saas",
        ],
        "relevant": True,
    },

    {
        "question": "What does Infrastructure as a Service provide?",
        "expected_keywords": [
            "processing",
            "storage",
            "networks",
            "virtualization",
        ],
        "relevant": True,
    },

    {
        "question": "What does Platform as a Service provide?",
        "expected_keywords": [
            "development platform",
            "software lifecycle",
            "applications",
        ],
        "relevant": True,
    },

    {
        "question": "What is Software as a Service?",
        "expected_keywords": [
            "applications",
            "internet",
            "browser",
        ],
        "relevant": True,
    },

    {
        "question": "What is a private cloud?",
        "expected_keywords": [
            "private cloud",
            "organization",
            "security",
        ],
        "relevant": True,
    },

    {
        "question": "What is a hybrid cloud?",
        "expected_keywords": [
            "combination",
            "private",
            "community",
            "public",
        ],
        "relevant": True,
    },

    {
        "question": "What security challenges are associated with cloud computing?",
        "expected_keywords": [
            "security",
            "information loss",
            "phishing",
            "threat",
        ],
        "relevant": True,
    },

    # -------------------------------------------------
    # Irrelevant questions
    # -------------------------------------------------

    {
        "question": "Who won the 2018 FIFA World Cup?",
        "expected_keywords": [],
        "relevant": False,
    },

    {
        "question": "What is the capital of Japan?",
        "expected_keywords": [],
        "relevant": False,
    },

    {
        "question": "How does photosynthesis work?",
        "expected_keywords": [],
        "relevant": False,
    },

    {
        "question": "Who wrote Romeo and Juliet?",
        "expected_keywords": [],
        "relevant": False,
    },

    {
        "question": "What is the largest planet in the solar system?",
        "expected_keywords": [],
        "relevant": False,
    },
]


# =================================================
# HELPERS
# =================================================

def contains_expected_keyword(text, keywords):
    text = text.lower()

    return any(
        keyword.lower() in text
        for keyword in keywords
    )


def classify_with_threshold(score):
    return score >= RELEVANCE_THRESHOLD


# =================================================
# EVALUATION
# =================================================

def evaluate():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("\nBuilding knowledge base...")

    knowledge_base = build_knowledge_base(
        PDF_PATH
    )

    print(
        f"Document chunks: "
        f"{len(knowledge_base['chunks'])}"
    )

    print("\n" + "=" * 75)
    print("RAG RETRIEVAL + RERANKING EVALUATION")
    print("=" * 75)

    relevant_total = 0
    retrieval_hits = 0
    reranking_hits = 0

    threshold_correct = 0

    relevant_scores = []
    irrelevant_scores = []

    for number, test in enumerate(
        test_questions,
        start=1
    ):

        question = test["question"]
        keywords = test["expected_keywords"]
        relevant = test["relevant"]

        print("\n" + "=" * 75)

        print(
            f"Question {number}: "
            f"{question}"
        )

        print(
            "Expected:",
            "RELEVANT"
            if relevant
            else "IRRELEVANT"
        )

        # -----------------------------------------
        # Retrieval
        # -----------------------------------------

        retrieved = retrieve_chunks(
            question,
            knowledge_base,
            top_k=TOP_K
        )

        # -----------------------------------------
        # Reranking
        # -----------------------------------------

        reranked = rerank_chunks(
            question,
            retrieved
        )

        # -----------------------------------------
        # Similarity
        # -----------------------------------------

        if retrieved:

            top_similarity = max(
                item["score"]
                for item in retrieved
            )

        else:
            top_similarity = 0.0

        # -----------------------------------------
        # Reranker score
        # -----------------------------------------

        if reranked:

            top_reranker_score = (
                reranked[0][
                    "rerank_score"
                ]
            )

        else:
            top_reranker_score = 0.0

        # -----------------------------------------
        # Threshold prediction
        # -----------------------------------------

        predicted_relevant = (
            classify_with_threshold(
                top_similarity
            )
        )

        if predicted_relevant == relevant:
            threshold_correct += 1

        if relevant:
            relevant_scores.append(
                top_similarity
            )
        else:
            irrelevant_scores.append(
                top_similarity
            )

        # -----------------------------------------
        # Display scores
        # -----------------------------------------

        print(
            "Top similarity:",
            round(
                top_similarity,
                4
            )
        )

        print(
            "Top reranker score:",
            round(
                top_reranker_score,
                4
            )
        )

        print(
            "Threshold prediction:",
            "RELEVANT"
            if predicted_relevant
            else "IRRELEVANT"
        )

        # -----------------------------------------
        # Relevant-question evaluation
        # -----------------------------------------

        if relevant:

            relevant_total += 1

            retrieval_hit = any(
                contains_expected_keyword(
                    item["text"],
                    keywords
                )
                for item in retrieved
            )

            reranking_hit = (
                contains_expected_keyword(
                    reranked[0]["text"],
                    keywords
                )
                if reranked
                else False
            )

            if retrieval_hit:
                retrieval_hits += 1

            if reranking_hit:
                reranking_hits += 1

            print(
                "Retrieval hit:",
                retrieval_hit
            )

            print(
                "Reranking Top-1 hit:",
                reranking_hit
            )

        # -----------------------------------------
        # Show best evidence
        # -----------------------------------------

        if reranked:

            print(
                "\nBest evidence:"
            )

            print(
                reranked[0]["text"][:250]
            )

    # =================================================
    # FINAL METRICS
    # =================================================

    total_questions = len(
        test_questions
    )

    recall_at_k = (
        retrieval_hits
        / relevant_total
        if relevant_total
        else 0
    )

    reranking_accuracy = (
        reranking_hits
        / relevant_total
        if relevant_total
        else 0
    )

    threshold_accuracy = (
        threshold_correct
        / total_questions
    )

    average_relevant_score = (
        sum(relevant_scores)
        / len(relevant_scores)
        if relevant_scores
        else 0
    )

    average_irrelevant_score = (
        sum(irrelevant_scores)
        / len(irrelevant_scores)
        if irrelevant_scores
        else 0
    )

    min_relevant_score = (
        min(relevant_scores)
        if relevant_scores
        else 0
    )

    max_irrelevant_score = (
        max(irrelevant_scores)
        if irrelevant_scores
        else 0
    )

    # =================================================
    # RESULTS
    # =================================================

    print("\n")
    print("=" * 75)
    print("FINAL EVALUATION RESULTS")
    print("=" * 75)

    print(
        f"Questions evaluated: "
        f"{total_questions}"
    )

    print(
        f"Relevant questions: "
        f"{relevant_total}"
    )

    print(
        f"Irrelevant questions: "
        f"{total_questions - relevant_total}"
    )

    print()

    print(
        f"Recall@{TOP_K}: "
        f"{recall_at_k:.2%}"
    )

    print(
        "Reranking Top-1 Accuracy: "
        f"{reranking_accuracy:.2%}"
    )

    print(
        "Threshold Classification Accuracy: "
        f"{threshold_accuracy:.2%}"
    )

    print()

    print(
        "Average relevant similarity: "
        f"{average_relevant_score:.4f}"
    )

    print(
        "Average irrelevant similarity: "
        f"{average_irrelevant_score:.4f}"
    )

    print(
        "Lowest relevant similarity: "
        f"{min_relevant_score:.4f}"
    )

    print(
        "Highest irrelevant similarity: "
        f"{max_irrelevant_score:.4f}"
    )

    print(
        "Current relevance threshold: "
        f"{RELEVANCE_THRESHOLD:.2f}"
    )

    print("=" * 75)


if __name__ == "__main__":
    evaluate()