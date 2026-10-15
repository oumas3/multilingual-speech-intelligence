from pathlib import Path

from transcribe import transcribe_audio
from audio_embeddings import extract_audio_embeddings
from llm import analyze_transcript
from rag import rag_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / "data" / "test_audio.mp3"


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    print("\n========================================")
    print(" MULTILINGUAL SPEECH INTELLIGENCE")
    print("========================================")

    # 1. Whisper
    print("\n[1/4] Transcribing audio with Whisper...")

    transcript = transcribe_audio(str(AUDIO_PATH))

    if not transcript.strip():
        raise ValueError("Whisper returned an empty transcript.")

    print("\nTranscript:")
    print(transcript)

    # 2. WavLM
    print("\n[2/4] Extracting WavLM audio features...")

    audio_embedding = extract_audio_embeddings(str(AUDIO_PATH))

    print("Audio embedding shape:")
    print(tuple(audio_embedding.shape))

    # 3. LLM speech analysis
    print("\n[3/4] Analyzing transcript with LLM...")

    summary = analyze_transcript(transcript)

    print("\nTranscript analysis:")
    print(summary)

    # 4. RAG
    print("\n[4/4] Research-document RAG")

    question = input(
        "\nAsk a question about your research PDF: "
    ).strip()

    if not question:
        question = transcript

    result = rag_answer(question)

    print("\nQuestion:")
    print(question)

    print("\nRAG answer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")

    print("\n========================================")
    print(" PIPELINE FINISHED SUCCESSFULLY")
    print("========================================")


if __name__ == "__main__":
    main()