
from pathlib import Path

from transcribe import transcribe_audio
from llm import analyze_transcript
from rag import rag_answer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIO_PATH = PROJECT_ROOT / "data" / "test_audio.mp3"


def main():
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            f"Audio file not found: {AUDIO_PATH}"
        )

    print("\n=== MULTILINGUAL SPEECH INTELLIGENCE ===")

    # Step 1: Convert speech into text.
    print("\n[1/3] Transcribing audio...")
    transcript = transcribe_audio(str(AUDIO_PATH))

    print("\nTranscript:")
    print(transcript)

    if not transcript.strip():
        raise ValueError("Whisper returned an empty transcript.")

    # Step 2: Summarize the recognized speech.
    print("\n[2/3] Analyzing transcript...")
    summary = analyze_transcript(transcript)

    print("\nTranscript analysis:")
    print(summary)

    # Step 3: Search research PDFs using the transcript.
    print("\n[3/3] Retrieving research evidence...")

    question = input(
        "\nAsk a question about your research PDF: "
    ).strip()

    if not question:
        question = transcript

    result = rag_answer(question)

    print("\nRAG answer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")

    print("\n=== PIPELINE FINISHED ===")


if __name__ == "__main__":
    main()
