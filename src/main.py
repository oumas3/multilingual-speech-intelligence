from transcribe import transcribe_audio
from llm import analyze_transcript


audio_path = "data/test_audio.mp3"

print("\nStep 1: Transcribing audio...")

transcript = transcribe_audio(audio_path)

print("\nTranscript:")
print(transcript)


print("\nStep 2: Analyzing transcript...")

answer = analyze_transcript(transcript)

print("\nLLM response:")
print(answer)