import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-tiny"

print("Loading Whisper...")

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

print("Whisper loaded ✅")

# Load our audio
audio, sr = librosa.load(
    "data/test_audio.mp3",
    sr=16000
)

print("\nAudio information:")
print("Sample rate:", sr)
print("Number of audio samples:", len(audio))

# Prepare the audio for Whisper
inputs = processor(
    audio,
    sampling_rate=sr,
    return_tensors="pt"
)

print("\nWhisper processor output:")
print("Input feature shape:", inputs.input_features.shape)
print("\nRunning Whisper model...")

generated_ids = model.generate(inputs.input_features)

print("Generated token IDs:")
print(generated_ids)
transcription = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print("\nTranscription:")
print(transcription)