import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_name = "openai/whisper-tiny"

print("Loading Whisper...")

processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

print("Whisper loaded ✅")


def transcribe_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=16000
    )

    inputs = processor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    generated_ids = model.generate(
        inputs.input_features
    )

    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return transcription