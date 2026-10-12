
from pathlib import Path

import librosa
import torch
from transformers import AutoFeatureExtractor, WavLMModel

MODEL_NAME = "microsoft/wavlm-base-plus"

print("Loading WavLM...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = WavLMModel.from_pretrained(MODEL_NAME)
model.eval()

print("WavLM loaded!")


def extract_audio_embeddings(audio_path: str) -> torch.Tensor:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, _ = librosa.load(str(audio_path), sr=16000, mono=True)

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Average the frame-level representations into one audio vector.
    audio_embedding = outputs.last_hidden_state.mean(dim=1)

    return audio_embedding.squeeze(0)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    audio_path = project_root / "data" / "test_audio.mp3"

    embedding = extract_audio_embeddings(str(audio_path))

    print("Audio embedding shape:", tuple(embedding.shape))
    print("First 10 values:", embedding[:10])
