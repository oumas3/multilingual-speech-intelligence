import librosa

audio, sr = librosa.load("data/test_audio.mp3", sr=16000)

duration = len(audio) / sr

print("Audio loaded ✅")
print("Sample rate:", sr)
print("Number of samples:", len(audio))
print("Duration:", round(duration, 2), "seconds")
print("First 10 audio values:", audio[:10])