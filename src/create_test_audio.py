from gtts import gTTS

text = "Hello, my name is Oumeyma and I am testing my multilingual speech intelligence project."

tts = gTTS(text=text, lang="en")
tts.save("data/test_audio.mp3")

print("Test audio created ✅")