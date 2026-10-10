with open("data/sample_document.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("Original document:\n")
print(text)

paragraphs = [p.strip() for p in text.splitlines() if p.strip()]

print("\nNumber of chunks:", len(paragraphs))

for i, chunk in enumerate(paragraphs):
    print(f"\nChunk {i + 1}:")
    print(chunk)