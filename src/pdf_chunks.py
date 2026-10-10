from pypdf import PdfReader


pdf_path = "data/paper.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(text):
    end = start + chunk_size

    chunk = text[start:end]

    chunks.append(chunk)

    start = end - overlap


print("Total characters:", len(text))
print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)