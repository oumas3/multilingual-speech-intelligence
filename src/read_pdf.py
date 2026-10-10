from pypdf import PdfReader

pdf_path = "data/paper.pdf"

reader = PdfReader(pdf_path)

print("Number of pages:", len(reader.pages))

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("\nExtracted text preview:\n")
print(text[:1000])