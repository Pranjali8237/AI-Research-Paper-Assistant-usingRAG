import fitz

pdf_path = "data/papers/Research_on_Machine_Learning_and_Its_Algorithms_an.pdf"

document = fitz.open(pdf_path)

text = ""

for page in document:
    text += page.get_text()

print(text)

document.close()
