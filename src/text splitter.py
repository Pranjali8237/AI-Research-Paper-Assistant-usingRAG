import fitz
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

# PDF location
pdf_path = "data/papers/AI-Research-Paper-Assistant.pdf"

# Open PDF
document = fitz.open(pdf_path)

# Extract text
text = ""

for page in document:
    text += page.get_text()

document.close()

# -----------------------------
# TEXT CLEANING
# -----------------------------

# Replace multiple spaces with one space
text = re.sub(r"\s+", " ", text)

# Remove unnecessary spaces at beginning and end
text = text.strip()

print("Cleaned characters:", len(text))

# -----------------------------
# TEXT CHUNKING
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# Display first 5 chunks
for i, chunk in enumerate(chunks[:5]):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
print("\n==============================")
print("CLEANED CHARACTERS:", len(text))
print("TOTAL CHUNKS:", len(chunks))
print("==============================")