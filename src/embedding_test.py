import fitz
from sentence_transformers import SentenceTransformer
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

# Clean text
import re
text = re.sub(r"\s+", " ", text).strip()

# Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks into embeddings
print("Starting embedding generation...")

embeddings = model.encode(chunks)

print("Embedding generation completed.")
print("Embedding shape:", embeddings.shape)