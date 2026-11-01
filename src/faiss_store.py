import fitz
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load PDF
pdf_path = "data/papers/AI-Research-Paper-Assistant.pdf"
document = fitz.open(pdf_path)

text = ""

for page in document:
    text += page.get_text()

document.close()

# 2. Clean text
text = re.sub(r"\s+", " ", text).strip()

# 3. Create chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

# 4. Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# 5. Create FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(embeddings)

print("Vectors stored in FAISS:", index.ntotal)

# 6. Test search
question = "What is this research paper about?"

question_embedding = model.encode([question])

question_embedding = np.array(question_embedding).astype("float32")

distances, indices = index.search(question_embedding, 3)

print("\nQuestion:", question)

print("\nMost relevant chunks:")

for i in indices[0]:
    print("\n--- Chunk", i + 1, "---")
    print(chunks[i])