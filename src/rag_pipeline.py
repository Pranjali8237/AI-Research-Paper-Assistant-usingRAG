import fitz
import re
import faiss
import numpy as np
import ollama
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

# 6. Continuous question-answering

while True:

    question = input("\nEnter your question (type 'exit' to stop): ")

    if question.strip() == "":
          print("Please enter a question.")
          continue

    if question.lower() == "exit":
         print("Exiting the AI Research Paper Assistant...")
         break

    # 7. Convert question into embedding
    question_embedding = model.encode([question])

    question_embedding = np.array(question_embedding).astype("float32")

    # 8. Search FAISS
    distances, indices = index.search(question_embedding, 5)

    # 9. Get relevant chunks
    relevant_chunks = []

    for i in indices[0]:
        relevant_chunks.append(chunks[i])

    # 10. Combine chunks
    context = "\n\n".join(relevant_chunks)

    # 11. Send context + question to Llama
    prompt = f"""You are an AI Research Paper Assistant.

Answer the user's question using only the information provided in the research paper context.

If the answer is not available in the context, say:
"I could not find this information in the research paper."

Do not make up information.

Research Paper Context:
{context}

User Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 12. Display answer
    answer = response["message"]["content"]

    print("\n===== FINAL ANSWER =====")
    print(answer)