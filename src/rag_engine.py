
import os
import fitz
import re
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_rag_engine(pdf_paths):

    # 1. LOAD ALL PDFS

    all_chunks = []
    chunk_sources = []
    paper_texts = {}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for pdf_path in pdf_paths:

        document = fitz.open(pdf_path)
        paper_name = os.path.basename(pdf_path)
        paper_pages = []

        try:
            for page_number, page in enumerate(document, start=1):

                text = page.get_text()
                text = re.sub(r"\s+", " ", text).strip()

                if not text:
                    continue

                paper_pages.append(text)
                page_chunks = splitter.split_text(text)

                for chunk in page_chunks:
                    all_chunks.append(chunk)
                    chunk_sources.append({
                        "pdf": paper_name,
                        "page": page_number
                    })
        finally:
            document.close()

        paper_texts[paper_name] = "\n\n".join(paper_pages)

    if not all_chunks:
        raise ValueError(
            "No readable text found in the uploaded PDFs."
        )

    # 2. CREATE EMBEDDINGS

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(all_chunks)
    embeddings = np.array(embeddings).astype("float32")

    # 3. CREATE FAISS INDEX

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # 4. ASK QUESTION FUNCTION (EXISTING Q&A)

    def ask_question(question, chat_history=None):

        if chat_history is None:
            chat_history = []

        search_question = question.strip()

        short_query_map = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "dl": "deep learning"
        }

        if search_question.lower() in short_query_map:
            search_question = short_query_map[
                search_question.lower()
            ]

        question_embedding = model.encode([search_question])
        question_embedding = np.array(
            question_embedding
        ).astype("float32")

        distances, indices = index.search(
            question_embedding, 5
        )

        relevant_chunks = []
        sources = []

        for i in indices[0]:

            if i < 0:
                continue

            relevant_chunks.append(all_chunks[i])
            source = chunk_sources[i]

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(relevant_chunks)

        previous_conversation = ""

        for chat in chat_history[-5:]:

            previous_conversation += f"""
User:
{chat["question"]}

Assistant:
{chat["answer"]}

"""

        prompt = f"""
You are an AI Research Paper Assistant.

Answer using ONLY the Research Paper Context.

SHORT FORM MAPPINGS:
ML = Machine Learning
AI = Artificial Intelligence
DL = Deep Learning

IMPORTANT INSTRUCTIONS:

1. Give a detailed, meaningful and well-explained answer.
2. Do not give unnecessarily short answers.
3. Include all important relevant points.
4. Explain clearly using paragraphs or bullet points.
5. Use ONLY information from the research paper context.
6. Do NOT use outside knowledge.
7. Do NOT invent facts.
8. Use previous conversation to understand follow-up questions.
9. If the information is absent, respond with:

I could not find this information in the research paper.

10. Do not answer unavailable questions using general knowledge.
11. Do not mention the context unnecessarily.
12. For simple questions such as "ML", explain the meaning
    and relevant information available in the paper.

Previous Conversation:
{previous_conversation}

Research Paper Context:
{context}

Current User Question:
{search_question}

Now provide a detailed and accurate answer based ONLY
on the research paper context.
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        not_found_phrases = [
            "i could not find",
            "i couldn't find",
            "could not find",
            "couldn't find",
            "not available in the research paper",
            "not mentioned in the research paper",
            "not found in the research paper",
            "information is not available",
            "information is unavailable"
        ]

        if any(
            phrase in answer.lower()
            for phrase in not_found_phrases
        ):
            sources = []

        return {
            "answer": answer,
            "sources": sources
        }

    # 5. EXISTING SUMMARY FUNCTION (FORMAT PRESERVED)

    def summarize_paper():

        full_text = "\n\n".join(all_chunks)
        max_characters = 30000

        if len(full_text) > max_characters:
            full_text = full_text[:max_characters]

        summary_prompt = f"""
You are an AI Research Paper Summarization Assistant.

Summarize the research paper using ONLY the information
provided below.

Do NOT use outside knowledge.
Do NOT invent facts.

Create a structured summary with exactly these sections:

## 📌 Abstract / Main Idea

Explain the main purpose, problem and overall idea
of the research paper.

## 🔬 Methodology

Explain the methods, algorithms, approaches,
experiments or techniques used in the research.

## 📊 Key Findings

Explain the important results, observations or findings
reported in the research paper.

## ⚠️ Limitations

Mention limitations, challenges, weaknesses or constraints
that are explicitly discussed or clearly stated in the paper.

## 🎯 Conclusion

Explain the main conclusion and overall takeaway
of the research paper.

IMPORTANT:

- Use only information available in the paper.
- Do not add general knowledge.
- Do not make unsupported assumptions.
- Keep each section meaningful and detailed.
- If a section is not clearly available in the paper,
  write:

"Not clearly stated in the research paper."

Research Paper:
{full_text}

Now generate the structured research paper summary.
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ]
        )

        summary = response["message"]["content"].strip()

        summary_sources = []

        for source in chunk_sources:
            if source not in summary_sources:
                summary_sources.append(source)

        return {
            "summary": summary,
            "sources": summary_sources
        }

    # 6. NEW COMPARISON FUNCTION

    def compare_papers(paper_name_1, paper_name_2):

        if paper_name_1 == paper_name_2:
            raise ValueError(
                "Please select two different research papers."
            )

        if paper_name_1 not in paper_texts:
            raise ValueError(
                f"Paper not found: {paper_name_1}"
            )

        if paper_name_2 not in paper_texts:
            raise ValueError(
                f"Paper not found: {paper_name_2}"
            )

        paper_1_text = paper_texts[paper_name_1]
        paper_2_text = paper_texts[paper_name_2]

        if not paper_1_text.strip() or not paper_2_text.strip():
            raise ValueError(
                "One or both PDFs contain no readable text."
            )

        # Keep prompt size manageable.

        max_chars_per_paper = 15000

        paper_1_text = paper_1_text[:max_chars_per_paper]
        paper_2_text = paper_2_text[:max_chars_per_paper]

        comparison_prompt = f"""
You are an AI Research Paper Comparison Assistant.

Compare the TWO research papers below.

PAPER 1: {paper_name_1}

{paper_1_text}

PAPER 2: {paper_name_2}

{paper_2_text}

IMPORTANT INSTRUCTIONS:

1. Use only information supported by the papers.
2. Do not use outside knowledge or invent results.
3. Keep the information from each paper separate.
4. If information is absent, clearly state that.
5. Explain similarities and differences accurately.

Return the comparison with these sections:

## 📌 1. Main Idea
Explain each paper's main idea separately.

## 🔬 2. Methodology / Approach
Compare the methods and approaches described in each paper.

## 📊 3. Key Findings
Explain and compare the findings of both papers.

## ⚠️ 4. Limitations
Report only limitations supported by the papers.

## 🎯 5. Conclusions
Summarize and compare the conclusions.

## 🧠 6. Overall Comparison
Explain similarities, differences and distinct contributions.

## 📋 7. Comparison Table
Create a Markdown table with columns:
Category | Paper 1 | Paper 2

Include main idea, methodology, key findings,
limitations and conclusion.

Now generate the comparison.
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": comparison_prompt
                }
            ]
        )

        comparison = response["message"]["content"].strip()

        return {
            "comparison": comparison,
            "papers": [paper_name_1, paper_name_2]
        }

    # 7. RETURN ALL THREE FUNCTIONS

    return ask_question, summarize_paper, compare_papers
