import os
import fitz
import re
import faiss
import numpy as np
import ollama

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_rag_engine(pdf_paths):

    # ------------------------------------------------
    # 1. LOAD ALL PDFS
    # ------------------------------------------------

    all_chunks = []
    chunk_sources = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for pdf_path in pdf_paths:

        document = fitz.open(pdf_path)

        for page_number, page in enumerate(
            document,
            start=1
        ):

            text = page.get_text()

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if not text:
                continue

            page_chunks = splitter.split_text(text)

            for chunk in page_chunks:

                all_chunks.append(chunk)

                chunk_sources.append(
                    {
                        "pdf": os.path.basename(pdf_path),
                        "page": page_number
                    }
                )

        document.close()


    # ------------------------------------------------
    # 2. CREATE EMBEDDINGS
    # ------------------------------------------------

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = model.encode(
        all_chunks
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")


    # ------------------------------------------------
    # 3. CREATE FAISS INDEX
    # ------------------------------------------------

    index = faiss.IndexFlatL2(
        embeddings.shape[1]
    )

    index.add(
        embeddings
    )


    # ------------------------------------------------
    # 4. ASK QUESTION FUNCTION
    # ------------------------------------------------

    def ask_question(
        question,
        chat_history=None
    ):

        if chat_history is None:
            chat_history = []


        # ------------------------------------------------
        # 5. SHORT FORM HANDLING
        # ------------------------------------------------

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


        # ------------------------------------------------
        # 6. QUESTION EMBEDDING
        # ------------------------------------------------

        question_embedding = model.encode(
            [search_question]
        )

        question_embedding = np.array(
            question_embedding
        ).astype("float32")


        # ------------------------------------------------
        # 7. FAISS SEARCH
        # ------------------------------------------------

        distances, indices = index.search(
            question_embedding,
            5
        )


        # ------------------------------------------------
        # 8. GET RELEVANT CHUNKS
        # ------------------------------------------------

        relevant_chunks = []
        sources = []

        for i in indices[0]:

            if i < 0:
                continue

            relevant_chunks.append(
                all_chunks[i]
            )

            source = chunk_sources[i]

            if source not in sources:

                sources.append(
                    source
                )


        # ------------------------------------------------
        # 9. CREATE CONTEXT
        # ------------------------------------------------

        context = "\n\n".join(
            relevant_chunks
        )


        # ------------------------------------------------
        # 10. CHAT HISTORY
        # ------------------------------------------------

        previous_conversation = ""

        for chat in chat_history[-5:]:

            previous_conversation += f"""
User:
{chat["question"]}

Assistant:
{chat["answer"]}

"""


        # ------------------------------------------------
        # 11. PROMPT
        # ------------------------------------------------

        prompt = f"""
You are an AI Research Paper Assistant.

Your job is to answer the user's question using ONLY
the information provided in the Research Paper Context.

SHORT FORM MAPPINGS:

ML = Machine Learning
AI = Artificial Intelligence
DL = Deep Learning

IMPORTANT INSTRUCTIONS:

1. Give a detailed, meaningful and well-explained answer.

2. Do not give unnecessarily short answers.

3. If the research paper contains multiple relevant
points, include all important points.

4. Explain concepts clearly using paragraphs or bullet
points when appropriate.

5. Use ONLY information from the research paper context.

6. Do NOT use outside knowledge.

7. Do NOT invent facts.

8. You may use previous conversation to understand
follow-up questions.

9. If the requested information is NOT present in the
research paper context, respond with:

I could not find this information in the research paper.

10. If the information is unavailable, do not try to
answer using your general knowledge.

11. Do not mention the research paper context unnecessarily.

12. For a simple question such as "ML", explain the
meaning and the relevant information available in the
research paper rather than giving only one sentence.

Previous Conversation:
{previous_conversation}

Research Paper Context:
{context}

Current User Question:
{search_question}

Now provide a detailed and accurate answer based ONLY
on the research paper context.
"""


        # ------------------------------------------------
        # 12. CALL LLAMA
        # ------------------------------------------------

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )


        # ------------------------------------------------
        # 13. GET ANSWER
        # ------------------------------------------------

        answer = response[
            "message"
        ][
            "content"
        ].strip()


        # ------------------------------------------------
        # 14. HIDE SOURCES FOR NOT-FOUND ANSWERS
        # ------------------------------------------------

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


        answer_lower = answer.lower()


        if any(
            phrase in answer_lower
            for phrase in not_found_phrases
        ):

            sources = []


        # ------------------------------------------------
        # 15. RETURN ANSWER + SOURCES
        # ------------------------------------------------

        return {
            "answer": answer,
            "sources": sources
        }


    # ------------------------------------------------
    # 16. SUMMARIZE RESEARCH PAPER
    # ------------------------------------------------

    def summarize_paper():

        # Combine all chunks from uploaded papers
        full_text = "\n\n".join(
            all_chunks
        )


        # Keep prompt size manageable
        max_characters = 30000

        if len(full_text) > max_characters:

            full_text = full_text[
                :max_characters
            ]


        # ------------------------------------------------
        # SUMMARY PROMPT
        # ------------------------------------------------

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


        # ------------------------------------------------
        # CALL LLAMA
        # ------------------------------------------------

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": summary_prompt
                }
            ]
        )


        # ------------------------------------------------
        # GET SUMMARY
        # ------------------------------------------------

        summary = response[
            "message"
        ][
            "content"
        ].strip()


        # ------------------------------------------------
        # SUMMARY SOURCES
        # ------------------------------------------------

        summary_sources = []

        for source in chunk_sources:

            if source not in summary_sources:

                summary_sources.append(
                    source
                )


        # ------------------------------------------------
        # RETURN SUMMARY
        # ------------------------------------------------

        return {
            "summary": summary,
            "sources": summary_sources
        }


    # ------------------------------------------------
    # 17. RETURN BOTH FUNCTIONS
    # ------------------------------------------------

    return ask_question, summarize_paper