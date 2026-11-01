1. About the Project

The AI Research Paper Assistant using RAG and LLM is an AI-powered application designed to make research papers easier to read, understand, and analyze.

Research papers often contain lengthy content and complex technical information. Finding specific details or understanding a paper manually can take a lot of time.

This application works like a personal assistant for research papers. Users can upload one or multiple PDF documents and interact with their content through simple questions.

The application extracts text from uploaded PDFs, divides it into smaller parts, converts the text into embeddings, and stores them using FAISS. When a user asks a question, the system finds the most relevant information and sends it to the Llama 3.2 language model through Ollama to generate an answer.

Along with question answering, the application also provides research paper summarization and comparison features.

In simple words, this project helps students and researchers find information, understand important concepts, summarize documents, and compare research papers without reading every page manually.

# AI Research Paper Assistant using RAG and LLM

## 1. Introduction

The AI Research Paper Assistant is a project that I developed to make reading and understanding research papers easier.

Research papers contain a lot of technical information, and sometimes it takes a long time to find specific details or understand the main idea of a paper.

To solve this problem, I built an application where users can upload research papers in PDF format and interact with them.

Users can ask questions, generate summaries, and compare two research papers using this application.

I developed this project using Python, Streamlit, Retrieval-Augmented Generation (RAG), FAISS, and the Llama 3.2 language model.

## 2. Why I Started This Project

I wanted to build a practical AI/ML project that could solve a real-world problem.

While working with research papers, I realized that reading long documents and finding specific information manually can be time-consuming.

So, I decided to create an AI-based assistant that allows users to get information directly from their uploaded research papers.

The main goal was to make research paper reading easier and save users' time.

## 3. Project Development Journey

This section explains how I developed the project step by step.

### Step 1: Project Planning and Setup

First, I decided the main purpose of the project and the features I wanted to include.

I planned to build an application with the following features:

- Upload research papers in PDF format.
- Ask questions about uploaded papers.
- Generate research paper summaries.
- Compare two research papers.
- Display relevant source information.

After planning, I created the main project folder named:

`AI-Research-Paper-Assistant`

I opened the project in Visual Studio Code and started organizing the files.

### Step 2: Creating the Project Structure

I created separate folders to keep the project files organized.

The main folders included:

- `app` – Contains the Streamlit application.
- `src` – Contains the main RAG implementation.
- `data/papers` – Stores the uploaded research papers.
- `models` – Created for model-related resources.
- `tests` – Created for testing.
- `vectorstore` – Created for vector storage-related resources.

I also created important files such as `requirements.txt`, `.env`, and `README.md`.

### Step 3: Setting Up Python Environment

I created a Python virtual environment to manage the project dependencies separately.

The following command was used:

```bash
python -m venv venv
```

I activated the environment and installed the required Python libraries using:

```bash
pip install -r requirements.txt
```

This helped me prepare the environment for development.

### Step 4: Extracting Text from PDF Files

The next step was to read research papers and extract their text.

For this, I used the PyMuPDF library through its `fitz` module.

I implemented PDF text extraction in the RAG engine.

The application reads the pages of a PDF and collects the available text.

I also added support for processing multiple research papers.

### Step 5: Cleaning the Extracted Text

After extracting text from PDF files, I noticed that the text could contain unnecessary spaces and formatting issues.

To improve the extracted content, I used Python regular expressions.

This helped clean the text before processing it further.

### Step 6: Dividing Text into Smaller Chunks

Research papers can contain a large amount of text.

Instead of processing the complete document at once, I divided the extracted text into smaller parts called chunks.

For this, I used `RecursiveCharacterTextSplitter` from LangChain Text Splitters.

I used these settings:

- Chunk Size: 1000
- Chunk Overlap: 200

Chunk size controls the amount of text in each chunk.

Chunk overlap keeps some text common between consecutive chunks so that important context is not lost.

### Step 7: Generating Text Embeddings

After creating chunks, the next step was to convert them into numerical representations called embeddings.

For this, I used Sentence Transformers with the model:

`all-MiniLM-L6-v2`

This model converts text into numerical vectors of 384 dimensions.

These embeddings help the application find text that has a similar meaning to the user's question.

### Step 8: Creating the FAISS Vector Index

After generating embeddings, I used FAISS to store and search the vectors.

I used:

`IndexFlatL2`

This index performs similarity searches using L2 distance.

When a user asks a question, the application searches the FAISS index and retrieves the top 5 relevant text chunks.

These chunks are then used to prepare the answer.

### Step 9: Implementing the RAG Pipeline

RAG stands for Retrieval-Augmented Generation.

It combines information retrieval with AI-based answer generation.

I connected the following steps to create the RAG pipeline:

1. Receive the user's question.
2. Convert the question into an embedding.
3. Search the FAISS index.
4. Retrieve the top 5 relevant text chunks.
5. Pass the question and retrieved text to the language model.
6. Generate an answer using the retrieved information.

This allows the application to answer questions using the content of uploaded research papers.

### Step 10: Integrating Llama 3.2 using Ollama

To generate answers, I integrated the Llama 3.2 language model.

I used Ollama to run the model.

The model receives the user's question along with relevant information retrieved from the research papers.

It then generates a response based on the provided context.

### Step 11: Connecting the RAG Engine with Streamlit

After implementing the main RAG functionality, I connected it with the Streamlit application.

I created the main application file:

`app/app.py`

The RAG-related functionality was maintained in:

`src/rag_engine.py`

The Streamlit application handles user interaction, while the RAG engine handles document processing and information retrieval.

During development, I also fixed a Python import error:

`ModuleNotFoundError: No module named 'src'`

I corrected the project import path in the application so that the RAG engine could be imported successfully.

### Step 12: Building the Application Interface

I used Streamlit to create the web interface.

The application includes:

- Project title and description.
- PDF upload option.
- Question input area.
- Answer display area.
- Source information.
- Summary functionality.
- Paper comparison functionality.
- Chat history.
- Clear Chat option.

The purpose was to make the application simple to use.

### Step 13: Implementing Question Answering

After connecting the interface with the RAG engine, I implemented the question-answering feature.

Users can enter questions related to the uploaded research papers.

The application retrieves relevant chunks from FAISS and sends them to Llama 3.2.

The generated answer is displayed in the Streamlit interface.

### Step 14: Adding Multiple PDF Support

Initially, I worked with individual PDF documents.

Later, I added support for multiple research papers.

The application can process more than one PDF and use their content for question answering.

I also tested the application using two research papers:

- `Research_on_Machine_Learning_and_Its_Algorithms_an.pdf`
- `An_Interactive_Pedagogical_Framework_for.pdf`

### Step 15: Adding Chat History and Clear Chat

I added chat history so that users could view their previous questions and answers during the current application session.

I also implemented the Clear Chat button.

It allows users to clear the current conversation and start asking new questions.

### Step 16: Implementing Research Paper Summarization

After completing the question-answering functionality, I added a research paper summarization feature.

This feature helps users understand the main content of a research paper without reading the entire document.

The application processes the selected paper and uses the language model to generate a summary.

I also tested the summary feature to check whether it was working correctly.

### Step 17: Implementing Research Paper Comparison

I added another feature that allows users to compare two research papers.

Users can select two different uploaded papers from the application.

The application uses their extracted text and the language model to generate a comparison.

This feature helps users understand the main points and differences between the selected papers.

I tested the comparison functionality and retained the original comparison format that I preferred.

### Step 18: Testing the Application

After implementing the main features, I tested the application to check its working.

The following features were tested:

- Uploading research papers.
- Asking questions about uploaded papers.
- Retrieving relevant information.
- Displaying source information.
- Uploading multiple PDF documents.
- Generating research paper summaries.
- Comparing two research papers.
- Viewing chat history.
- Clearing the chat.

Testing helped me check the main functions of the application.

## 4. Main Features

### PDF Upload

Users can upload research papers in PDF format. The application supports multiple PDF documents.

### Question Answering

Users can ask questions about uploaded research papers and receive answers based on retrieved document information.

### Research Paper Summarization

Users can generate summaries to understand the important information in a research paper.

### Research Paper Comparison

Users can select two research papers and compare their content.

### Source Information

The application displays relevant source information, including PDF names and page information when available.

### Chat History

Users can view previous questions and answers during the current session.

### Clear Chat

Users can clear the current chat and start a new conversation.

## 5. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Building the web application |
| PyMuPDF | Extracting text from PDF files |
| Regular Expressions | Cleaning extracted text |
| LangChain Text Splitters | Dividing text into chunks |
| Sentence Transformers | Generating text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Storing and searching vector embeddings |
| NumPy | Numerical operations |
| Ollama | Running the language model |
| Llama 3.2 | Generating AI responses |
| Visual Studio Code | Writing and managing project code |

## 6. How the Application Works

The complete working process is:

1. User uploads a research paper.
2. PyMuPDF extracts the text.
3. Regular expressions clean the extracted text.
4. LangChain Text Splitters divide the text into chunks.
5. Sentence Transformers generates embeddings.
6. FAISS stores and searches the embeddings.
7. User enters a question.
8. The system retrieves the top 5 relevant chunks.
9. The retrieved information is passed to Llama 3.2 using Ollama.
10. The model generates an answer.
11. Streamlit displays the answer and available source information.

## 7. Project Folder Structure

```text
AI-Research-Paper-Assistant/
│
├── app/
│   └── app.py
│
├── src/
│   └── rag_engine.py
│
├── data/
│   └── papers/
│       ├── Research_on_Machine_Learning_and_Its_Algorithms_an.pdf
│       └── An_Interactive_Pedagogical_Framework_for.pdf
│
├── models/
├── tests/
├── vectorstore/
├── venv/
│
├── .env
├── requirements.txt
└── README.md
```

## 8. Important Project Files

**app/app.py**

This file contains the Streamlit application. It manages the user interface, PDF upload, question input, summaries, comparisons, and chat interactions.

**src/rag_engine.py**

This file contains the main RAG logic, including PDF text extraction, text cleaning, chunking, embedding generation, FAISS search, and language model integration.

**data/papers/**

This folder contains the research papers used in the application.

**requirements.txt**

This file contains the Python libraries required to run the project.

**.env**

This file is used for environment configuration and secret values, if required.

**README.md**

This file contains information about the project, its development, features, technologies, and setup instructions.

## 9. How to Run the Project

### Step 1: Open the Project

Open the `AI-Research-Paper-Assistant` folder in Visual Studio Code.

### Step 2: Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 3: Install the Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4: Run Ollama

Make sure Ollama is installed and the Llama 3.2 model is available.

To download the model:

```bash
ollama pull llama3.2
```

### Step 5: Start the Application

Run the following command from the main project folder:

```bash
streamlit run app/app.py
```

The application will open in the web browser.

## 10. What I Learned

While developing this project, I gained practical experience in:

- Python programming.
- Working with PDF documents.
- Text preprocessing.
- Text chunking.
- Understanding embeddings.
- Using Sentence Transformers.
- Working with FAISS.
- Understanding Retrieval-Augmented Generation.
- Integrating Llama 3.2 using Ollama.
- Building applications with Streamlit.
- Connecting different AI technologies in one project.
- Testing and debugging an application.

## 11. Challenges Faced During Development

During development, I worked on different parts of the application and fixed issues step by step.

Some of the challenges included:

- Setting up the Python environment and dependencies.
- Fixing Python import errors.
- Extracting and processing text from PDF documents.
- Connecting the FAISS search with the language model.
- Supporting multiple PDF documents.
- Improving question-answering and comparison results.
- Testing the different features together.

These challenges helped me understand how the different parts of an AI application work together.

## 12. Future Improvements

In the future, I would like to improve the application by adding:

- More detailed paper comparison.
- Better source references with page numbers.
- Support for scanned PDF documents using OCR.
- Support for additional language models.
- Downloadable summaries and comparison reports.
- Improved conversation memory.
- Online deployment.

## 13. Conclusion

The AI Research Paper Assistant is a practical project that helps users work with research papers more easily.

It brings together PDF text extraction, text processing, embeddings, FAISS, RAG, and a Large Language Model in one application.

Through this project, I learned how to use different AI technologies together to solve a real-world problem.

It also gave me practical experience in developing, connecting, testing, and debugging an AI-based application.

## 14. Project Details

**Project Name:** AI Research Paper Assistant using RAG and LLM

**Domain:** Artificial Intelligence and Machine Learning

**Programming Language:** Python

**Interface:** Streamlit

**Main Technologies:** RAG, FAISS, Sentence Transformers, Ollama, Llama 3.2

**Project Type:** AI/ML Portfolio Project
