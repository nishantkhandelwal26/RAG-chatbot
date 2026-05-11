# Conversational RAG PDF Chatbot

A conversational AI chatbot built with **Streamlit**, **LangChain**, **Groq LLMs**, and **Hugging Face Embeddings** that allows users to upload PDF documents and ask questions based on their content using **Retrieval-Augmented Generation (RAG)**.

## Live Demo

🚀 **Try the App:**  
https://rag-pdf-chat-bot.streamlit.app/

## GitHub Repository

📂 **Source Code:**  
https://github.com/nishantkhandelwal26/RAG-chatbot

---

# Features

- 📄 Upload multiple PDF files
- 💬 Conversational question answering
- 🧠 Context-aware chat history
- 🔍 Retrieval-Augmented Generation (RAG)
- ⚡ Fast responses using Groq LLMs
- 🗂 Session-based memory support
- 🔗 Vector storage with ChromaDB
- 🤗 Hugging Face sentence embeddings

---

# Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq (`Llama 4 Scout`)
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
- **PDF Processing:** PyPDFLoader

---

# Project Architecture

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embeddings Generation
    ↓
Chroma Vector Store
    ↓
Retriever
    ↓
History-Aware RAG Chain
    ↓
LLM Response
```

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/nishantkhandelwal26/RAG-chatbot.git

cd RAG-chatbot
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Create `.env` File

Create a `.env` file in the root directory:

```env
HF_TOKEN=your_huggingface_token
```

---

## 5. Run the Application

```bash
streamlit run app.py
```

---

# How It Works

1. User uploads one or more PDF documents.
2. PDFs are loaded using `PyPDFLoader`.
3. Documents are split into chunks using `RecursiveCharacterTextSplitter`.
4. Chunks are converted into embeddings using Hugging Face embeddings.
5. Embeddings are stored in ChromaDB.
6. User asks questions about the uploaded PDFs.
7. LangChain retrieves relevant chunks.
8. Groq LLM generates contextual answers using retrieved data and chat history.

---

# Key Components

## Embedding Model

```python
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

## LLM Used

```python
ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct"
)
```

## Vector Store

```python
Chroma.from_documents()
```

---

# Example Use Cases

- 📚 Research paper assistant
- 📖 PDF-based Q&A system
- 🏢 Company document chatbot
- 🎓 Study material assistant
- 📑 Resume or report analysis

---

# Future Improvements

- Persistent vector database support
- Authentication system
- Better UI/UX
- Streaming responses
- Multi-user deployment
- Source citations in answers

---

# Screenshots

_Add screenshots of your application here._

---

# Requirements

Example dependencies:

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-groq
langchain-huggingface
langchain-chroma
chromadb
pypdf
python-dotenv
sentence-transformers
```

---

# Deployment

This project is deployed on **Streamlit Cloud**.

To deploy:

1. Push your code to GitHub
2. Connect repository with Streamlit Cloud
3. Add environment variables
4. Deploy

---

# Author

## 👨‍💻 Nishant Khandelwal

- B.Tech ECE Student at MNIT Jaipur
- Competitive Programmer
- AI & Backend Developer

### Connect With Me

- GitHub: https://github.com/nishantkhandelwal26

---