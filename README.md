# 🏥 Evidence-Grounded Medical RAG Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pinecone-Vector%20DB-00A67E?style=for-the-badge&logo=pinecone&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-Inference%20API-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
</p>

<p align="center">
  A fully conversational, evidence-grounded medical Q&A chatbot built on top of the <strong>Gale Encyclopedia of Medicine (2nd Edition)</strong>. Ask any medical question and get concise, context-aware answers — backed by a 637-page knowledge base with full conversation memory.
</p>

---

## ✨ Features

- 📄 **PDF Ingestion Pipeline** — Loads, chunks, and embeds a 637-page medical encyclopedia
- 🔍 **Semantic Search** — Uses Pinecone vector store for fast, accurate retrieval
- 🧠 **Conversation Memory** — Remembers previous turns using `RunnableWithMessageHistory`
- 🔄 **History-Aware Retrieval** — Rephrases follow-up questions before retrieval using chat history
- 💬 **Streamlit Chat UI** — Clean, real-time chat interface with markdown rendering
- 🧩 **Modular Codebase** — Cleanly separated concerns: config, embeddings, vectorstore, chains, UI

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
│                      (run once, offline)                         │
│                                                                  │
│   Medical_book.pdf                                               │
│        │                                                         │
│        ▼                                                         │
│   PyPDFLoader  ──►  RecursiveCharacterTextSplitter               │
│                           │                                      │
│                           ▼                                      │
│               HuggingFaceEmbeddings                              │
│          (all-MiniLM-L6-v2 → 384-dim)                           │
│                           │                                      │
│                           ▼                                      │
│               PineconeVectorStore.from_documents()               │
│                    [Vectors stored ✅]                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                         │
│                      (runs on every query)                       │
│                                                                  │
│   User Question + Chat History                                   │
│        │                                                         │
│        ▼                                                         │
│   history_aware_retriever                                        │
│   (rephrases question using history, then retrieves from         │
│    Pinecone via semantic similarity search  → top-k chunks)      │
│        │                                                         │
│        ▼                                                         │
│   create_stuff_documents_chain                                   │
│   (stuffs retrieved chunks into prompt context)                  │
│        │                                                         │
│        ▼                                                         │
│   Llama-3.2-1B-Instruct (HuggingFace Inference API)             │
│        │                                                         │
│        ▼                                                         │
│   Answer  ──►  ChatMessageHistory (stored for next turn)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗂️ Project Structure

```
Evidence-Grounded-Medical-RAG/
│
├── medChat/                    # Core pipeline modules
│   ├── __init__.py
│   ├── config.py               # Env vars & constants
│   ├── embeddings.py           # HuggingFace embedding model
│   ├── vectorstore.py          # Pinecone connection
│   ├── retriever.py            # Retriever setup
│   ├── chains.py               # RAG chain + memory
│   └── prompts.py              # Prompt templates
│
├── notebooks/                   # Jupyter notebooks
│   └── notebook.ipynb       # Full pipeline experiments
│
├── app.py                      # Streamlit UI entry point
├── .env                        # API keys (not committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🧠 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| **Document Loader** | `PyPDFLoader` | Load and parse the medical PDF |
| **Text Splitter** | `RecursiveCharacterTextSplitter` | Chunk documents into overlapping segments |
| **Embedding Model** | `sentence-transformers/all-MiniLM-L6-v2` | Generate 384-dim dense embeddings |
| **Vector Store** | `Pinecone` | Store & retrieve embeddings via ANN search |
| **LLM** | `meta-llama/Llama-3.2-1B-Instruct` | Answer generation via HuggingFace Inference API |
| **RAG Framework** | `LangChain (LCEL)` | Chain orchestration, memory, retrieval |
| **Memory** | `RunnableWithMessageHistory` | Per-session conversation history |
| **UI** | `Streamlit` | Interactive chat interface |

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/Tipto-Ghosh/Evidence-Grounded-Medical-RAG.git
cd Evidence-Grounded-Medical-RAG
```

### 2. Create and activate a conda environment

```bash
conda create -n medical-rag python=3.10 -y
conda activate medical-rag
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

> 🔑 Get your Pinecone API key at [app.pinecone.io](https://app.pinecone.io)
> 🤗 Get your HuggingFace token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Running the App

### Step 1 — Run the ingestion pipeline (first time only)

If you haven't already uploaded embeddings to Pinecone, run the ingestion notebook:

```bash
jupyter notebook research/experiments.ipynb
```

> ⚠️ Skip this step if embeddings are already stored in your Pinecone index.

### Step 2 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💬 Example Conversation

```
You:       What is Acromegaly?
MedBot:    Acromegaly is a hormonal disorder caused by excess growth hormone
           in adults, typically due to a benign pituitary tumor. It leads to
           enlargement of the hands, feet, and facial features over time.

You:       What are its symptoms?

MedBot:    Symptoms include enlarged hands and feet, coarsened facial features,
           joint pain, fatigue, and sleep apnea. Because this follows from your
           earlier question about Acromegaly, I can confirm these symptoms are
           specifically associated with that condition.

You:       How is it treated?

MedBot:    Treatment options include surgical removal of the pituitary tumor,
           radiation therapy, and medications such as somatostatin analogs that
           suppress growth hormone production.
```

> The chatbot correctly resolves follow-up questions like *"What are its symptoms?"* without needing the user to repeat context — powered by the history-aware retriever.

---

## 📦 Requirements

```txt
langchain
langchain-community
langchain-huggingface
langchain-pinecone
pinecone>=5.0.0
sentence-transformers
streamlit
python-dotenv
pypdf
huggingface_hub
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔍 How RAG Works (Simplified)

```
1. RETRIEVAL
   User question → embedded into a vector → 
   Pinecone finds the top-5 most similar chunks from the medical book

2. AUGMENTATION
   Retrieved chunks are injected into the LLM prompt as context

3. GENERATION
   LLM reads the context + question → generates a grounded answer
   (answers are always tied to the source material, not hallucinated)
```

This approach grounds every answer in the actual medical encyclopedia, significantly reducing hallucination compared to a plain LLM.

---

## 📊 Chunking Strategy

| Parameter | Value |
|---|---|
| `chunk_size` | 500 |
| `chunk_overlap` | 50 |
| `splitter` | `RecursiveCharacterTextSplitter` |
| `total_pages` | 637 |

Overlapping chunks ensure that context spanning chunk boundaries is not lost during retrieval.

---

## 🧩 Memory Design

Each user session gets a unique `session_id`. The `RunnableWithMessageHistory` wrapper:

1. Loads the chat history for that session before each invocation
2. Appends the new `(user, assistant)` turn after each invocation
3. Passes the full history to the `history_aware_retriever` for query rewriting

This means follow-up questions like *"What are its symptoms?"* are correctly resolved to *"What are the symptoms of Acromegaly?"* before hitting Pinecone.

---

## ⚠️ Limitations

- **LLM size** — `Llama-3.2-1B-Instruct` is lightweight; answers may be less detailed than larger models
- **Knowledge scope** — Limited to the Gale Encyclopedia of Medicine (2nd Edition)
- **Free tier Pinecone** — Indexes may be deleted after 7 days of inactivity
- **Not a medical substitute** — This is a portfolio/research project, not a clinical decision tool

---

## 🛣️ Future Improvements

- [ ] Add source citation (show which page/section the answer came from)
- [ ] Upgrade LLM to `Llama-3.1-8B` or `Qwen2.5-7B-Instruct` for better answers
- [ ] Add a confidence indicator based on retrieval similarity scores
- [ ] Implement hybrid search (dense + sparse) in Pinecone
- [ ] Add multi-PDF support for broader medical knowledge
- [ ] Deploy on Streamlit Cloud or Hugging Face Spaces

---

## 👤 Author

**Tipto Ghosh**
Undergraduate CS Student | AI/ML Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Tipto--Ghosh-181717?style=flat&logo=github)](https://github.com/Tipto-Ghosh)

---

## 📄 License

This project is licensed under the MIT License. The medical content used (Gale Encyclopedia of Medicine) is used solely for educational and research purposes.

---

<p align="center">
  Built with ❤️ using LangChain · Pinecone · HuggingFace · Streamlit
</p>