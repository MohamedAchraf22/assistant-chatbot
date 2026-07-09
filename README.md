# 🤖 Assistant Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI**, **LangChain**, **Google Gemini**, **ChromaDB**, and **MinIO**.

The project supports **incremental ingestion**, **automatic synchronization**, and **conversation memory**, allowing the knowledge base to stay up-to-date without rebuilding the vector database every time documents change.

---

# Features

* RAG (Retrieval-Augmented Generation)
* FastAPI backend
* Chainlit chat interface
* Chroma vector database
* MinIO (S3-compatible object storage)
* Incremental document ingestion
* Automatic polling-based synchronization
* Conversation memory using a separate Chroma collection
* Semantic search with Sentence Transformers
* Metadata-based document management
* Automatic handling of:

  * New documents
  * Updated documents
  * Deleted documents

---

# Architecture

```text
                    +----------------------+
                    |   Upload Documents   |
                    +----------+-----------+
                               |
                               v
                      +------------------+
                      |      MinIO       |
                      | Knowledge Bucket |
                      +---------+--------+
                                |
                                | Polling
                                v
                    +------------------------+
                    | Snapshot Generator     |
                    +-----------+------------+
                                |
                                v
                    +------------------------+
                    | Change Detection       |
                    | New / Update / Delete  |
                    +-----------+------------+
                                |
                                v
                 +-------------------------------+
                 | Incremental Ingestion Service |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 | Recursive Text Splitter       |
                 +---------------+---------------+
                                 |
                                 v
                 +-------------------------------+
                 | Sentence Transformer          |
                 | Embedding Generation          |
                 +---------------+---------------+
                                 |
                                 v
                        +------------------+
                        |     ChromaDB     |
                        +---------+--------+
                                  |
                                  v
User → FastAPI → RAG Retrieval → Gemini → Final Answer
```

---

# Tech Stack

* Python
* FastAPI
* Chainlit
* LangChain
* Google Gemini
* ChromaDB
* MinIO
* APScheduler
* Sentence Transformers
* Docker

---

# Project Structure

```text
assistant-chatbot/

├── app.py
├── main.py
├── rag.py
├── vector_store.py
├── ingest.py
├── config.py
├── llm.py
├── conversation_memory.py
├── chainlit_app.py
│
├── storage/
│   └── minio_client.py
│
├── state/
│   ├── comparator.py
│   ├── scheduler.py
│   ├── snapshot.py
│   ├── state_manager.py
│   └── sync.py
│
├── chroma/
│
├── data/
│
├── ingestion_state.json
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/MohamedAchraf22/assistant-chatbot.git

cd assistant-chatbot
```

---

Create a virtual environment.

Windows

```bash
python -m venv .venv

.venv\Scripts\Activate.ps1
```

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

Install dependencies.

```bash
pip install -U pip

pip install -r requirements.txt
```

---

# Configure Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=YOUR_API_KEY

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=rag-knowledge-base
```

Adjust the values if your MinIO server uses different credentials.

---

# Start MinIO

Run MinIO using Docker.

```bash
docker run ^
  -p 9000:9000 ^
  -p 9001:9001 ^
  -e "MINIO_ROOT_USER=minioadmin" ^
  -e "MINIO_ROOT_PASSWORD=minioadmin" ^
  -v minio_data:/data ^
  quay.io/minio/minio server /data --console-address ":9001"
```

Open the MinIO Console:

```
http://localhost:9001
```

Login using your configured credentials.

---

# Create the Bucket

Create a bucket named

```
rag-knowledge-base
```

Upload your Markdown documents while preserving your folder structure.

Example:

```text
rag-knowledge-base/

HR/
    policies/
        vacation.md

MentalHealth/
    OCD/
        common_obsessions_compulsions_ocd.md

Sports/
    world_cup_2022.md
```

---

# Configure Storage Provider

In `config.py`

```python
STORAGE_PROVIDER = "minio"
```

---

# Build the Vector Database

Run the ingestion process once.

```bash
python ingest.py
```

This will:

* Read all documents from MinIO
* Split them into chunks
* Generate embeddings
* Store them inside ChromaDB

---

# Incremental Synchronization

The project supports incremental ingestion.

Instead of rebuilding the entire vector database, only changed documents are processed.

Detected operations include:

* New documents
* Updated documents
* Deleted documents

To synchronize manually:

```bash
python -m state.sync
```

---

# Automatic Synchronization

A background scheduler periodically checks MinIO for changes.

It automatically:

* Detects new files
* Detects updated files
* Removes deleted documents from Chroma
* Adds new embeddings

The scheduler starts automatically when the FastAPI application starts.

Polling interval can be configured inside:

```
state/scheduler.py
```

---

# Running the Backend

Start FastAPI.

```bash
uvicorn app:app --reload
```

Default address

```
http://localhost:8000
```

---

# Chainlit UI

Run the chat interface.

```bash
chainlit run chainlit_app.py -w
```

The UI communicates with the FastAPI backend.

---

# CLI Mode

You can also use the chatbot directly from the terminal.

```bash
python main.py
```

---

# API

## POST

```
/chat
```

Request

```json
{
    "question":"What is OCD?"
}
```

Response

```json
{
    "answer":"..."
}
```

---

# Conversation Memory

The chatbot stores previous conversations inside a separate Chroma collection.

Conversation history is retrieved semantically and injected into the prompt to provide contextual responses.

---

# How Incremental Ingestion Works

1. Generate a snapshot of all MinIO objects.
2. Compare it with the previous snapshot.
3. Detect:

   * New files
   * Updated files
   * Deleted files
4. Process only changed documents.
5. Save the latest snapshot.

This avoids rebuilding the entire vector database after every document update.

---

# Troubleshooting

### MinIO returns `NoSuchKey`

Verify the object path passed to MinIO exactly matches the uploaded file.

---

### JSONDecodeError when running sync

Delete the empty `ingestion_state.json` file and run synchronization again.

---

### No answers returned

Verify that:

* The document exists in MinIO.
* The document has been ingested.
* Chroma contains the document.
* Retrieval threshold is not too strict.

---

### Embedding model downloads every run

The first execution downloads the embedding model.

Subsequent executions reuse the cached model.

---


