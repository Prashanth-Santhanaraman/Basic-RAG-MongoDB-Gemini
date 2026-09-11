# Basic RAG with MongoDB Atlas and Gemini

A simple **Retrieval-Augmented Generation (RAG)** project built from scratch using Python, Sentence Transformers, MongoDB Atlas Vector Search, and Google Gemini.

The project demonstrates the core concepts of RAG without using frameworks such as LangChain.

## 🚀 What is RAG?

**Retrieval-Augmented Generation (RAG)** combines information retrieval with Large Language Models (LLMs).

Instead of asking an LLM to answer a question only from its trained knowledge, RAG first retrieves relevant information from a knowledge base and provides that information to the LLM as context.

### RAG Flow

```text
                User Question
                      |
                      v
             Create Query Embedding
                      |
                      v
              MongoDB Vector Search
                      |
                      v
             Retrieve Relevant Chunks
                      |
                      v
              Build Context + Question
                      |
                      v
                  Gemini LLM
                      |
                      v
                Final Answer
```

## 🛠️ Technologies Used

- Python
- Sentence Transformers
- `all-MiniLM-L6-v2`
- MongoDB Atlas
- MongoDB Atlas Vector Search
- Google Gemini
- `google-genai`
- PyMongo
- python-dotenv

## 📁 Project Structure

```text
basic-rag-mongodb-gemini/
│
├── data/
│   └── company_policy.txt
│
├── main.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 📄 Knowledge Base

The current project uses a built-in text document:

```text
data/company_policy.txt
```

The document contains company information such as:

- Annual Leave
- Sick Leave
- Work From Home
- Employee Benefits
- Working Hours

Example:

```text
Employees are entitled to 24 days of annual leave per calendar year.

Employees should submit annual leave requests at least 3 working days
before the requested leave date.

Annual leave requires approval from the employee's manager.
```

# 🔄 How the Project Works

## 1. Load the Document

The application reads the built-in text file.

```python
with open(file_path, "r", encoding="utf-8") as file:
    document = file.read()
```

## 2. Split the Document into Chunks

The document is divided into smaller pieces.

```python
def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks
```

Chunking allows the application to search smaller sections of the document instead of searching the entire document at once.

> Note: The current implementation uses simple character-based chunking. Future versions can use sentence/paragraph-based chunking with overlap.

## 3. Generate Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

from Sentence Transformers.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)
```

The model converts each text chunk into a numerical vector.

The model produces:

```text
384-dimensional embeddings
```

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
[0.021, -0.183, 0.472, ...]
```

These vectors represent the semantic meaning of the text.

## 4. Store Embeddings in MongoDB Atlas

Each chunk is stored along with its embedding.

Example document:

```json
{
    "text": "Employees are entitled to 24 days of annual leave...",
    "embedding": [0.021, -0.183, 0.472],
    "chunk_id": 0
}
```

MongoDB stores:

- Original text
- Embedding vector
- Chunk ID

## 5. MongoDB Vector Search

MongoDB Atlas Vector Search is used to find the chunks that are semantically closest to the user's question.

The project currently uses exact nearest-neighbor search:

```python
"$vectorSearch": {
    "index": "vector_index",
    "path": "embedding",
    "queryVector": query_embedding,
    "exact": True,
    "limit": 3
}
```

The MongoDB Vector Search index uses:

```text
Dimensions: 384
Similarity: cosine
Path: embedding
Type: vector
```

## 6. Convert the User Question into an Embedding

The user's question is converted using the **same embedding model**.

Example:

```text
How many annual leave days do employees get?
```

↓

```python
query_embedding = model.encode(query)
```

↓

```text
384-dimensional vector
```

The query vector is then compared with the document vectors.

## 7. Retrieve Relevant Documents

MongoDB returns the most relevant chunks.

Example:

```text
Chunk ID: 0
Similarity Score: 0.85

Employees are entitled to 24 days of annual leave per calendar year.
```

The similarity score indicates how closely the retrieved chunk matches the query.

## 8. Send Context to Gemini

The retrieved chunks are combined into a context.

```text
CONTEXT:
Employees are entitled to 24 days of annual leave per calendar year.

QUESTION:
How many annual leave days do employees get?
```

This information is sent to Gemini.

## 9. Generate the Final Answer

Gemini generates the final response using the retrieved context.

Example:

```text
Employees are entitled to 24 days of annual leave per calendar year.
```

This is the **Generation** part of RAG.

# 🧠 Components and Their Responsibilities

| Component | Responsibility |
|---|---|
| Text File | Knowledge source |
| Chunking | Splits documents |
| Sentence Transformer | Creates embeddings |
| MongoDB Atlas | Stores documents and vectors |
| Vector Search | Retrieves relevant information |
| Gemini | Generates the final answer |

# 🔑 Environment Variables

Create a `.env` file in the project root.

```env
MONGODB_URI=mongodb+srv://USERNAME:PASSWORD@YOUR_CLUSTER.mongodb.net/
GEMINI_API_KEY=your_gemini_api_key
```

**Do not commit your `.env` file to GitHub.**

# 📦 Installation

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

# 🗄️ MongoDB Atlas Setup

Create a MongoDB Atlas cluster and create:

```text
Database:
rag_database

Collection:
documents
```

Create a Vector Search index named:

```text
vector_index
```

Use the following configuration:

```json
{
  "fields": [
    {
      "numDimensions": 384,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

Make sure the index status becomes:

```text
READY
```

# ▶️ Run the Project

Run:

```bash
python main.py
```

The application will:

1. Load the company policy document
2. Split it into chunks
3. Generate embeddings
4. Store the embeddings in MongoDB
5. Convert the user question into an embedding
6. Perform vector search
7. Retrieve relevant document chunks
8. Send the retrieved context to Gemini
9. Generate the final answer

# 🧪 Example

### User Question

```text
How many annual leave days do employees get?
```

### Vector Search

```text
Chunk ID: 0
Similarity Score: 0.85

Employees are entitled to 24 days of annual leave per calendar year.
```

### Gemini Response

```text
Employees are entitled to 24 days of annual leave per calendar year.
```

# 🔐 Security

Never commit API keys or database credentials.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

# 📚 Learning Objectives

This project demonstrates the fundamental concepts behind RAG:

- Document ingestion
- Text chunking
- Embeddings
- Semantic search
- Vector databases
- Query embeddings
- Context retrieval
- LLM generation
- Retrieval-Augmented Generation


# 🎯 Goal

The goal of this project is to build a RAG system **from the fundamentals upward**, understanding what each component does rather than hiding the implementation behind a high-level framework.

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
    ↓
Semantic Retrieval
    ↓
Context
    ↓
Gemini
    ↓
Answer
```

