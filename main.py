from pathlib import Path
import os
import time

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

mongodb_uri = os.getenv("MONGODB_URI")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not mongodb_uri:
    raise ValueError("MONGODB_URI is not set in the .env file")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the .env file")


# ============================================================
# 2. CONNECT TO MONGODB ATLAS
# ============================================================

print("Connecting to MongoDB Atlas...")

client = MongoClient(mongodb_uri)

# Test connection
client.admin.command("ping")

print("Connected to MongoDB Atlas")


db = client["rag_database"]
collection = db["documents"]


# ============================================================
# 3. LOAD DOCUMENT
# ============================================================

file_path = Path("data/company_policy.txt")

print("\nLoading document...")

with open(file_path, "r", encoding="utf-8") as file:
    document = file.read()

print(f"Document loaded: {file_path}")


# ============================================================
# 4. SPLIT DOCUMENT INTO CHUNKS
# ============================================================

def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


chunks = create_chunks(document)

print(f"\nNumber of chunks: {len(chunks)}")


for i, chunk in enumerate(chunks):

    print(f"\n--- Chunk {i} ---")
    print(chunk)


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")


# ============================================================
# 6. CREATE EMBEDDINGS
# ============================================================

print("\nCreating embeddings...")

embeddings = model.encode(chunks)

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")


# ============================================================
# 7. PREPARE DOCUMENTS FOR MONGODB
# ============================================================

documents = []

for i, (chunk, embedding) in enumerate(
    zip(chunks, embeddings)
):

    documents.append({

        "text": chunk,

        "embedding": [
            float(value)
            for value in embedding
        ],

        "chunk_id": i
    })


# ============================================================
# 8. STORE DOCUMENTS IN MONGODB
# ============================================================

print("\nClearing old documents...")

collection.delete_many({})


print("Inserting documents into MongoDB...")

collection.insert_many(documents)

print(f"Inserted {len(documents)} documents into MongoDB")


# ============================================================
# 9. WAIT FOR VECTOR INDEX
# ============================================================

print("\nWaiting for MongoDB Vector Search index...")

time.sleep(10)

print("Continuing...")


# ============================================================
# 10. INITIALIZE GEMINI
# ============================================================

print("\nInitializing Gemini...")

gemini_client = genai.Client(
    api_key=gemini_api_key
)

print("Gemini initialized")


# ============================================================
# 11. VECTOR SEARCH FUNCTION
# ============================================================

def search_similar_documents(query, limit=3):

    print("\nCreating query embedding...")

    query_embedding = model.encode(query)

    query_embedding = [
        float(value)
        for value in query_embedding
    ]

    print(
        f"Query embedding dimensions: "
        f"{len(query_embedding)}"
    )


    # MongoDB Vector Search pipeline

    pipeline = [

        {
            "$vectorSearch": {

                "index": "vector_index",

                "path": "embedding",

                "queryVector": query_embedding,

                # Exact nearest-neighbor search
                "exact": True,

                "limit": limit
            }
        },

        {
            "$project": {

                "_id": 0,

                "chunk_id": 1,

                "text": 1,

                "score": {
                    "$meta": "vectorSearchScore"
                }
            }
        }
    ]


    print("\nExecuting MongoDB Vector Search...")

    results = collection.aggregate(pipeline)

    return list(results)


# ============================================================
# 12. GENERATE ANSWER USING GEMINI
# ============================================================

def generate_answer(query, retrieved_documents):

    # Combine retrieved chunks into one context

    context = "\n\n".join(

        document["text"]

        for document in retrieved_documents
    )


    # Prompt given to Gemini

    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't have enough information in the provided documents."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""


    print("\nSending context to Gemini...")

    response = gemini_client.models.generate_content(

        model="gemini-2.5-flash-lite",

        contents=prompt
    )


    return response.text


# ============================================================
# 13. USER QUESTION
# ============================================================

query = "How many hours should employee work per day ?"

print("\n" + "=" * 60)
print("USER QUESTION")
print("=" * 60)

print(query)


# ============================================================
# 14. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

results = search_similar_documents(
    query,
    limit=3
)


# ============================================================
# 15. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print("\n" + "=" * 60)
print("SEARCH RESULTS")
print("=" * 60)


if not results:

    print("No relevant documents found.")

else:

    for result in results:

        print("\n" + "-" * 60)

        print(
            f"Chunk ID: {result['chunk_id']}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']}"
        )

        print("\nText:")

        print(result["text"])


# ============================================================
# 16. GENERATE FINAL ANSWER
# ============================================================

if results:

    answer = generate_answer(
        query,
        results
    )


    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)


# ============================================================
# 17. CLOSE MONGODB CONNECTION
# ============================================================

client.close()

print("\nMongoDB connection closed.")