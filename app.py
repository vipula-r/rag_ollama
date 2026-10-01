from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# 1. READ PDF
# ==========================================

pdf_file = "sample.pdf"

print("Reading PDF...")

reader = PdfReader(pdf_file)

text = ""

for page_number, page in enumerate(reader.pages):

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print("Number of pages:", len(reader.pages))
print("Total characters extracted:", len(text))


# ==========================================
# 2. CHECK PDF TEXT
# ==========================================

if not text.strip():

    print("\nERROR: No text could be extracted from the PDF.")

    print("The PDF may be scanned/image-based.")

    exit()


# ==========================================
# 3. CREATE CHUNKS
# ==========================================

def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


chunks = create_chunks(text)

print("Number of chunks:", len(chunks))


# ==========================================
# 4. LOAD EMBEDDING MODEL
# ==========================================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==========================================
# 5. GENERATE DOCUMENT EMBEDDINGS
# ==========================================

print("\nGenerating document embeddings...")

embeddings = embedding_model.encode(chunks)

print("Embeddings generated.")


# ==========================================
# 6. CREATE CHROMADB
# ==========================================

print("\nCreating ChromaDB...")

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ==========================================
# 7. CREATE COLLECTION
# ==========================================

collection = client.get_or_create_collection(
    name="documents"
)


# ==========================================
# 8. CREATE IDS
# ==========================================

ids = []

for i in range(len(chunks)):

    ids.append(f"chunk_{i}")


# ==========================================
# 9. STORE DOCUMENTS
# ==========================================

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)


print("\n====================================")
print("PDF stored successfully!")
print("====================================")


# ==========================================
# 10. RAG QUESTION LOOP
# ==========================================

while True:

    question = input(
        "\nAsk a question (type 'exit' to stop): "
    )


    if question.lower() == "exit":

        print("Program stopped.")

        break


    # ======================================
    # 11. CREATE QUESTION EMBEDDING
    # ======================================

    question_embedding = embedding_model.encode(
        [question]
    )


    # ======================================
    # 12. SEARCH CHROMADB
    # ======================================

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=5
    )


    # ======================================
    # 13. GET RELEVANT DOCUMENTS
    # ======================================

    relevant_chunks = results["documents"][0]


    # ======================================
    # 14. COMBINE CHUNKS
    # ======================================

    context = "\n\n".join(relevant_chunks)


    # ======================================
    # 15. CREATE PROMPT
    # ======================================

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using the information provided
in the document context below.

If the question asks for a summary or overview, summarize
the information available in the context.

Do not make up information that is not present in the context.

Document Context:
{context}

User Question:
{question}

Answer:
"""


    # ======================================
    # 16. SEND TO OLLAMA
    # ======================================

    response = ollama.chat(

        model="llama3.2",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )


    # ======================================
    # 17. DISPLAY ANSWER
    # ======================================

    answer = response["message"]["content"]

    print("\n====================================")
    print("ANSWER")
    print("====================================")

    print(answer)

