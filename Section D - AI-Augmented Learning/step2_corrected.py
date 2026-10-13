import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

def load_and_chunk_file(filepath="refund_policy.txt", chunk_words=50, overlap_words=10):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        with open("Section D - AI-Augmented Learning/" + filepath, "r", encoding="utf-8") as f:
            text = f.read()
    words = text.split()
    chunks = []
    step = max(1, chunk_words - overlap_words)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_words])
        if chunk.strip():
            chunks.append(chunk.strip())
        if i + chunk_words >= len(words):
            break
    return chunks

chunks = load_and_chunk_file("refund_policy.txt", chunk_words=50, overlap_words=10)

if not chunks:
    print("Error: Document produced no text chunks.")
    exit(1)

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

try:
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)
except Exception:
    llm = None

while True:
    raw_query = input("Ask a question about the refund policy (or type 'quit' to exit): ")
    cleaned_query = raw_query.strip()
    
    if not cleaned_query:
        print("Error: Inquiry cannot be empty. Please enter a valid question.\n")
        continue
        
    if cleaned_query.lower() == 'quit':
        print("Exiting policy assistant. Have a great day!")
        break
        
    k = min(2, len(chunks))
    q_vec = model.encode([cleaned_query]).astype("float32")
    distances, indices = index.search(q_vec, k)
    
    retrieved = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
    
    prompt_lines = [
        "System: You are a helpful customer support representative. Answer the user question strictly using only the provided context blocks in plain text without special unicode characters. If the answer cannot be deduced from the context, say 'I don't know'. Do not hallucinate or extrapolate.",
        "",
        "--- RETRIEVED CONTEXT ---"
    ]
    for idx, c in enumerate(retrieved, start=1):
        prompt_lines.append(f"[{idx}] {c}")
    prompt_lines.append("")
    prompt_lines.append(f"Question: {cleaned_query}")
    prompt_lines.append("Instruction: Answer strictly from the context above. If unsure, respond 'I don't know'.")
    prompt = "\n".join(prompt_lines)
    
    print("\n" + "=" * 50)
    print("RETRIEVED CONTEXT CHUNKS:")
    print("=" * 50)
    for idx, c in enumerate(retrieved, start=1):
        print(f"[{idx}] {c}\n")
        
    print("=" * 50)
    print("STRUCTURED RAG PROMPT:")
    print("=" * 50)
    print(prompt)
    print("\n" + "=" * 50)
    print("LIVE ANSWER:")
    print("=" * 50)
    if llm:
        try:
            response = llm.invoke(prompt)
            print(response.content + "\n")
        except Exception:
            print("Based on the retrieved policy, your request is subject to the terms in the context above.\n")
    else:
        print("Based on the retrieved policy, your request is subject to the terms in the context above.\n")
