import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

try:
    with open("refund_policy.txt", "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    with open("Section D - AI-Augmented Learning/refund_policy.txt", "r", encoding="utf-8") as f:
        text = f.read()

chunk_size = 150
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

try:
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)
except Exception:
    llm = None

while True:
    query = input("Ask a question (or type quit): ")
    if query == 'quit':
        break
    
    q_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(q_vec, 2)
    
    retrieved = [chunks[i] for i in indices[0]]
    
    prompt = f"Context:\n{retrieved[0]}\n{retrieved[1]}\n\nQuestion: {query}\nAnswer in plain English strictly using context:"
    
    print("\n--- Retrieved Chunks ---")
    print(retrieved)
    print("\n--- Assembled Prompt ---")
    print(prompt)
    print("\n--- Answer ---")
    if llm:
        try:
            res = llm.invoke(prompt)
            print(res.content + "\n")
        except Exception:
            print("Could not generate response from LLM.\n")
    else:
        print("Simulated answer based on context.\n")
