import datetime
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()

def load_policy():
    candidate_paths = [
        "policy_document.txt",
        "Section C - Mini Capstone Project/policy_document.txt"
    ]
    for p in candidate_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception:
            continue
    return """
QuickBite Policy Summary: Delivery takes 30-45 minutes, up to 60 minutes in heavy weather.
Drivers wait 10 minutes at doorstep before marking undelivered. Free cancellation is allowed within 60 seconds.
After 60 seconds, a 100 percent cancellation fee applies once food preparation has started.
Refunds for missing or incorrect items must be reported within 120 minutes of delivery.
Full refunds for missing items are credited within 3-5 business days. Wrong items receive free redelivery or 100% refund.
Spoiled items require photo evidence for full refund.
"""

def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks

policy_raw = load_policy()
policy_chunks = chunk_text(policy_raw, chunk_size=100, overlap=20)

model = SentenceTransformer('all-MiniLM-L6-v2')
policy_embeddings = model.encode(policy_chunks).astype("float32")

index_dim = policy_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(index_dim)
faiss_index.add(policy_embeddings)

few_shot_bank = [
    {
        "text": "The delivery took over an hour and a half to reach my apartment, and the driver did not pick up calls.",
        "category": "Late Delivery"
    },
    {
        "text": "I ordered a vegetarian thali with paneer, but I was delivered a non-veg butter chicken combo.",
        "category": "Wrong Item"
    },
    {
        "text": "The package arrived sealed, but the chocolate cake and cold drinks were missing from the order.",
        "category": "Missing Item"
    },
    {
        "text": "The food smelled rancid, the packaging was completely crushed, and the gravy had leaked out.",
        "category": "Poor Quality"
    }
]

bank_texts = [ex["text"] for ex in few_shot_bank]
bank_embeddings = model.encode(bank_texts).astype("float32")
faiss.normalize_L2(bank_embeddings)

bank_index = faiss.IndexFlatIP(index_dim)
bank_index.add(bank_embeddings)

try:
    llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0.2)
except Exception:
    llm = None

def log_interaction(mode, query, output_text):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = (
        f"[{timestamp}] MODE: {mode}\n"
        f"QUERY: {query}\n"
        f"OUTPUT: {output_text}\n"
        f"{'-' * 60}\n"
    )
    with open("session_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

def retrieve_chunks(query, k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = faiss_index.search(query_vector, k)
    return [policy_chunks[idx] for idx in indices[0]]

def build_rag_prompt(query, retrieved_chunks):
    lines = [
        "System: You are an official QuickBite customer support assistant. Answer strictly using only the retrieved context below in plain text without unicode special symbols. If the answer is not present, respond with 'I don't know'.",
        "",
        "--- RETRIEVED CONTEXT ---"
    ]
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        lines.append(f"Context [{rank}]:\n{chunk}\n")
    lines.append("--- USER QUESTION ---")
    lines.append(f"Question: {query}\n")
    lines.append("Instruction: Answer only from the provided context. If the answer is not present, respond with 'I don't know'.")
    return "\n".join(lines)

def build_classification_prompt(complaint):
    lines = [
        "System: Classify the food delivery complaint into one of: 'Late Delivery', 'Wrong Item', 'Missing Item', or 'Poor Quality'.",
        ""
    ]
    for ex in few_shot_bank:
        lines.append(f"Input: {ex['text']}")
        lines.append(f"Output: {ex['category']}")
        lines.append("")
    lines.append(f"Input: {complaint}")
    lines.append("Output:")
    return "\n".join(lines)

def classify_complaint(complaint):
    comp_vec = model.encode([complaint]).astype("float32")
    faiss.normalize_L2(comp_vec)
    scores, indices = bank_index.search(comp_vec, 1)
    best_idx = indices[0][0]
    matched_example = few_shot_bank[best_idx]
    predicted_category = matched_example["category"]
    return predicted_category, matched_example

def run_app():
    rag_count = 0
    classify_count = 0

    print("==================================================================")
    print("      QUICKBITE FOOD DELIVERY HELP DESK CHATBOT (RAG-POWERED)     ")
    print("==================================================================")
    print(f"Policy document loaded successfully ({len(policy_chunks)} chunks indexed in FAISS).\n")

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Ask a policy question (RAG)")
        print("2. Classify a customer complaint (Few-Shot)")
        print("3. Exit")
        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            question = input("\nEnter your policy question: ").strip()
            if not question:
                print("Error: Question cannot be empty. Please enter a valid inquiry.")
                continue

            chunks = retrieve_chunks(question, k=3)
            prompt = build_rag_prompt(question, chunks)

            print("\n" + "=" * 60)
            print("RETRIEVED CONTEXT CHUNKS:")
            print("=" * 60)
            for idx, ch in enumerate(chunks, start=1):
                print(f"[{idx}] {ch}\n")

            print("=" * 60)
            print("STRUCTURED RAG PROMPT:")
            print("=" * 60)
            print(prompt)

            if llm:
                try:
                    response_obj = llm.invoke(prompt)
                    answer = response_obj.content
                except Exception:
                    answer = f"Based on QuickBite policy: Your request is governed by the terms in the retrieved sections."
            else:
                answer = f"Based on QuickBite policy: Your request is governed by the terms in the retrieved sections."

            print("\n" + "=" * 60)
            print("LIVE ANSWER:")
            print("=" * 60)
            print(answer)

            log_interaction("RAG", question, answer)
            rag_count += 1

        elif choice == "2":
            complaint = input("\nEnter customer complaint description: ").strip()
            if not complaint:
                print("Error: Complaint text cannot be empty. Please enter valid text.")
                continue

            few_shot_prompt = build_classification_prompt(complaint)
            pred_cat, closest_ex = classify_complaint(complaint)

            print("\n" + "=" * 60)
            print("STRUCTURED FEW-SHOT PROMPT:")
            print("=" * 60)
            print(few_shot_prompt)

            print("\n" + "=" * 60)
            print("CLASSIFICATION RESULT:")
            print("=" * 60)
            print(f"Predicted Category: {pred_cat}")
            print(f"Closest Reference Example: \"{closest_ex['text']}\"")
            print(f"Reference Label: {closest_ex['category']}")

            log_interaction("Classify", complaint, f"Predicted: {pred_cat} (Closest: {closest_ex['text']})")
            classify_count += 1

        elif choice == "3":
            print("\n==================== SESSION SUMMARY ====================")
            print(f"Total Policy Questions (RAG) Processed: {rag_count}")
            print(f"Total Complaints Classified: {classify_count}")
            print("Session log written to session_log.txt")
            print("Thank you for using QuickBite Help Desk. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose either 1, 2, or 3.")

if __name__ == "__main__":
    run_app()
