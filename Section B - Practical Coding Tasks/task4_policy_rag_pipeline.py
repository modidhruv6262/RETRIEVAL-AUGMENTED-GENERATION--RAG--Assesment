import faiss
from sentence_transformers import SentenceTransformer

policy_document = """
Welcome to the QuickBite Food Delivery Service Terms and Operating Policies. Our goal is to provide transparent guidelines for all customers regarding delivery timelines, cancellation terms, and refund eligibility across all partner restaurants and cloud kitchens.

Delivery Windows and Service Level Agreements:
All standard orders are scheduled within an estimated delivery window of 30 to 45 minutes from restaurant confirmation. During severe weather, heavy traffic congestion, or peak festival hours, this window may extend up to 60 minutes to ensure rider safety. If a delivery exceeds 60 minutes without proactive notification from customer support, the customer qualifies for priority compensation or delivery fee reimbursement upon request. Delivery drivers are required to wait at the customer doorstep for a maximum of 10 minutes while attempting phone contact before marking the order as undeliverable due to customer unavailability.

Order Cancellation Terms:
Customers are permitted to cancel an order free of penalty within 60 seconds of initial placement. Once the restaurant partner acknowledges the order and begins food preparation, cancellation is no longer possible through the app interface. If a customer cancels after kitchen preparation begins, a cancellation fee equal to 100 percent of the food subtotal is charged to reimburse merchant ingredient costs and kitchen staff preparation labor.

Refund Rules and Eligibility:
Refunds are issued based on verified customer claims submitted within 120 minutes of delivery. For missing items, customers must submit an in-app report detailing the absent dishes; upon verification, a full refund for the price of the missing items is credited directly to the original payment method within 3 to 5 business days. For incorrect orders where the wrong dish is delivered, customers receive either an immediate free re-delivery or a complete refund including delivery fees. Spoiled, damaged, or unhygienic meals require clear photo evidence submitted through support chat for an immediate full monetary refund.
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

chunks = chunk_text(policy_document, chunk_size=100, overlap=20)
print(f"Total words in policy: {len(policy_document.split())}")
print(f"Total overlapping chunks created: {len(chunks)}\n")

model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks).astype("float32")

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

def retrieve(query, k=3):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return [chunks[idx] for idx in indices[0]]

def build_rag_prompt(query, retrieved_chunks):
    prompt_sections = [
        "System: You are an official QuickBite customer support assistant. Answer the user question strictly using only the provided context blocks below. If the answer cannot be determined from the context, you must explicitly state 'I don't know'. Do not extrapolate or hallucinate outside the given text.",
        "",
        "--- RETRIEVED CONTEXT BLOCKS ---"
    ]
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        prompt_sections.append(f"Context [{rank}]:\n{chunk}\n")
    prompt_sections.append("--- USER QUESTION ---")
    prompt_sections.append(f"Question: {query}\n")
    prompt_sections.append("Instruction: Answer only from the provided context. If the answer is not present, respond with 'I don't know'.")
    return "\n".join(prompt_sections)

test_question = "What is the refund policy for missing items?"
retrieved_blocks = retrieve(test_question, k=3)
assembled_prompt = build_rag_prompt(test_question, retrieved_blocks)

simulated_answer = (
    "According to the policy, if an item is missing, customers must submit an in-app report "
    "within 120 minutes of delivery. Upon verification, a full refund for the price of the "
    "missing items will be credited directly to the original payment method within 3 to 5 business days."
)

print("==================== RETRIEVED CONTEXT BLOCKS ====================")
for idx, blk in enumerate(retrieved_blocks, start=1):
    print(f"Chunk [{idx}]:\n{blk}\n")

print("==================== ASSEMBLED RAG PROMPT ====================")
print(assembled_prompt)
print("\n" + "=" * 60)
print("SIMULATED ANSWER:")
print(simulated_answer)
