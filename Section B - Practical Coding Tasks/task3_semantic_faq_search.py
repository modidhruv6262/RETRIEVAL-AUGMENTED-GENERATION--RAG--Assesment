import faiss
from sentence_transformers import SentenceTransformer

faqs = [
    "Standard delivery time ranges from 30 to 45 minutes depending on traffic and kitchen preparation delays.",
    "Orders can only be cancelled within 60 seconds of placement before the restaurant accepts the ticket.",
    "Refund eligibility applies when items are missing, entirely spoiled, or delivered over 60 minutes late.",
    "If an ingredient is out of stock, restaurant staff will call to confirm an item substitution before cooking.",
    "Customer support is accessible 24/7 through the in-app live chat window and toll-free helpline.",
    "The minimum order value for free delivery is 250 rupees across all partner restaurants."
]

model = SentenceTransformer('all-MiniLM-L6-v2')

faq_embeddings = model.encode(faqs).astype("float32")

dimension = faq_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(faq_embeddings)

print(f"Total FAQs indexed: {index.ntotal}")
print(f"Index dimension: {index.d}")
print(f"Index matches FAQ list length: {index.ntotal == len(faqs)}\n")

def search_faq(query, k=2):
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    results = []
    for rank in range(k):
        idx = indices[0][rank]
        dist = distances[0][rank]
        results.append((faqs[idx], float(dist)))
    return results

test_queries = [
    "How do I get my money back for a ruined meal?",
    "Can I swap out French fries for a green salad?"
]

for query in test_queries:
    print(f"Query: '{query}'")
    matches = search_faq(query, k=2)
    for rank, (faq_text, dist) in enumerate(matches, start=1):
        print(f"  {rank}. [L2 Distance: {dist:.4f}] {faq_text}")
    print("-" * 70)
