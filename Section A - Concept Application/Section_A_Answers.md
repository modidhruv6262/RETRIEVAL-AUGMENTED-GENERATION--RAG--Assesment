# Section A — Concept Application

---

## Scenario 1: Prompt Clarity & Decision Rules in Customer Support Chatbots

### 1. Distinguishing Clear Prompts from Unclear Prompts
In conversational AI systems for domain-specific operations (such as food delivery support), a **clear prompt** establishes deterministic boundaries:
- **Explicit Role & Persona:** Defines precisely who the agent is (e.g., a tier-1 customer support specialist).
- **Hard Decision Logic:** Outlines strict conditional branching (IF order is marked delivered and item reported missing within 2 hours, THEN refund; ELSE escalate).
- **Strict Output Schema:** Mandates fixed formatting, required fields, and token/word limits (e.g., JSON output or responses under 80 words).
- **Grounding & Guardrails:** Forbids hallucinated approvals or policy exceptions.

Conversely, an **unclear prompt** relies on ambiguous natural language like *"Help the customer with their order and be polite"*. Without strict decision criteria, the LLM falls back on probabilistic next-token generation, leading to stochastic behavior—approving unverified refunds in one turn and rejecting valid customer claims in another.

### 2. Two Specific Prompt Modifications & Justifications

#### Change 1: Implementation of a Strict Verification Decision Tree (Step-by-Step Guardrail)
* **Prompt Addition:**
  > "Before issuing any refund or rejection, verify the following 3 criteria in order:
  > 1. Is the order status verified as DELIVERED?
  > 2. Is the complaint timestamp within 120 minutes of delivery?
  > 3. Does the issue type fall into ['missing item', 'wrong item', 'spoilage']?
  > If all 3 are YES, approve refund for the affected item only. If order status is pending or not verified, request verification and DO NOT issue a refund."
* **Justification:** Eliminates subjective judgment. The LLM must explicitly evaluate verifiable state variables before choosing an outcome, preventing premature financial payouts on unverified orders.

#### Change 2: Structured Output Enforcement with Strict Word Limits
* **Prompt Addition:**
  > "Your response must follow this exact format:
  > [Verification Status]: Verified / Unverified
  > [Decision]: Approved / Escalated / Rejected
  > [Customer Message]: Professional explanation not exceeding 60 words."
* **Justification:** Constraining output format prevents verbose filler and forces the LLM to commit to a binary decision before generating the customer-facing message, ensuring uniformity across thousands of support interactions.

---

## Scenario 2: Zero-Shot vs Few-Shot Prompting for Complaint Classification

### 1. Comparison of Strategies
* **Zero-Shot Prompting:** The LLM is provided solely with category names and definitions without any sample demonstrations. While computationally minimal and consuming few context tokens, zero-shot classification often struggles with nuanced semantic edge cases (e.g., classifying *"the pizza arrived completely cold and crushed"* as either 'Late Delivery' or 'Poor Quality').
* **Few-Shot Prompting:** The prompt includes a curated set of input-output demonstration pairs. This conditions the model's in-context learning to mirror the exact taxonomy, classification threshold, and output formatting expected.

### 2. Strategy Justification
**Selected Approach: Few-Shot Prompting (with 4–8 balanced exemplars).**
* Given the context window limitation (which cannot fit all 200 training examples), few-shot prompting represents the optimal trade-off. 
* By selecting 1 to 2 prototypical exemplars per class (total 4–8 examples), the prompt consumes less than 350 tokens while establishing unambiguous decision boundaries between overlapping categories (such as distinguishing 'Missing Item' from 'Wrong Item').

### 3. Factors Determining Few-Shot Example Count
1. **Context Window & Latency Budget:** Every exemplar increases prompt tokens, directly raising inference latency and API cost.
2. **Number of Target Classes:** At minimum, 1 balanced example per class is required to prevent majority-class bias.
3. **Class Ambiguity / Semantic Overlap:** Subtle boundary cases require counter-examples (e.g., showing that an order delivered 45 minutes late is 'Late Delivery' even if the customer mentions cold food).
4. **Model Parameter Scale:** Smaller models require 4–8 examples to reliably align with the schema, whereas larger models generalize with 2–4 examples.

---

## Scenario 3: Chain-of-Thought (CoT) Prompting for Delivery Stop Optimization

### 1. Why Chain-of-Thought Improves Output Quality
Direct prompting forces the model to predict the final sequence in a single forward pass without intermediate computation tokens. Combinatorial problems (like traveling salesperson variations involving 5 stops, variable traffic delays, and priority levels) exceed the single-step predictive capacity of autoregressive transformers.

Chain-of-Thought (CoT) prompting allows the model to generate intermediate reasoning tokens, breaking the complex optimization down into manageable sub-evaluations:
1. Sorting orders by priority level (high-priority medical or express orders first).
2. Calculating estimated transit time per stop incorporating traffic delays.
3. Evaluating trade-offs between geographical proximity and SLA deadlines.

### 2. Restructured CoT Prompt Example
```text
System: You are an AI operations dispatcher. Solve delivery route optimization problems step-by-step.
User: Optimize the stop sequence for the following 5 orders:
- Order A: 2 km away, Low Priority, 5 min delay
- Order B: 6 km away, High Priority (SLA expiring in 15 min), 10 min delay
- Order C: 3 km away, Medium Priority, 2 min delay
- Order D: 7 km away, Low Priority, 12 min delay
- Order E: 4 km away, High Priority (SLA expiring in 25 min), 4 min delay

Before stating the final sequence, show your reasoning:
Step 1: Identify all High Priority orders and check their urgency deadlines.
Step 2: Compare travel times and traffic delays to determine the fastest order to fulfill critical SLAs.
Step 3: Cluster remaining medium and low priority stops along the return route to minimize backtrack distance.
Step 4: Output the final optimal stop sequence.
```

### 3. Limitation of Chain-of-Thought Prompting
* **No Guarantee of Mathematical Optimality (Hallucinated Arithmetic):** While CoT forces logical structure, LLMs are still probabilistic token predictors rather than exact graph solvers. For complex constraints or floating-point delay math, the model can logically describe a path while incorrectly computing total elapsed minutes.

---

## Scenario 4: RAG vs Fine-Tuning for a 150-Page Quarterly Updated Policy Handbook

### 1. Justification of RAG Over Fine-Tuning
1. **Seamless Handling of Quarterly Updates:**
   - In RAG, when policy terms change each quarter, updating the system requires only deleting the obsolete chunk embeddings from the vector database and inserting newly embedded PDF chunks. This indexing process takes seconds to minutes with zero training downtime.
   - In contrast, fine-tuning requires compiling a supervised dataset of question-answer pairs, validating data hygiene, renting GPU compute, executing training epochs, and running safety regression checks every three months.
2. **Fact Grounding & Verifiable Citations:**
   - Delivery partners querying payout dispute policies require exact clause numbers and legal phrasing. RAG retrieves the verbatim handbook text and provides it as context, allowing the LLM to cite exact section headers. Fine-tuned models encode information into internal weights, making them prone to subtle hallucinations without source verifiability.

### 2. Scenario Where Fine-Tuning is the Better Choice
Fine-tuning is superior when the goal is **teaching the model a specialized domain tone, jargon, or unique grammatical syntax** rather than factual lookup—for example, teaching an LLM to automatically parse unstructured multilingual delivery rider shorthand/slang into structured JSON tickets, or fine-tuning a small 3B model to run locally on low-cost edge servers with zero external database dependencies.

---

## Scenario 5: Chunk Size and Overlap Strategy for Restaurant Menu PDFs

### 1. Role of Chunk Size and Chunk Overlap
* **Chunk Size:** Determines the semantic window volume (e.g., 100 words vs 400 words). If too small, interrelated entities (dish title, ingredients, price) are fragmented across boundaries. If too large, irrelevant noise dilutes the dense vector embedding, lowering retrieval precision.
* **Chunk Overlap:** Duplicates boundary tokens across successive chunks (e.g., 20% overlap). It ensures that concepts spanning adjacent blocks remain unified in at least one chunk embedding.

### 2. Adjusting Parameters for Restaurant Menus
* **Problem:** In an 8-page menu PDF, prices frequently appear on lines or paragraphs below dish descriptions, causing them to fall into separate chunks.
* **Adjustment:**
  1. **Increase Chunk Size:** Expand from short ~100-word snippets to **300–400 words** (or chunk hierarchically by menu section / item block using markdown/delimiter parsing).
  2. **Increase Chunk Overlap:** Increase overlap to **50–80 words (approx. 20–25%)**.
* **Result:** Every dish name and its corresponding pricing/addon description are guaranteed to co-occur within a single coherent retrieval chunk.

### 3. Introduced Trade-Off
* **Context Dilution & Increased Token Usage:** Larger chunk sizes consume significantly more LLM context window tokens per query. If retrieving top-k=3 chunks of 400 words each (1200 words), inference cost and latency rise, and the model may ingest extraneous menu items that distract from the user's specific target dish.

---

## Scenario 6: Vector Store Evaluation — FAISS vs ChromaDB

### 1. Comparison of FAISS and ChromaDB
| Feature | FAISS (Facebook AI Similarity Search) | ChromaDB |
| :--- | :--- | :--- |
| **Architecture** | Low-level C++ vector indexing and search library with Python bindings. In-memory matrix indexing. | Full-featured embedded or client-server document-oriented vector database. |
| **Metadata Handling** | Primitive. Maps vectors strictly to integer IDs (`int64`). Requires maintaining separate external dictionary to map IDs to text/metadata. | Native. Stores raw text, document IDs, and rich metadata dictionaries (`{"status": "resolved"}`) alongside vectors. |
| **CRUD Operations** | Vector deletion is cumbersome and requires rebuilding or using specific indices like `IndexIDMap2`. | Full native CRUD. Simple `collection.delete(ids=[...])` or `collection.update(...)` commands. |
| **Best Fit Scenario** | Ultra-high performance, large-scale (millions of vectors) similarity search where low search latency is critical and index is mostly static. | Dynamic applications requiring rapid document ingestion, frequent record updates, metadata filtering, and ease of prototyping. |

### 2. Does Switching to ChromaDB Solve the Outdated-Record Problem?
**Yes, structurally.** In FAISS, removing outdated records is difficult because traditional `IndexFlatL2` lacks native document deletion; deleted records remain indexed unless manually filtered or re-indexed from scratch. ChromaDB supports direct document deletion via `collection.delete(ids=[complaint_id])` as soon as a ticket is resolved.

### 3. Mandatory Step Required Regardless of Vector Store
**Active Data Lifecycle & State Synchronization:** Regardless of whether FAISS or ChromaDB is used, the engineering team must implement an **event-driven hook or automated cleanup pipeline**. Whenever a complaint status is updated to `RESOLVED` in the primary SQL/transactional database, an automated event must trigger deletion or mark metadata flag `is_active: false` in the vector index. A vector store cannot automatically divine business state changes without upstream synchronization.
