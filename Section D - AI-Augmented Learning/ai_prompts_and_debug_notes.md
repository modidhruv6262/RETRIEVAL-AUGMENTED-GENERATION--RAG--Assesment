# Section D — AI-Augmented Learning

---

## 1. Exact Prompt Given to the AI Tool

```text
Write a Python program that reads a plain-text food delivery refund policy file from disk ('refund_policy.txt'), splits it into fixed-size chunks, generates embeddings using sentence-transformers, and stores them in a FAISS index.

Accept a user question typed at the command line, retrieve the top-2 most relevant chunks from the index, and construct a structured RAG prompt using those chunks as context.

Print the retrieved chunks, the assembled prompt, and a simulated placeholder answer.

Run the program in a loop so the user can ask multiple questions without restarting the script, and exit cleanly when the user types 'quit'.
```

---

## 2. Code Comparison

* **AI Original Code:** Located at `step1_ai_original.py`
* **Corrected Version:** Located at `step2_corrected.py`

---

## 3. Debugging Notes: Identified Limitations and Changes Made

The AI's initial solution contained four critical defects:
1. **Mid-Word / Mid-Sentence Chunking:** The AI used arbitrary character slicing (`text[i:i+150]`), which severed words in half and severed sentences across chunk boundaries; this was fixed by implementing word-boundary slicing with a 10-word sliding overlap.
2. **Fixed `k` Assumption Index Crash:** The AI hardcoded `k=2` and indexed `retrieved[0]` and `retrieved[1]`, which throws an `IndexError` whenever the document produces fewer than 2 chunks; this was resolved by dynamically bounding `k = min(2, len(chunks))`.
3. **Fragile Loop Exit & Unsanitized Input:** The AI checked `if query == 'quit':` without whitespace stripping or case normalization, causing trailing spaces or `'Quit'` to keep running, and allowed empty questions; this was fixed with `.strip().lower()` and explicit non-empty input validation.
4. **Weak Prompt Guardrails:** The AI's assembled prompt lacked anti-hallucination instructions; we added explicit system instructions commanding the model to answer strictly from context and say *"I don't know"* when information is missing.
