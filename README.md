# 📚 Prompt Engineering & Retrieval-Augmented Generation (RAG)
### Assessment Code: `M18-A1` | Program: Data Science

---

## 📋 Overview
This repository contains the complete assessment solution for **Modules 17–18: Prompt Engineering & Retrieval-Augmented Generation (RAG)**. The assessment is divided into four modular sections:

- **Section A — Concept Application:** In-depth technical justifications and case studies on prompt clarity, few-shot classification, Chain-of-Thought (CoT), RAG vs Fine-Tuning, chunking strategies, and vector store architectures (FAISS vs ChromaDB).
- **Section B — Practical Coding Tasks:** Four standalone scripts demonstrating structured prompt building, few-shot prompt construction, semantic FAQ search using FAISS, and an end-to-end policy RAG pipeline.
- **Section C — Mini Capstone Project:** An interactive, menu-driven Food Delivery Help Desk console application with FAISS vector indexing over a 550+ word policy document, few-shot complaint classification, input validation, and session logging.
- **Section D — AI-Augmented Learning:** Step 1 AI-generated baseline code vs. Step 2 human-debugged and hardened RAG solution with detailed comparative documentation.

---

## 🗂️ Project Structure

```
RETRIEVAL-AUGMENTED-GENERATION-(RAG)-Assesment/
│
├── .env                                       # Secure API key storage
├── .gitignore                                 # Git rules ignoring .env, env/, cache
├── README.md                                  # Comprehensive documentation
│
├── Section A - Concept Application/
│   └── Section_A_Answers.md                   # Detailed solutions for Scenarios 1 to 6
│
├── Section B - Practical Coding Tasks/
│   ├── task1_structured_prompt_builder.py    # Role-based prompt builder with validation
│   ├── task2_few_shot_classifier.py          # Few-shot complaint classification prompt generator
│   ├── task3_semantic_faq_search.py          # Semantic FAQ retrieval with SentenceTransformers & FAISS
│   └── task4_policy_rag_pipeline.py          # End-to-end policy document RAG pipeline
│
├── Section C - Mini Capstone Project/
│   ├── food_delivery_helpdesk.py             # Interactive console chatbot with RAG & few-shot modes
│   ├── policy_document.txt                   # 550+ word comprehensive food delivery policy
│   └── session_log.txt                       # Automated timestamped session interaction log
│
└── Section D - AI-Augmented Learning/
    ├── refund_policy.txt                     # Plain-text refund policy knowledge base
    ├── step1_ai_original.py                  # Initial naive AI-generated RAG implementation
    ├── step2_corrected.py                    # Hardened, human-debugged RAG script
    └── ai_prompts_and_debug_notes.md         # Exact AI prompts & 4-point debugging rationale
```

---

## ⚙️ Setup & Activation

### 1. Activate the Virtual Environment
```powershell
cd "C:\Users\DHRUV MODI\Documents\TOPS\Assessment\RETRIEVAL-AUGMENTED-GENERATION-(RAG)\RETRIEVAL-AUGMENTED-GENERATION-(RAG)-Assesment"
.\env\Scripts\activate
```

### 2. Run Section B Tasks
```powershell
python "Section B - Practical Coding Tasks\task1_structured_prompt_builder.py"
python "Section B - Practical Coding Tasks\task2_few_shot_classifier.py"
python "Section B - Practical Coding Tasks\task3_semantic_faq_search.py"
python "Section B - Practical Coding Tasks\task4_policy_rag_pipeline.py"
```

### 3. Run Section C Mini Capstone Project
```powershell
python "Section C - Mini Capstone Project\food_delivery_helpdesk.py"
```

### 4. Run Section D AI-Augmented Learning
```powershell
cd "Section D - AI-Augmented Learning"
python step2_corrected.py
```
