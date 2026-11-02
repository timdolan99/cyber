# 🎓 MSc Socratic CS Chatbot

An adaptive Socratic tutoring system built for OCR A-level Computer Science using LangGraph, LangChain, and Streamlit. This application dynamically ingests specification documents and guides students through computer science concepts using pedagogical scaffolding and safety-critical guardrails.

## 📂 Directory Structure

```text
msc-socratic-chatbot/
├── .streamlit/
│   └── secrets.toml
├── chroma_db/
├── syllabus/
│   └── GCSE_computer_science_Syllabus.pdf
├── app.py
├── ingest_corpus.py
├── requirements.txt
└── socratic_fsm.py
```

## 🚀 Quick Start Guide
1. Install Dependencies
Install all core libraries using the requirements file:

pip install -r requirements.txt

2. Configure the Gemini API Key
Instead of using terminal environment variables, this project utilizes native Streamlit secrets management.

Create a folder named .streamlit in the project root directory.

Inside that folder, create a file named secrets.toml.

Add your Gemini API key inside it exactly like this:

GOOGLE_API_KEY = "your-actual-gemini-api-key-here"

3. Ingest the Syllabus Document (Optional)
The vector database is already populated under chroma_db/. However, if you wish to re-initialize or update the corpus from the PDF folder, run:

python ingest_corpus.py

4. Launch the Tutor Workspace
Run the Streamlit local application server to begin the learning session:

python -m streamlit run app.py

## 🛠️ Automated Evaluation Suite
This framework features a multi-tiered software verification pipeline to validate conversational boundaries, safety invariants, and pedagogical containment completely without human participant trials.

1. Low-Level Unit Tests (test_socratic_fsm.py)
Validates deterministic state transitions, threshold triggers, and structural FSM invariants using strict assertions:

python -m unittest test_socratic_fsm.py

2. Isolated RAG Proof-of-Concept Baseline (evaluate_rag.py)
Runs an isolated, sandbox retrieval evaluation to establish precision baselines for vector database retrieval and output groundedness before exposing the system to live, multi-turn logic:

python evaluate_rag.py

3. Behavioral Guardrail Testing (evaluate_system.py)
Automates user simulation by stress-testing the LangGraph orchestration layer against 20 distinct synthetic student profiles to check state-routing performance:

python evaluate_system.py

4. Quantitative RAG Quality Matrix (evaluate_system_rag.py)
Benchmarks factual faithfulness (Groundedness) and Context Relevance under adversarial conditions using an LLM-as-a-Judge framework:

python evaluate_system_rag.py

