# AI-Powered Security Awareness Chatbot

An LLM-powered chatbot designed to help students learn cybersecurity concepts by answering cybersecurity questions and generating quizzes. The chatbot uses **Retrieval-Augmented Generation (RAG)** to ground its responses in course materials.

## Project Structure

```text
AI-Powered-Security-Awareness-Chatbot/
│
├── data/
│   └── knowledge_base/
│       └── course materials (.txt)
│
├── vector_db/
│
├── ingestion.py
├── retrieval.py
├── retrieval_evaluation.py
├── llm_answering.py
├── quiz_generation.py
├── streamlit_app.py
│
├── requirements.txt
└── README.md
```

### Main Components

* **`data_ingestion.py`** – chunks course material, creates embeddings, and stores them in the vector store.
* **`retrieval.py`** – retrieves relevant knowledge from the vector database.
* **`llm_answering.py`** – generates grounded answers using the retrieved information.
* **`quiz_generation.py`** – generates and evaluates cybersecurity quizzes.
* **`streamlit_app.py`** – provides the chatbot, quiz, and document-upload interface.

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Add the required API key as an environment variable before running the application.

Run the application with:

```bash
streamlit run streamlit_app.py
```

Course materials can be uploaded through the **Add Document** interface.

## Important Instructions

* Use the agreed project structure and file ownership.
* Keep course materials inside the knowledge-base directory.
* Test each component independently before integration.
* Do not change interfaces between components without informing the responsible team member.
* Keep the retrieval, LLM, and quiz components separate so they can be tested independently.
* Changes to the vector database or chunking method should be communicated to the team.
* Each member should document and test the component they are responsible for.

## Work Distribution

| Person       | Main Responsibility                                                        | Report Section                           |
| ------------ | -------------------------------------------------------------------------- | ---------------------------------------- |
| **Person 1** | Data ingestion, chunking, embeddings, vector store                         | Introduction & Motivation                |
| **Person 2** | Retrieval logic and retrieval-quality evaluation                           | System Architecture                      |
| **Person 3** | LLM integration, prompt design, answer generation and groundedness testing | LLM Integration & Prompt Design          |
| **Person 4** | Quiz generation, quiz evaluation and Streamlit UI                          | Quiz Generation, UI & Evaluation Summary |

### Person 1

Responsible for the knowledge base, document ingestion, chunking, embeddings, and vector-store preparation.

### Person 2

Responsible for retrieval, retrieval accuracy testing, and evaluation using test questions and known source chunks.

### Person 3

Responsible for prompt engineering, LLM answer generation, and testing whether responses are properly grounded in retrieved information.

### Person 4

Responsible for quiz generation, quiz quality review, and the Streamlit user interface.

The original project division also specifies that each member writes the report section associated with their implementation, with one member compiling the final report.

## Development Flow

```text
Course Materials
       ↓
Data Ingestion
       ↓
Vector Store
       ↓
Retrieval
       ↓
LLM Answer / Quiz Generation
       ↓
Streamlit Interface
```

## Evaluation

The project evaluates:

* **Retrieval accuracy** – whether relevant source chunks are retrieved.
* **LLM groundedness** – whether answers are supported by retrieved information.
* **Quiz quality** – relevance, correct answers, and plausible distractors.
* **Overall system performance**.

## Team Integration

Each component should be developed and tested independently before being integrated into the final chatbot.

The final demonstration should show the complete flow:

```text
Ingestion → Retrieval → Chat Answer → Quiz
```

All team members should be able to explain and demonstrate their assigned component.
