# Intelligent Hybrid Analysis Agent

An intelligent data analysis and document question-answering system that combines **structured data analytics, Retrieval-Augmented Generation (RAG), and hybrid reasoning** using LangGraph.

The system can understand a user's question, determine the appropriate reasoning path, analyze structured datasets, retrieve information from supporting documents, or combine both sources to produce a final answer.

---

## Overview

Traditional data-analysis applications and document-based RAG systems generally operate independently.

The **Intelligent Hybrid Analysis Agent** combines both capabilities into a single agentic workflow.

A user can:

* Ask analytical questions about a CSV dataset
* Ask questions about supporting PDF/TXT documentation
* Ask questions that require both dataset analysis and document knowledge
* Upload a new CSV dataset dynamically
* Upload additional documents to the knowledge base
* View dataset statistics and previews
* See which reasoning route was selected
* View document sources used for RAG responses

The application uses **LangGraph** to orchestrate specialized agents and route each query to the most appropriate workflow.

---

## Key Features

### 1. Intelligent Query Routing

The system classifies incoming questions into three routes:

```text
ANALYTICS
RAG
HYBRID
```

The router determines whether the question should be answered using structured dataset analysis, document retrieval, or both.

---

### 2. Analytics Agent

The analytics workflow works directly with the currently loaded Pandas DataFrame.

It can perform operations such as:

* Row counting
* Unique-value analysis
* Value-frequency analysis
* Numerical aggregation
* Dataset filtering
* Dataset comparisons
* Other structured analytical operations supported by the analytics tools

The analytics workflow is dynamically created around the currently loaded dataset.

---

### 3. Retrieval-Augmented Generation

The RAG workflow allows the system to answer questions using information stored in the document knowledge base.

The document pipeline includes:

```text
Document
   ↓
Document Loading
   ↓
Text Splitting
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Similarity Retrieval
   ↓
LLM Response
```

The application supports PDF and TXT documents through the Streamlit interface.

RAG responses can also expose the document sources used to generate the answer.

---

### 4. Hybrid Reasoning

The hybrid workflow combines both sources of information.

```text
                  User Question
                       │
                       ▼
                    Router
                       │
                       ▼
                    HYBRID
                   /      \
                  ▼        ▼
             Analytics     RAG
                  │        │
                  └────┬───┘
                       ▼
                   Synthesis
                       │
                       ▼
                  Final Answer
```

The system first obtains:

* An analytical result from the dataset
* A retrieved knowledge result from the document store

A hybrid agent then synthesizes both results into a single response.

This allows questions that require both **data-driven evidence and contextual documentation** to be answered in one workflow.

---

## Architecture

The high-level architecture is:

```text
                         ┌─────────────────────┐
                         │       User          │
                         │    Question         │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   LangGraph Router  │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
          │  Analytics  │    │     RAG     │    │   Hybrid    │
          │    Route    │    │    Route    │    │    Route    │
          └──────┬──────┘    └──────┬──────┘    └──────┬──────┘
                 │                  │                  │
                 ▼                  ▼                  ▼
          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
          │   Pandas    │    │  ChromaDB   │    │  Analytics  │
          │   Dataset   │    │ Vector Store│    │      +      │
          └──────┬──────┘    └──────┬──────┘    │     RAG     │
                 │                  │            └──────┬──────┘
                 │                  │                   │
                 │                  │                   ▼
                 │                  │            ┌─────────────┐
                 │                  │            │  Synthesis  │
                 │                  │            │    Agent    │
                 │                  │            └──────┬──────┘
                 │                  │                   │
                 └──────────────────┴───────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │ Final Answer  │
                            └───────────────┘
```

---

## Application Workflow

### Dataset workflow

A default dataset is loaded from:

```text
survey.csv
```

Users can also upload a new CSV through the Streamlit sidebar.

When a new dataset is uploaded:

```text
CSV Upload
    ↓
Dataset Loader
    ↓
Pandas DataFrame
    ↓
Dataset Profiler
    ↓
New LangGraph Workflow
```

The application displays:

* Number of rows
* Number of columns
* Number of numerical columns
* Number of categorical columns
* Dataset preview

---

### Document workflow

Users can upload PDF or TXT files through the knowledge-base section.

The uploaded document is processed through the RAG ingestion pipeline and stored in the vector database.

```text
PDF / TXT
   ↓
Document Loader
   ↓
Text Splitter
   ↓
Embedding Service
   ↓
ChromaDB
```

The resulting vector store can then be queried by the RAG agent.

---

## Technology Stack

| Technology             | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| Python                 | Core programming language                |
| LangGraph              | Agent workflow orchestration             |
| LangChain              | LLM and retrieval framework              |
| Groq                   | LLM inference                            |
| Pandas                 | Structured data analysis                 |
| ChromaDB               | Vector database                          |
| Hugging Face           | Embeddings                               |
| Streamlit              | Interactive web application              |
| Pydantic / Typed State | Structured agent state and data handling |

---

## Project Structure

```text
Intelligent-Hybrid-Analysis-Agent/
│
├── documents/
│   └── PEI_US_2016_Dataset_Detailed_Guide.pdf
│
├── src/
│   ├── agents/
│   │   ├── analytics_agent.py
│   │   ├── analytics_graph.py
│   │   ├── hybrid_agent.py
│   │   ├── rag_agent.py
│   │   └── router.py
│   │
│   ├── models/
│   │   └── state.py
│   │
│   ├── services/
│   │   ├── analytics_service.py
│   │   ├── analytics_tools.py
│   │   ├── embedding_service.py
│   │   ├── rag_ingestion.py
│   │   ├── text_splitter.py
│   │   └── vector_store.py
│   │
│   ├── utils/
│   │   ├── dataset_loader.py
│   │   ├── dataset_pofiler.py
│   │   └── document_loader.py
│   │
│   ├── app.py
│   ├── config.py
│   ├── graph.py
│   ├── ingest.py
│   ├── llm.py
│   └── test_retrieval.py
│
├── documents/
├── survey.csv
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Harshul-Negi/Intelligent-Hybrid-Analysis-Agent.git
```

```bash
cd Intelligent-Hybrid-Analysis-Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
MODEL_NAME=your_model_name
```

The `.env` file is intentionally excluded from Git through `.gitignore`.

Never commit API keys or other credentials to the repository.

---

## Running the Application

From the project root, run:

```bash
streamlit run src/app.py
```

The Streamlit interface will open in your browser.

---

## Using the Application

### Default dataset

The application initially loads:

```text
survey.csv
```

You can immediately ask analytical questions about the dataset.

### Uploading another dataset

Use:

**Sidebar → Dataset → Upload a CSV dataset**

The application loads and profiles the new dataset and rebuilds the LangGraph workflow around it.

### Adding knowledge documents

Use:

**Sidebar → Knowledge Base → Upload a document**

Supported formats:

```text
PDF
TXT
```

The document is processed and added to the RAG knowledge base.

---

## Example Questions

### Analytics questions

```text
How many records are in the dataset?
```

```text
What are the unique values in this column?
```

```text
What is the average value of National Electoral Integrity?
```

```text
Which state has the highest value?
```

### RAG questions

```text
What methodology was used to calculate the Electoral Integrity score?
```

```text
What does the National Electoral Integrity measure represent?
```

### Hybrid questions

```text
How does the dataset's Electoral Integrity score relate to the methodology described in the documentation?
```

```text
Which states have high integrity scores and how does the documentation explain the interpretation of these scores?
```

The router determines whether the question requires analytics, RAG, or both.

---

## Dataset

The repository includes a sample survey dataset and its supporting documentation:

```text
survey.csv
```

and:

```text
documents/
└── PEI_US_2016_Dataset_Detailed_Guide.pdf
```

The documentation provides contextual information required by the RAG workflow.

The included dataset is used as the default dataset for demonstrating the analytics capabilities of the application.

---

## Design Principles

### Separation of Responsibilities

Different components are responsible for different tasks:

```text
Router
   ↓
Decision making

Analytics Agent
   ↓
Structured data reasoning

RAG Agent
   ↓
Document retrieval and question answering

Hybrid Agent
   ↓
Combining analytical and retrieved results

LangGraph
   ↓
Workflow orchestration
```

This separation makes the system easier to extend and maintain.

### Dynamic Dataset Handling

The graph is created through:

```python
create_graph(df)
```

This allows the application to rebuild its analytics workflow when the user uploads a different CSV dataset.

### Source-Aware RAG

RAG responses retain source information where available, allowing the Streamlit interface to display the documents and pages associated with retrieved information.

---

## Future Improvements

Potential improvements include:

* Streaming LLM responses
* More advanced analytical tools
* Additional document formats
* Improved citation handling
* Persistent user-specific knowledge bases
* Authentication and multi-user support
* Evaluation benchmarks for analytics and RAG accuracy
* Automated testing for agent routing
* Observability and tracing
* Deployment using Docker and cloud infrastructure

---

## Author

**Harshul Negi**

Computer Science & Engineering

Interested in building intelligent AI systems using:

* Generative AI
* Large Language Models
* Agentic AI
* Retrieval-Augmented Generation
* Machine Learning
* AI-powered applications
