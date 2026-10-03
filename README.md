# Spotify × Airtel Advertising RAG Chatbot

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chatbot built with **LangChain, FAISS, Hugging Face embeddings, and OpenRouter**.

The chatbot uses a CSV dataset containing advertising campaign information and allows users to ask natural-language questions about campaigns, audiences, objectives, platforms, impressions, clicks, and campaign descriptions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What is RAG?](#what-is-rag)
- [Project Architecture](#project-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Example Questions](#example-questions)
- [How the Code Works](#how-the-code-works)
- [FAISS Vector Store](#faiss-vector-store)
- [Embeddings](#embeddings)
- [LLM](#llm)
- [RAG Prompt](#rag-prompt)
- [Updating the Dataset](#updating-the-dataset)
- [Troubleshooting](#troubleshooting)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Learning Outcomes](#learning-outcomes)
- [License](#license)

---

# Project Overview

The **Spotify × Airtel Advertising RAG Chatbot** is a small RAG application designed as a beginner project for understanding how retrieval-augmented generation works.

The application reads advertising campaign information from a CSV file, converts each campaign into a LangChain `Document`, generates vector embeddings, stores those embeddings in a FAISS vector database, and retrieves the most relevant campaigns when a user asks a question.

The retrieved campaign information is then provided to an LLM through a carefully designed prompt.

### Basic flow

```text
User Question
      │
      ▼
LangChain Retriever
      │
      ▼
FAISS Vector Database
      │
      ▼
Relevant Campaign Records
      │
      ▼
Prompt + Retrieved Context
      │
      ▼
OpenRouter LLM
      │
      ▼
Natural Language Answer
```

---

# What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**.

A traditional LLM answers questions using information learned during its training.

A RAG application adds an external knowledge source.

Instead of asking:

```text
User → LLM → Answer
```

we use:

```text
User
  │
  ▼
Retrieve relevant information
  │
  ▼
Add information to prompt
  │
  ▼
LLM
  │
  ▼
Answer
```

This allows the application to answer questions using information contained in a private or custom dataset.

In this project, the custom knowledge source is:

```text
data/company_data.csv
```

---

# Project Architecture

The application contains five major stages.

## 1. Load data

Pandas reads the CSV file.

```text
company_data.csv
       │
       ▼
    Pandas
```

---

## 2. Convert records into Documents

Each CSV row is converted into a LangChain `Document`.

For example:

```text
Campaign ID: A001
Brand: Airtel
Campaign: 5G Awareness
Platform: Spotify
Year: 2024
Ad Type: Audio Ad
Target Audience: Youth
Objective: 5G Awareness
Impressions: 25000000
Clicks: 850000
Description: Airtel promotes awareness of its 5G services...
```

This becomes a searchable document.

---

## 3. Generate embeddings

The text is converted into numerical vectors using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

```text
Campaign text
      │
      ▼
Embedding Model
      │
      ▼
[0.21, -0.14, 0.73, ...]
```

Similar pieces of text should produce vectors that are close to one another in vector space.

---

## 4. Store vectors in FAISS

FAISS is used as the vector store.

```text
Documents
    │
    ▼
Embeddings
    │
    ▼
FAISS
```

When a user asks a question, the question is also converted into an embedding.

FAISS then searches for campaign records that are semantically similar to the question.

---

## 5. Generate the answer

The retrieved records are inserted into the prompt.

The LLM receives:

```text
Context:
<retrieved campaign records>

Question:
<user question>
```

It then generates an answer based on the retrieved context.

---

# Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| LangChain | RAG application framework |
| Pandas | CSV/data processing |
| Hugging Face | Local embedding model |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| OpenRouter | LLM API |
| python-dotenv | Environment variable management |

---

# Project Structure

```text
spotify_airtel_rag/
│
├── .env
├── requirements.txt
├── rag_chatbot.py
│
├── data/
│   └── company_data.csv
│
└── faiss_index/
    ├── index.faiss
    └── index.pkl
```

### `.env`

Contains API credentials and model configuration.

```text
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=...
```

The `.env` file should **never be committed to GitHub**.

---

### `requirements.txt`

Contains the Python dependencies required by the project.

---

### `rag_chatbot.py`

Contains the complete RAG application.

It is responsible for:

- Loading the CSV
- Creating LangChain documents
- Creating embeddings
- Creating/loading FAISS
- Creating the LLM
- Retrieving relevant documents
- Building the prompt
- Generating answers
- Running the terminal chatbot

---

### `data/company_data.csv`

Contains the advertising campaign data.

---

### `faiss_index/`

Contains the locally persisted FAISS vector database.

This directory is generated automatically by the application.

---

# Dataset

The CSV contains the following columns:

```text
campaign_id
brand
campaign
platform
year
ad_type
target_audience
objective
impressions
clicks
description
```

Example:

```csv
campaign_id,brand,campaign,platform,year,ad_type,target_audience,objective,impressions,clicks,description
A001,Airtel,5G Awareness,Spotify,2024,Audio Ad,Youth,5G Awareness,25000000,850000,"Airtel promotes awareness of its 5G services among younger Spotify listeners."
```

### Column descriptions

| Column | Description |
|---|---|
| `campaign_id` | Unique campaign identifier |
| `brand` | Brand associated with campaign |
| `campaign` | Campaign name |
| `platform` | Advertising platform |
| `year` | Campaign year |
| `ad_type` | Type of advertisement |
| `target_audience` | Intended audience |
| `objective` | Campaign objective |
| `impressions` | Number of impressions |
| `clicks` | Number of clicks |
| `description` | Human-readable campaign description |

> The sample campaign records included with this learning project are illustrative/synthetic data and should not be treated as verified real-world campaign metrics.

---

# Prerequisites

Before running the project, make sure you have:

- Python 3.9+
- pip
- An OpenRouter API key
- Internet access for the OpenRouter LLM
- Basic Python knowledge

The embedding model runs locally, so a separate embedding API key is not required.

---

# Installation

## 1. Clone the repository

If the project is hosted on GitHub:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Then:

```bash
cd spotify_airtel_rag
```

---

## 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
```

Replace:

```text
your_openrouter_api_key
```

with your actual API key.

Do not commit `.env` to Git.

A recommended `.gitignore` entry is:

```gitignore
.env
venv/
__pycache__/
faiss_index/
```

---

# Running the Application

From the project root:

```bash
python rag_chatbot.py
```

You should see something similar to:

```text
==============================
 Spotify x Airtel RAG Chatbot
==============================

Loaded 8 campaign records.
Creating FAISS index...
FAISS index saved.

Chatbot is ready!
Ask questions about your campaigns.
Type 'exit' to stop.

You:
```

You can now start asking questions.

---

# Example Questions

### Campaign information

```text
What is the 5G Awareness campaign about?
```

### Target audience

```text
Which campaigns target youth?
```

### Campaign objective

```text
What was the objective of campaign A001?
```

### Platform

```text
Which campaigns were run on Spotify?
```

### Campaign comparison

```text
Tell me about the Airtel campaigns on Spotify.
```

### Metrics

```text
What were the impressions for the 5G Awareness campaign?
```

### Exit

```text
exit
```

or:

```text
quit
```

---

# How the Code Works

## Step 1 — Load environment variables

The application starts with:

```python
from dotenv import load_dotenv

load_dotenv()
```

This loads values from `.env`.

The OpenRouter API key is then retrieved:

```python
API_KEY = os.getenv("OPENROUTER_API_KEY")
```

---

## Step 2 — Read the CSV

Pandas loads the dataset:

```python
df = pd.read_csv(CSV_PATH).fillna("")
```

Each row represents an advertising campaign.

---

## Step 3 — Create LangChain Documents

Every row is converted into a `Document`.

```python
Document(
    page_content=content,
    metadata=metadata
)
```

The actual campaign information goes into `page_content`.

Some useful identifiers are stored in `metadata`:

```python
metadata = {
    "campaign_id": str(row["campaign_id"]),
    "brand": str(row["brand"]),
    "platform": str(row["platform"]),
    "year": str(row["year"]),
}
```

Metadata is useful for identifying the source documents returned by the retriever.

---

# Embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

through LangChain's Hugging Face integration.

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

The model converts text into vectors.

For example:

```text
"Spotify audio advertising for young users"
```

might conceptually become:

```text
[0.12, -0.43, 0.81, ...]
```

The exact vector values aren't meaningful to us directly.

What matters is the distance between vectors.

---

# FAISS Vector Store

FAISS is responsible for vector similarity search.

The application creates the vector database:

```python
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)
```

It is then saved locally:

```python
vector_store.save_local(str(FAISS_PATH))
```

On subsequent executions, the application loads the existing index instead of rebuilding it.

```python
vector_store = FAISS.load_local(
    str(FAISS_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

# Retrieval

When the user asks a question, the application creates a retriever:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)
```

`k=4` means the application attempts to retrieve the four most relevant documents.

The query is then passed to the retriever:

```python
retrieved_docs = retriever.invoke(question)
```

For example:

```text
User:
Which Airtel campaigns target young audiences?
```

FAISS searches the campaign vectors and returns the records that are semantically relevant.

---

# Prompt Construction

The retrieved documents are combined into context:

```python
context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)
```

The prompt then contains:

```text
Context:
<campaign 1>

<campaign 2>

<campaign 3>

<campaign 4>

User question:
Which Airtel campaigns target young audiences?
```

The model is instructed to answer using the supplied context.

---

# LLM

The LLM is accessed through OpenRouter using an OpenAI-compatible API endpoint.

```python
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)
```

`temperature=0` is used to make responses more deterministic.

The exact free model available through OpenRouter can change, so the model specified in `.env` can be replaced with another currently available model.

---

# RAG Prompt

The chatbot uses a grounding instruction similar to:

```text
Answer the user's question using ONLY the
campaign information provided in the context.

Do not invent campaign facts or metrics.

If the answer is not in the context, say:
"I could not find that information in the dataset."
```

This is important because an LLM can otherwise generate information that isn't actually present in the dataset.

The goal of the prompt is to reduce hallucination and keep answers grounded in retrieved campaign records.

---

# Updating the Dataset

If you add or modify records in:

```text
data/company_data.csv
```

you need to rebuild the FAISS index.

Delete:

```text
faiss_index/
```

Then run:

```bash
python rag_chatbot.py
```

The application will:

```text
CSV
 ↓
Documents
 ↓
Embeddings
 ↓
New FAISS index
```

---

# Troubleshooting

## `OPENROUTER_API_KEY is missing`

Check that `.env` exists in the project root:

```text
spotify_airtel_rag/
├── .env
└── rag_chatbot.py
```

And that it contains:

```env
OPENROUTER_API_KEY=your_key
```

---

## CSV file not found

Make sure the dataset exists at:

```text
data/company_data.csv
```

The structure should be:

```text
project/
├── rag_chatbot.py
└── data/
    └── company_data.csv
```

---

## FAISS gives unexpected results after changing the CSV

Delete:

```text
faiss_index/
```

and run the application again.

The old index contains embeddings generated from the previous dataset.

---

## OpenRouter model error

The configured model may no longer be available or may have changed its free-tier availability.

Check the currently available models in OpenRouter and update:

```env
OPENROUTER_MODEL=...
```

---

## Slow first startup

The first execution may take longer because the Hugging Face embedding model needs to be downloaded.

Later executions can reuse the downloaded model and the saved FAISS index.

---

# Current Limitations

This is intentionally a small beginner RAG project.

## 1. It is not a complete analytics system

Although the CSV contains:

```text
impressions
clicks
```

FAISS is primarily performing semantic retrieval.

For example, asking:

```text
What is the total number of Airtel impressions?
```

should ideally be handled by a structured data operation rather than relying solely on semantic retrieval.

A future version can use Pandas for:

```text
SUM
AVERAGE
COUNT
MAX
MIN
GROUP BY
FILTER
```

and use RAG for qualitative questions.

---

## 2. FAISS does not understand business logic

FAISS understands vector similarity.

It does not inherently understand:

```text
"highest impressions"
```

as:

```python
df["impressions"].max()
```

That distinction is important when building production RAG systems.

---

## 3. The dataset is small

This project uses a small CSV so that the RAG architecture is easy to understand.

A production application could have:

- Thousands of campaigns
- PDF reports
- Marketing documents
- Product documentation
- Customer research
- Multiple data sources

---

## 4. No conversation memory

The current chatbot treats each question independently.

For example:

```text
You: Tell me about A001.

Bot: ...

You: What was its objective?
```

The second question does not necessarily understand that "its" refers to A001.

Conversation memory can be added later.

---

# Future Improvements

Once the basic RAG chatbot works, the project can evolve gradually.

## Version 2 — Better Retrieval

Add:

- Metadata filtering
- Better chunking
- Retrieval scores
- More sophisticated retrievers
- Source citations

---

## Version 3 — RAG + Data Analytics

Add Pandas-based tools for questions such as:

```text
What is the total number of impressions?

Which campaign has the highest clicks?

What is the average CTR?

How many Airtel campaigns ran on Spotify?

Show campaigns from 2025.
```

This creates a **hybrid RAG + structured-data system**.

---

## Version 4 — Conversation Memory

Allow follow-up questions:

```text
You:
Tell me about campaign A001.

Bot:
A001 is the 5G Awareness campaign...

You:
Who was its target audience?

Bot:
The campaign targeted youth.
```

---

## Version 5 — Web Interface

Replace the terminal interface with a simple Streamlit UI.

Potential interface:

```text
+--------------------------------------+
|     Spotify × Airtel RAG Chatbot     |
+--------------------------------------+
|                                      |
|  Ask a question...                   |
|                                      |
|  [ Which campaign had the most       |
|    impressions?                  ]   |
|                                      |
|  Answer:                             |
|  ...                                 |
|                                      |
|  Sources:                            |
|  A001                                |
|  A007                                |
+--------------------------------------+
```

---

## Version 6 — Production Architecture

A larger implementation could look like:

```text
                  ┌──────────────┐
                  │ CSV / PDFs   │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Data Loader  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Chunking    │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │  Embeddings  │
                  └──────┬───────┘
                         │
                         ▼
                  ┌──────────────┐
                  │    FAISS     │
                  └──────┬───────┘
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Semantic RAG             Data Analytics
             │                       │
             └───────────┬───────────┘
                         ▼
                       LLM
                         │
                         ▼
                    Final Answer
```

---

# Learning Outcomes

By completing this project, you will understand the fundamental components of a RAG application:

- What RAG is
- How documents are represented in LangChain
- What embeddings are
- How semantic similarity works
- How FAISS performs vector search
- What a retriever does
- How retrieved context is passed to an LLM
- How prompt grounding works
- How to persist a vector database
- How to use environment variables for API keys
- Why RAG and structured data analytics are different problems
- How a basic RAG system can evolve into a more capable application

---

# Key Concepts

The most important architecture to remember is:

```text
DATA
 ↓
DOCUMENTS
 ↓
EMBEDDINGS
 ↓
VECTOR STORE
 ↓
RETRIEVER
 ↓
CONTEXT
 ↓
PROMPT
 ↓
LLM
 ↓
ANSWER
```

In this project:

```text
CSV
 ↓
LangChain Documents
 ↓
Hugging Face Embeddings
 ↓
FAISS
 ↓
Retriever
 ↓
Campaign Context
 ↓
OpenRouter
 ↓
Answer
```

---

# Why This Project?

This project is intentionally small.

Instead of immediately building a complex AI agent, the goal is to understand the core RAG pipeline first.

Once the basic pipeline is understood, additional capabilities such as:

- structured data analysis,
- conversation memory,
- metadata filtering,
- multiple data sources,
- web interfaces,
- evaluation,
- and production deployment

can be added incrementally.

The fundamental idea remains the same:

> **Retrieve the relevant information first, then ask the LLM to generate an answer using that information.**

---

# License

This project is intended for educational and experimentation purposes.

Add an appropriate open-source license if you plan to publish the project publicly.
