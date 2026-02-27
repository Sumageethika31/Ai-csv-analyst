# AI CSV Analyst System  
### Smart Data Assistant (Pandas Compute + RAG + Intelligent Routing)

---

## Overview

AI CSV Analyst is an AI-powered data analysis web application that allows users to upload structured CSV datasets and query them using natural language.

The system intelligently routes questions to either:

- **Compute Engine (Pandas Agent)** for exact numeric analysis  
- **RAG Engine (Retrieval-Augmented Generation)** for semantic insights  

It automatically generates structured results and visualizations using a Streamlit interface.

This project demonstrates hybrid AI system design by combining deterministic computation with LLM-based reasoning.

---

## Problem Statement

Traditional data analysis requires writing Pandas or SQL queries. Many users struggle to extract insights from datasets without technical expertise.

This system bridges that gap by:

1. Accepting CSV uploads  
2. Allowing natural language questions  
3. Automatically choosing the correct processing engine  
4. Returning structured answers and visualizations  

---

## Key Features

- Natural language dataset querying  
- Smart routing (Compute vs RAG)  
- Exact numeric aggregation using Pandas  
- Embedding-based semantic search with FAISS  
- Automatic chart generation  
- Interactive Streamlit UI  
- Safe execution sandbox  
- Full-dataset deterministic analysis  

---

## Tech Stack

### Frontend
- Streamlit

### Backend / Processing
- Pandas
- LangChain (modern architecture)
- FAISS (vector database)

### AI Layer
- OpenAI API (GPT-4o-mini)
- Embedding-based retrieval

### Language
- Python 3

## Project Structure

```
ai-csv-analyst/
│
├── streamlit_app.py        # Main application
├── requirements.txt        # Dependencies
├── .gitignore
├── README.md
│
├── screenshots/            # UI screenshots (optional)
└── sample_data/            # Example datasets (optional)
```

---

## How It Works

### Compute Engine (Pandas Agent)

- LLM generates Pandas code from user question  
- Code executes on the full dataset  
- Produces exact numeric results  
- Generates visualizations automatically  

Best for:
- Top N queries  
- Totals and averages  
- Filtering and grouping  
- Trend analysis  
- KPI reporting  

---

### RAG Engine (Semantic Retrieval)

- Converts dataset rows into embeddings  
- Stores embeddings in FAISS  
- Retrieves top-k relevant rows  
- LLM generates contextual answer  

Best for:
- Summarization  
- Text search  
- Comments/notes analysis  
- Fuzzy semantic questions  

---

### Smart Router

Automatically selects:

- Numeric queries → Compute  
- Semantic queries → RAG  

This simulates production-grade AI assistant architecture.

---

## Example Queries

### Compute Mode

```
Top 5 products by Sales
Total Profit by Region
Monthly Sales trend
Average Discount by Segment
Sales between 2020 and 2021
```

Output:
- Structured DataFrame
- Auto-generated chart

---

### RAG Mode

```
Summarize customer complaints
Find rows mentioning shipping delay
Describe customer feedback trends
```

Output:
- Semantic summary generated from retrieved dataset rows

---

## Installation Guide

### Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/ai-csv-analyst.git
cd ai-csv-analyst
```

### Create Virtual Environment

Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

###  Install Dependencies

```bash
pip install -r requirements.txt
```

### Add Environment Variable

Create a `.env` file:

```
OPENAI_API_KEY=your_openai_key_here
```

---

## ▶️ Run Application

```bash
streamlit run streamlit_app.py
```

Open browser:

```
http://127.0.0.1:8501
```

---

## 🔍 Internal Processing Flow

**Compute Mode**
- LLM generates Pandas code
- Code safely executed
- Result stored in DataFrame
- Chart generated automatically

**RAG Mode**
- Rows embedded
- Stored in FAISS
- Top-k retrieved
- LLM generates insight

---

## Security Notes

- Never commit `.env`
- Keep API keys private
- Use `.gitignore`
- Restrict unsafe code execution

---

## Future Enhancements

- Hybrid mode (Compute + explanation)
- Conversational memory
- SQL backend integration
- Multi-file dataset support
- Cloud deployment
- Role-based dashboards
- Downloadable reports

