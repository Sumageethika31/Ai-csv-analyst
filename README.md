AI CSV Analyst System

An AI-powered data analysis web application that allows users to upload a CSV file and query it using natural language. The system intelligently evaluates the question and returns structured results including:

Exact numeric analysis (totals, rankings, trends)

Missing insights or summaries (for text-based queries)

Automatic chart visualizations

Smart routing between computation and semantic retrieval

This project demonstrates applied AI engineering using Streamlit, LangChain, FAISS, and OpenAI APIs.

Project Overview

Data analysts typically write Pandas or SQL queries to explore datasets. This project automates that workflow by allowing users to interact with structured data using plain English.

The system works by:

Accepting a CSV upload

Accepting a natural language question

Automatically determining whether the query requires numeric computation or semantic reasoning

Executing the appropriate AI workflow

Displaying structured results and optional charts in a clean UI

Key Features

Real-time CSV analysis

Smart routing (Compute vs RAG)

Exact dataset-wide aggregation

Semantic search over structured rows

Automated chart generation

Interactive web interface

Safe LLM-generated code execution

Environment-isolated dependency management

Tech Stack
Frontend

Streamlit

Core Logic

Pandas

LangChain

FAISS

AI Layer

OpenAI API (GPT-4o-mini)

Embedding-based retrieval

Language

Python 3

System Architecture
User Upload CSV
        ↓
Streamlit UI
        ↓
User Question
        ↓
Smart Router
   ↓              ↓
Compute         RAG
(Pandas)       (FAISS)
   ↓              ↓
Structured Result / Insight
        ↓
Automatic Chart Rendering
        ↓
UI Display
Project Structure
AI-CSV-Analyst/
│
├── streamlit_app.py        # Main application logic
├── requirements.txt        # Dependencies
├── .gitignore
├── README.md
│
├── screenshots/            # UI screenshots (optional)
└── sample_data/            # Example CSV files (optional)
Installation Guide
1. Clone Repository
git clone https://github.com/YOUR_USERNAME/ai-csv-analyst.git
cd ai-csv-analyst
2. Create Virtual Environment

Mac/Linux:

python3 -m venv .venv
source .venv/bin/activate

Windows:

python -m venv .venv
.venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
4. Set Environment Variable

Create a .env file:

OPENAI_API_KEY=your_openai_key_here
Run Application
streamlit run streamlit_app.py

Open in browser:

http://127.0.0.1:8501
Example Use Case
Compute Example

Input:

Top 5 products by Sales

Output:

Structured ranking table
Auto-generated bar chart
RAG Example

Input:

Summarize customer complaints

Output:

Semantic summary based on retrieved dataset rows
Development Workflow

Run locally:

streamlit run streamlit_app.py

Modify code → Save → Streamlit auto-refreshes

Troubleshooting

If port 8501 is already in use:

kill -9 $(lsof -ti :8501)

If OpenAI key not detected:

Ensure .env exists

Restart terminal

Verify variable name is correct

If dependency conflicts occur:

pip install --upgrade pip
Security Notes

Never commit .env

Keep API keys secure

Use .gitignore

Prevent unsafe code execution

Future Enhancements

Planned improvements:

Hybrid mode (Compute + explanation)

Conversational memory

SQL backend integration

Multi-dataset comparison

Downloadable reports

Cloud deployment

Role-based analytics dashboards

Performance Considerations

Full dataset deterministic computation via Pandas

Embedding-based retrieval optimized with FAISS

Lightweight Streamlit interface

Controlled LLM execution pipeline
