**AI CSV Analyst System**

An AI-powered data analysis web application that allows users to upload a CSV file and query it using natural language. The system intelligently evaluates each question and returns structured results including:

Exact numeric analysis (totals, rankings, trends)

Semantic insights from textual data

Automatic chart visualizations

Smart routing between computation and semantic retrieval

This project demonstrates applied AI engineering using Streamlit, LangChain, FAISS, and OpenAI APIs.

Project Overview

Data analysts typically write Pandas or SQL queries to explore datasets. This project automates that workflow by allowing users to interact with structured data using plain English.

Instead of manually writing aggregation code, the system:

Accepts a CSV file upload

Accepts a natural language question

Automatically determines whether the query requires numeric computation or semantic reasoning

Executes the appropriate AI workflow

Displays structured results and optional charts in a clean dashboard

The result is a lightweight AI-powered business intelligence system.

Key Features

Real-time CSV analysis

Smart routing (Compute vs RAG)

Deterministic dataset-wide aggregation

Semantic search over structured rows

Embedding-based retrieval using FAISS

Automated chart generation

Interactive web interface

Safe execution of LLM-generated code

Environment-isolated dependency management

Tech Stack
Frontend

Streamlit

Data Processing

Pandas

Python

AI Orchestration

LangChain

OpenAI (GPT-4o-mini)

Vector Search

FAISS

Visualization

Matplotlib

System Architecture
User Upload CSV
        ↓
Streamlit UI
        ↓
User Question
        ↓
Smart Router
     ↙        ↘
Compute       RAG
(Pandas)      (FAISS)
     ↓            ↓
Structured Result / Insight
        ↓
Automatic Chart Rendering
        ↓
UI Display
Project Structure
AI-CSV-Analyst/
│
├── streamlit_app.py       # Main application logic
├── requirements.txt       # Dependencies
├── .gitignore
├── README.md
│
├── screenshots/           # UI screenshots (optional)
└── sample_data/           # Example CSV files (optional)
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
Example Use Cases
Compute Example

Input:

Top 5 products by Sales

Output:

Structured ranking table

Automatically generated bar chart

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

If OpenAI key is not detected:

Ensure .env exists

Restart terminal

Verify variable name is correct

If dependency conflicts occur:

pip install --upgrade pip
Security Notes

Never commit .env

Store API keys securely

Use .gitignore

Restrict unsafe code execution

Validate model-generated outputs before execution

Future Enhancements

Planned improvements:

Hybrid mode (Compute + narrative explanation)

Conversational memory

SQL backend integration

Multi-dataset comparison

Downloadable reports

Cloud deployment

Role-based analytics dashboards

Performance Considerations

Full dataset deterministic computation via Pandas

Efficient embedding retrieval using FAISS

Lightweight Streamlit interface

Controlled LLM execution pipeline

Smart routing reduces unnecessary API calls
