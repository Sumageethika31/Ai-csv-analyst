import os
import re
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


# ---------- Setup ----------
load_dotenv()
st.set_page_config(page_title="AI CSV Analyst (Pandas + RAG)", layout="wide")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found. Put it in .env as OPENAI_API_KEY=...")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()


# ---------- Helpers ----------
def read_csv_safely(uploaded_file) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=enc)
        except UnicodeDecodeError:
            continue

    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding="utf-8", errors="replace")


def build_vectorstore_from_df(df: pd.DataFrame) -> FAISS:
    rows = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1).tolist()
    splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    docs = splitter.create_documents(rows)
    return FAISS.from_documents(docs, embeddings)


def is_chart_request(q: str) -> bool:
    q = q.lower()
    keywords = ["plot", "chart", "graph", "trend", "distribution", "hist", "bar", "line", "scatter"]
    return any(k in q for k in keywords)


def safe_exec(code: str, local_vars: dict):
    banned = [
        "import os", "import sys", "subprocess", "open(", "eval(", "exec(",
        "socket", "shutil", "pathlib", "__", "pip", "conda"
    ]
    lower = code.lower()
    if any(b in lower for b in banned):
        raise ValueError("Blocked unsafe code.")
    exec(code, {"__builtins__": {}}, local_vars)


def plot_result_df(result_df: pd.DataFrame, chart_type: str = "Auto"):
    if result_df is None or result_df.empty:
        st.warning("No data to plot.")
        return

    numeric_cols = result_df.select_dtypes(include=["number"]).columns.tolist()
    non_numeric_cols = [c for c in result_df.columns if c not in numeric_cols]

    if not numeric_cols:
        st.warning("No numeric columns found in result to chart.")
        return

    y = numeric_cols[0]
    x = non_numeric_cols[0] if non_numeric_cols else result_df.index

    if chart_type == "Auto":
        if len(result_df) <= 25 and non_numeric_cols:
            chart_type_use = "Bar"
        else:
            chart_type_use = "Line"
    else:
        chart_type_use = chart_type

    fig, ax = plt.subplots()

    if chart_type_use == "Bar":
        ax.bar(result_df[x].astype(str), result_df[y])
        ax.set_xticklabels(result_df[x].astype(str), rotation=45, ha="right")
    elif chart_type_use == "Line":
        ax.plot(result_df[x], result_df[y], marker="o")
    elif chart_type_use == "Histogram":
        ax.hist(result_df[y].dropna(), bins=30)

    ax.set_title(f"{chart_type_use} Chart: {y}")
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    st.pyplot(fig, clear_figure=True)


def llm_generate_pandas_plan(df: pd.DataFrame, question: str) -> dict:
    cols = df.columns.tolist()
    sample = df.head(8).to_dict(orient="records")

    system = (
        "You are a senior data analyst. "
        "You write correct pandas code to answer questions about a DataFrame named df. "
        "Return ONLY valid JSON. No markdown."
    )

    user = {
        "task": "Generate pandas code to answer the question using df.",
        "df_columns": cols,
        "df_sample_rows": sample,
        "question": question,
        "requirements": {
            "must_define_one": ["result_df", "result_text"],
            "result_df_rule": "If the output is tabular, set result_df to a pandas DataFrame.",
            "result_text_rule": "If the output is narrative, set result_text to a string.",
            "chart": "If the user asks for a chart/plot/trend/graph, also create matplotlib code that sets fig = plt.gcf() after plotting.",
            "no_imports": "Do not import anything. Assume pandas as pd and matplotlib.pyplot as plt already exist.",
        }
    }

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}
        ]
    )

    try:
        obj = json.loads(resp.content)
    except Exception:
        m = re.search(r"\{.*\}", resp.content, flags=re.S)
        if not m:
            raise ValueError("Model did not return valid JSON.")
        obj = json.loads(m.group(0))

    if "pandas_code" not in obj:
        if "code" in obj:
            obj["pandas_code"] = obj["code"]
        elif "python_code" in obj:
            obj["pandas_code"] = obj["python_code"]
        elif "analysis_code" in obj:
            obj["pandas_code"] = obj["analysis_code"]

    if "pandas_code" not in obj:
        retry_prompt = (
            "Return ONLY valid JSON with these keys exactly: "
            "pandas_code (string), chart_code (string). "
            "No other text."
        )
        resp2 = llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
                {"role": "user", "content": retry_prompt},
            ]
        )
        try:
            obj2 = json.loads(resp2.content)
        except Exception:
            m2 = re.search(r"\{.*\}", resp2.content, flags=re.S)
            if not m2:
                raise ValueError("Model did not return valid JSON on retry.")
            obj2 = json.loads(m2.group(0))

        if "pandas_code" not in obj2 and "code" in obj2:
            obj2["pandas_code"] = obj2["code"]

        if "pandas_code" not in obj2:
            raise ValueError("Missing pandas_code in model output (after retry).")

        obj = obj2

    obj.setdefault("chart_code", "")
    return obj


def rag_answer(vectorstore: FAISS, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    context_docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in context_docs])

    prompt = (
        "You are a data analyst assistant.\n"
        "Use ONLY the provided CSV row context to answer.\n"
        "If the context is insufficient, say what is missing.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer clearly and specifically."
    )

    resp = llm.invoke(prompt)
    return resp.content


def route_question(df: pd.DataFrame, question: str) -> str:
    """
    Returns: "compute" or "rag"
    Heuristic first (fast), then LLM fallback.
    """
    q = question.lower().strip()

    compute_kw = [
        "top", "bottom", "rank", "highest", "lowest", "sum", "total", "average", "mean",
        "median", "min", "max", "count", "group by", "by ", "trend", "over time",
        "month", "weekly", "daily", "year", "compare", "percentage", "percent",
        "filter", "where", "only", "between", "greater than", "less than",
        "correlation", "distribution", "hist", "chart", "plot", "graph"
    ]

    rag_kw = [
        "summarize", "summary", "explain", "why", "what does it say", "insights",
        "describe", "find rows", "mentions", "complaints", "feedback", "notes",
        "comment", "reason", "issue", "problem", "root cause"
    ]

    if any(k in q for k in compute_kw):
        return "compute"
    if any(k in q for k in rag_kw):
        return "rag"

    cols_lower = [c.lower() for c in df.columns]
    if any(c in q for c in cols_lower):
        return "compute"

    cols = df.columns.tolist()
    sample = df.head(5).to_dict(orient="records")

    system = (
        "You are a routing classifier for a CSV assistant. "
        "Choose the best tool:\n"
        "- compute: for aggregations, filtering, ranking, numeric analysis, charts.\n"
        "- rag: for fuzzy semantic questions, summarization, searching text/notes.\n"
        "Return ONLY one word: compute or rag."
    )

    user = {
        "question": question,
        "columns": cols,
        "sample_rows": sample,
        "rule": "If the user needs exact numbers across the dataset, choose compute."
    }

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}
        ]
    )

    decision = resp.content.strip().lower()
    return "rag" if "rag" in decision else "compute"


# ---------- UI ----------
st.title("AI CSV Analyst Assistant")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if not uploaded:
    st.info("Upload a CSV to begin.")
    st.stop()

df = read_csv_safely(uploaded)

st.subheader("Preview")
st.dataframe(df.head(20), use_container_width=True)

with st.expander("Column info"):
    st.write("Rows:", len(df))
    st.write("Columns:", df.columns.tolist())

if "vectorstore" not in st.session_state:
    with st.spinner("Building vector index (FAISS) for RAG..."):
        st.session_state.vectorstore = build_vectorstore_from_df(df)

mode = st.radio(
    "Choose answer mode",
    ["Smart (Auto)", "Compute (Pandas Agent)", "RAG (Semantic Retrieval)"],
    horizontal=True
)

question = st.text_input(
    "Ask a question about your CSV",
    placeholder="e.g., Top 5 products by revenue, or plot sales trend by month"
)

show_chart = st.checkbox("Auto-generate chart (Compute mode)", value=True)
chart_type = st.selectbox("Chart type", ["Auto", "Bar", "Line", "Histogram"], index=0)

if st.button("Run") and question.strip():
    q = question.strip()

    if mode == "Smart (Auto)":
        routed = route_question(df, q)
        st.caption(f"Auto-selected mode: **{'Compute' if routed == 'compute' else 'RAG'}**")
    elif mode == "RAG (Semantic Retrieval)":
        routed = "rag"
    else:
        routed = "compute"

    if routed == "rag":
        with st.spinner("Retrieving relevant rows and generating answer..."):
            ans = rag_answer(st.session_state.vectorstore, q)
        st.subheader("Answer (RAG)")
        st.write(ans)

    else:
        st.subheader("Answer (Compute)")
        with st.spinner("Generating pandas plan..."):
            try:
                plan = llm_generate_pandas_plan(df, q)
            except Exception as e:
                st.error(f"Could not generate pandas code. Try rephrasing.\n\nDetails: {e}")
                st.stop()

        pandas_code = plan["pandas_code"]
        chart_code = plan.get("chart_code", "")

        st.code(pandas_code, language="python")

        local_vars = {
            "df": df,
            "pd": pd,
            "plt": plt,
            "result_df": None,
            "result_text": None,
            "fig": None
        }

        try:
            safe_exec(pandas_code, local_vars)
        except Exception as e:
            st.error(f"Error running pandas code: {e}")
            st.stop()

        if local_vars.get("result_df") is not None:
            out_df = local_vars["result_df"]
            st.dataframe(out_df, use_container_width=True)

            if show_chart:
                st.subheader("Chart")
                plot_result_df(out_df, chart_type=chart_type)

        elif local_vars.get("result_text") is not None:
            st.write(local_vars["result_text"])
        else:
            st.warning("No result_df or result_text produced. Try rephrasing the question.")

        if is_chart_request(q) and chart_code.strip():
            st.subheader("Model Chart (Optional)")
            st.code(chart_code, language="python")
            try:
                plt.close("all")
                safe_exec(chart_code, local_vars)
                fig = local_vars.get("fig") or plt.gcf()
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Error generating model chart: {e}")