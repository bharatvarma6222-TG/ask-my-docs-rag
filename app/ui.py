import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Ask My Docs", page_icon="📄", layout="centered")
st.title("📄 Ask My Docs")

st.caption(
    "Start API first:  python -m uvicorn app.api:app --reload --host 127.0.0.1 --port 8000")

# -------------------------
# API Status
# -------------------------
col1, col2 = st.columns([1, 3])
with col1:
    st.write("API Status:")
with col2:
    try:
        _ = requests.get(f"{API_URL}/", timeout=2)
        st.success("Online")
    except Exception:
        st.warning("Offline (start uvicorn on port 8000)")

st.divider()

# -------------------------
# Upload + Ingest PDF
# -------------------------
st.subheader("1) Upload a PDF")
uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])

ingest_col1, ingest_col2 = st.columns([1, 2])
with ingest_col1:
    ingest_btn = st.button("Ingest PDF", type="primary",
                           disabled=(uploaded is None))
with ingest_col2:
    clear_btn = st.button("Clear UI")

if clear_btn:
    st.rerun()

if ingest_btn and uploaded is not None:
    try:
        with st.spinner("Ingesting and indexing..."):
            r = requests.post(
                f"{API_URL}/ingest/pdf",
                files={
                    "file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=180,
            )
            r.raise_for_status()
            data = r.json()
        st.success(f"Ingested: {data.get('doc_id', uploaded.name)}")
    except requests.exceptions.ConnectionError:
        st.error("API is not running. Start it with uvicorn on port 8000.")
    except requests.exceptions.Timeout:
        st.error("Ingest timed out. Try again (or use a smaller PDF).")
    except Exception as e:
        try:
            st.error(f"Error: {r.text}")
        except Exception:
            st.error(f"Error: {e}")

st.divider()

# -------------------------
# Ask Question
# -------------------------
st.subheader("2) Ask a question")
show_debug = st.checkbox("Show debug (citations + confidence)", value=True)

question = st.text_input(
    "Your question", value="What is RAG and how does it work?")
top_k = st.slider("Top K evidence chunks", min_value=1, max_value=10, value=6)

ask_btn = st.button("Ask", type="primary")

if ask_btn:
    try:
        with st.spinner("Retrieving evidence and answering..."):
            r = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": int(top_k)},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()

        st.subheader("Answer")
        st.write(data.get("answer", ""))

        if show_debug:
            st.caption(f"Confidence: {data.get('confidence', 0):.2f}")

            st.subheader("Citations")
            citations = data.get("citations", [])
            if not citations:
                st.info("No citations returned.")
            else:
                for c in citations:
                    st.markdown(
                        f"**{c['doc_id']}** — page {c['page']} — `{c['chunk_id']}`")
                    st.write(c.get("snippet", ""))

    except requests.exceptions.ConnectionError:
        st.error(
            "API is not running. Start it with: python -m uvicorn app.api:app --reload (port 8000)")
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try again or reduce top_k.")
    except Exception:
        try:
            st.error(r.text)
        except Exception as e:
            st.error(str(e))

st.divider()

# -------------------------
# Refusal Demo Button
# -------------------------
st.subheader("3) Quick demo: refusal behavior")
st.caption(
    "This tests that the system refuses when the answer is not in the documents.")

if st.button("Test refusal (FIFA 2010)"):
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": "Who won the FIFA World Cup 2010?", "top_k": 6},
            timeout=30,
        )
        r.raise_for_status()
        st.json(r.json())
    except Exception:
        try:
            st.error(r.text)
        except Exception as e:
            st.error(str(e))
