"""
build_index.py
--------------
Step 2 of 2 in the data pipeline.

Reads data/elder_qa_master.csv (produced by preprocess.py), embeds every
answer using paraphrase-MiniLM-L6-v2, and persists the Chroma vector store
to models/chroma_db/.

Full pipeline:
    python scripts/preprocess.py   <- Step 1: clean & merge raw CSVs
    python scripts/build_index.py  <- Step 2: embed & index into Chroma
    python -m streamlit run app/main.py
"""

import os
import sys
import logging
import contextlib
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# Suppress noisy model-loader warnings (Python-level loggers)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


@contextlib.contextmanager
def _silence_fds():
    """Redirect stdout and stderr at the file-descriptor level.

    Needed because the safetensors Rust extension prints the
    'BertModel LOAD REPORT' directly to the underlying fd, bypassing
    Python's logging and sys.stdout/stderr wrappers.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (devnull, saved_out, saved_err):
            os.close(fd)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Paths ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH    = os.path.join(BASE_DIR, "data", "elder_qa_master.csv")
CHROMA_DIR  = os.path.join(BASE_DIR, "models", "chroma_db")
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"

def build():
    # 1. Load dataset
    print(f"Loading dataset from {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  {len(df)} rows loaded across labels: {df['label'].unique().tolist()}")

    # 2. Build LangChain Documents
    # Each document = the answer text; metadata carries question + label
    # so the UI can show the source context to the user
    documents = []
    for _, row in df.iterrows():
        doc = Document(
            page_content=str(row["answer"]),
            metadata={
                "question": str(row["question"]),
                "label":    str(row["label"]),
            }
        )
        documents.append(doc)

    print(f"  Built {len(documents)} documents.")

    # 3. Load embedding model
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  (First run will download ~90 MB — subsequent runs use cache)")
    with _silence_fds():
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    print("  Model loaded.")

    # 4. Embed + persist to Chroma
    print(f"\nEmbedding and indexing into Chroma at {CHROMA_DIR} ...")
    os.makedirs(CHROMA_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print(f"\nDone! {vectorstore._collection.count()} vectors stored.")
    print(f"Index saved to: {CHROMA_DIR}")
    print("\nYou can now run the app:  streamlit run app/main.py")

if __name__ == "__main__":
    print("\n" + "=" * 58)
    print("  Step 2/2 — Building vector index")
    print("=" * 58 + "\n")
    build()
