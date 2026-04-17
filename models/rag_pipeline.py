"""
rag_pipeline.py
---------------
Loads the persisted Chroma index and exposes a single
query() function used by the Streamlit chatbot tab.

Flow:
    user question
        → embed with paraphrase-MiniLM-L6-v2
        → retrieve top-k answers from Chroma
        → build prompt with retrieved context
        → send to Groq (llama3-8b-8192)
        → return (answer_text, source_docs)
"""

import os
import logging
import contextlib
from dotenv import load_dotenv

# Suppress noisy model-loader warnings (Python-level loggers)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


@contextlib.contextmanager
def _silence_fds():
    """Redirect stdout/stderr at the fd level to suppress safetensors load report."""
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
from langchain_chroma import Chroma
from groq import Groq

load_dotenv()

# ── Paths & config ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR  = os.path.join(BASE_DIR, "models", "chroma_db")
EMBED_MODEL = "sentence-transformers/paraphrase-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"
TOP_K       = 4          # number of chunks to retrieve

# ── Singleton loaders (cached across Streamlit reruns) ─────
_embeddings  = None
_vectorstore = None
_groq_client = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _silence_fds():
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embeddings


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                f"Chroma index not found at {CHROMA_DIR}.\n"
                "Run:  python scripts/build_index.py"
            )
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=_get_embeddings(),
        )
    return _vectorstore


def _get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. "
                "Add it to your .env file or set it as an environment variable."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _build_prompt(question: str, context_docs: list) -> str:
    """Build the RAG prompt by injecting retrieved context."""
    context_text = "\n\n".join(
        f"[{doc.metadata.get('label','?')}] {doc.page_content}"
        for doc in context_docs
    )
    prompt = f"""You are an expert on Alexander Elder's trading methodology and philosophy.
Use ONLY the context below to answer the question.
If the context does not contain enough information, say so honestly.

--- CONTEXT ---
{context_text}
--- END CONTEXT ---

Question: {question}

Answer:"""
    return prompt


def query(question: str, top_k: int = TOP_K):
    """
    Parameters
    ----------
    question : str
        The user's question.
    top_k : int
        Number of context chunks to retrieve.

    Returns
    -------
    answer : str
        The LLM-generated answer.
    sources : list[Document]
        The retrieved source documents shown in the UI.
    """
    # 1. Semantic search
    vs      = _get_vectorstore()
    sources = vs.similarity_search(question, k=top_k)

    # 2. Build prompt
    prompt  = _build_prompt(question, sources)

    # 3. Call Groq
    client  = _get_groq()
    chat    = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
    )
    answer = chat.choices[0].message.content.strip()

    return answer, sources
