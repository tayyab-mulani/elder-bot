"""
main.py  —  Alexander Elder AI Assistant
-----------------------------------------
Streamlit app with two tabs:

    Tab 1 — Elder Chatbot
        RAG-powered Q&A over Alexander Elder's trading methodology.
        Uses paraphrase-MiniLM-L6-v2 embeddings + Chroma + Groq (Llama 3).

    Tab 2 — Triple Screen Backtester
        Interactive dashboard implementing the
        Elder Triple Screen trading engine (EMA + MACD + RSI).
        Users select any US ticker, date range, and strategy parameters.

Run:
    streamlit run app/main.py
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

# Make sure project root is on the path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.rag_pipeline    import query as rag_query
from models.backtest_engine import run_backtest

# Official portrait hosted on Dr. Elder’s site (Wix CDN) — see https://www.elder.com/about
ELDER_PORTRAIT_URL = (
    "https://static.wixstatic.com/media/4249d8_4d7cedadde2249478b3eeb7e0b0ccbbb~mv2.jpeg"
    "/v1/fill/w_520,h_390,al_c,q_85,usm_0.66_1.00_0.01/Dr%20Alexander%20Elder.jpeg"
)
ELDER_SITE_ABOUT = "https://www.elder.com/about"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alexander Elder AI Assistant",
    page_icon="📈",
    layout="wide",
)

# ── Global styling ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --bg-deep: #050810;
        --bg-mid: #0c1222;
        --ink: #f1f5ff;
        --muted: #a8b8d8;
        --primary: #8b5cf6;
        --secondary: #06b6d4;
        --accent: #f59e0b;
        --surface: rgba(17, 24, 39, 0.72);
        --surface-strong: rgba(15, 23, 42, 0.94);
        --stroke: rgba(129, 140, 248, 0.22);
        --text-chip-bg: rgba(30, 41, 59, 0.85);
    }
    .stApp {
        background-color: var(--bg-deep);
        background-image:
            radial-gradient(ellipse 120% 80% at 20% -30%, rgba(99, 102, 241, 0.35), transparent 55%),
            radial-gradient(ellipse 90% 60% at 100% 0%, rgba(6, 182, 212, 0.22), transparent 50%),
            radial-gradient(ellipse 70% 50% at 80% 100%, rgba(245, 158, 11, 0.14), transparent 45%),
            linear-gradient(165deg, var(--bg-deep) 0%, var(--bg-mid) 45%, #0a0f1c 100%),
            repeating-linear-gradient(
                -12deg,
                transparent,
                transparent 80px,
                rgba(255, 255, 255, 0.015) 80px,
                rgba(255, 255, 255, 0.015) 81px
            );
        color: var(--ink);
    }
    .main .block-container {
        padding-top: 1.2rem;
        max-width: 1200px;
    }
    .main .block-container p, .main .block-container li {
        color: #e2e8f0;
    }
    .hero-card {
        background: linear-gradient(130deg, rgba(139, 92, 246, 0.35) 0%, rgba(6, 182, 212, 0.24) 58%, rgba(245, 158, 11, 0.24) 100%);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.18);
        backdrop-filter: blur(8px);
        border-radius: 22px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: 0 16px 55px rgba(2, 6, 23, 0.42);
    }
    .hero-kpis {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.65rem;
        margin-top: 0.95rem;
    }
    .kpi-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 12px;
        padding: 0.5rem 0.7rem;
        font-size: 0.86rem;
    }
    .section-title {
        font-weight: 800;
        font-size: 1.28rem;
        margin-top: 0.5rem;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
        background: linear-gradient(105deg, #f8fafc 0%, #c7d2fe 40%, #5eead4 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-subtitle {
        color: var(--muted);
        margin-bottom: 1rem;
        padding: 0.55rem 0.85rem;
        background: var(--text-chip-bg);
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        line-height: 1.5;
    }
    .glass-panel {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.55), rgba(15, 23, 42, 0.75));
        border: 1px solid var(--stroke);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        backdrop-filter: blur(10px);
        margin-bottom: 0.75rem;
        color: #e2e8f0;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    }
    .glass-panel strong {
        color: #c4b5fd;
    }
    .hint-banner {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.75rem 1rem;
        margin-bottom: 0.85rem;
        border-radius: 14px;
        background: linear-gradient(90deg, rgba(6, 182, 212, 0.12), rgba(139, 92, 246, 0.15));
        border: 1px solid rgba(6, 182, 212, 0.35);
        color: #cbd5f5;
        font-size: 0.95rem;
        line-height: 1.45;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }
    .hint-banner .hint-icon {
        font-size: 1.15rem;
        flex-shrink: 0;
    }
    .portrait-frame {
        position: relative;
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.22);
        box-shadow:
            0 12px 40px rgba(0, 0, 0, 0.45),
            0 0 0 1px rgba(139, 92, 246, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.12);
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.8));
    }
    .portrait-frame img {
        display: block;
        width: 100%;
        height: auto;
        vertical-align: middle;
    }
    .portrait-frame::after {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(180deg, transparent 55%, rgba(5, 8, 16, 0.65) 100%);
    }
    .photo-credit {
        margin: 0.5rem 0 0 0;
        font-size: 0.78rem;
        color: #8896b5;
        text-align: center;
    }
    .photo-credit a {
        color: #7dd3fc;
        text-decoration: none;
    }
    .photo-credit a:hover {
        text-decoration: underline;
    }
    .tips-deck {
        margin: 0.5rem 0 1.25rem 0;
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 27, 75, 0.55));
        border: 1px solid rgba(129, 140, 248, 0.28);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .tips-deck-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e0e7ff;
        margin-bottom: 0.65rem;
        letter-spacing: -0.01em;
    }
    .tips-deck-foot {
        margin-top: 0.75rem;
        font-size: 0.8rem;
        color: #8896b5;
        line-height: 1.45;
    }
    .tips-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 0.65rem;
    }
    .tip-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.75rem 0.85rem;
        color: #e2e8f0;
        font-size: 0.9rem;
        line-height: 1.45;
    }
    .tip-card .tip-label {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #a5b4fc;
        margin-bottom: 0.35rem;
    }
    [data-testid="stTabs"] {
        margin-top: 0.35rem;
    }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.35rem;
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.88), rgba(15, 23, 42, 0.92));
        border-radius: 16px;
        padding: 0.28rem;
        border: 1px solid rgba(129, 140, 248, 0.24);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        height: 3rem;
        justify-content: center;
        border-radius: 12px;
        border: 1px solid transparent;
        color: #cbd5e1;
        font-weight: 700;
        font-size: 1.03rem;
        letter-spacing: -0.01em;
        transition: all 0.18s ease;
        margin: 0 !important;
        padding: 0.2rem 0.65rem;
        background: rgba(30, 41, 59, 0.3);
    }
    [data-testid="stTabs"] [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        border-color: rgba(6, 182, 212, 0.35);
        background: rgba(30, 41, 59, 0.56);
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: linear-gradient(120deg, rgba(124, 58, 237, 0.92), rgba(6, 182, 212, 0.86));
        color: #fff;
        border-color: rgba(255, 255, 255, 0.28);
        box-shadow: 0 8px 24px rgba(76, 29, 149, 0.28);
    }
    div[data-testid="stMetric"] {
        background: var(--surface-strong);
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 0.65rem 0.9rem;
        box-shadow: 0 10px 24px rgba(2, 6, 23, 0.28);
    }
    div[data-testid="stExpander"] {
        background: var(--surface-strong);
        border: 1px solid var(--stroke);
        border-radius: 14px;
    }
    .stAlert {
        border-radius: 14px;
        background: rgba(30, 41, 59, 0.88) !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
        color: #e2e8f0 !important;
    }
    .stAlert [data-testid="stMarkdownContainer"] p {
        color: #e2e8f0 !important;
    }
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(129, 140, 248, 0.35), transparent);
        margin: 1rem 0;
    }
    div[data-testid="stDateInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        background: rgba(30, 41, 59, 0.85) !important;
        color: #f8fbff !important;
        border-color: rgba(129, 140, 248, 0.25) !important;
    }
    /* All Streamlit buttons: fix white-on-white secondary + match theme */
    .stApp .stButton > button {
        border-radius: 999px !important;
        font-weight: 600 !important;
        transition: box-shadow 0.2s ease, transform 0.15s ease, border-color 0.2s ease !important;
    }
    .stApp .stButton > button[kind="secondary"],
    .stApp .stButton > button[data-testid="baseButton-secondary"] {
        background: linear-gradient(180deg, rgba(51, 65, 85, 0.95), rgba(30, 41, 59, 0.98)) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(139, 92, 246, 0.45) !important;
        box-shadow: 0 2px 14px rgba(0, 0, 0, 0.35);
    }
    .stApp .stButton > button[kind="secondary"]:hover,
    .stApp .stButton > button[data-testid="baseButton-secondary"]:hover {
        border-color: rgba(6, 182, 212, 0.65) !important;
        box-shadow: 0 0 22px rgba(139, 92, 246, 0.25);
    }
    .stApp .stButton > button[kind="primary"],
    .stApp .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(120deg, #7c3aed 0%, #06b6d4 100%) !important;
        border: 0 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.35);
    }
    /* Chat composer */
    [data-testid="stChatInput"] {
        background: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(129, 140, 248, 0.3) !important;
        border-radius: 999px !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {
        color: #f1f5f9 !important;
        caret-color: #67e8f9;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #64748b !important;
    }
    /* Chat bubbles — softer surfaces */
    [data-testid="stChatMessage"] {
        background: rgba(15, 23, 42, 0.45) !important;
        border: 1px solid rgba(148, 163, 184, 0.12) !important;
        border-radius: 16px !important;
    }
    div[data-testid="stCaptionContainer"] {
        color: #94a3b8 !important;
    }
    div[data-testid="stCaptionContainer"] > div {
        background: rgba(30, 41, 59, 0.55);
        padding: 0.35rem 0.65rem;
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────
hero_left, hero_right = st.columns([1.55, 1.0], gap="large")
with hero_left:
    st.markdown(
        """
        <div class="hero-card">
            <h2 style="margin:0 0 0.25rem 0;">📈 Alexander Elder AI Command Deck</h2>
            <p style="margin:0; opacity:0.95; font-size: 1.02rem;">
                A cinematic workspace for Elder-style research, strategy testing, and decision support.
            </p>
            <div class="hero-kpis">
                <div class="kpi-pill">💬 RAG chat with cited context</div>
                <div class="kpi-pill">📊 Multi-layer visual backtesting</div>
                <div class="kpi-pill">⚙️ Fast strategy parameter tuning</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    st.markdown(
        f"""
        <div class="portrait-frame">
            <img src="{ELDER_PORTRAIT_URL}" alt="Dr. Alexander Elder" loading="lazy" />
        </div>
        <p class="photo-credit">
            Portrait from Dr. Elder’s official site ·
            <a href="{ELDER_SITE_ABOUT}" target="_blank" rel="noopener noreferrer">elder.com/about</a>
        </p>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="tips-deck">
        <div class="tips-deck-title">📚 Dr. Elder — principles &amp; tips to trade by</div>
        <div class="tips-grid">
            <div class="tip-card">
                <span class="tip-label">Triple Screen</span><br/>
                Use three views: trend, then momentum, then timing — avoid signals that contradict the higher timeframe trend.
            </div>
            <div class="tip-card">
                <span class="tip-label">Risk — 2% Rule</span><br/>
                Cap how much you can lose on a single trade relative to account size so one bad streak cannot sink you.
            </div>
            <div class="tip-card">
                <span class="tip-label">Psychology</span><br/>
                Treat trading as a profession: calm execution, honest review, and rules you follow before emotions kick in.
            </div>
            <div class="tip-card">
                <span class="tip-label">Process</span><br/>
                Journal entries, exits, and rationale — improve the system, not just the next trade’s outcome.
            </div>
            <div class="tip-card">
                <span class="tip-label">Discipline</span><br/>
                Wait for setups that match your plan; skipping mediocre trades often beats chasing every move.
            </div>
            <div class="tip-card">
                <span class="tip-label">Indicators</span><br/>
                Combine tools with clear roles (e.g. trend vs. timing); redundancy without logic adds noise, not edge.
            </div>
        </div>
        <div class="tips-deck-foot">
            Summarized from recurring themes in Dr. Elder’s teaching and books (e.g. risk rules, Triple Screen, and trader psychology).
            For his latest work and programs, see
            <a href="{ELDER_SITE_ABOUT}" target="_blank" rel="noopener noreferrer" style="color:#7dd3fc;">elder.com</a>.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

tab1, tab2 = st.tabs(["💬 Elder Chatbot", "📊 Triple Screen Backtester"])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — CHATBOT
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Ask Alexander Elder</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Get grounded answers on psychology, timing, risk management, '
        'and the Triple Screen method.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-panel">
            <strong>How to get the best answer:</strong> ask one clear trading problem at a time,
            include your context, and then iterate with follow-up prompts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # ── Resolve input: suggestion button OR typed input ─────────────────
    user_input = None

    # Suggested questions — always visible
    suggestions = [
        "Who is Dr. Alexander Elder?",
        "What is the 2% Rule in risk management?",
        "How does the Triple Screen Trading System work?",
        "What role does psychology play in trading according to Elder?",
        "How does Elder use MACD to time trade entries?",
    ]
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="hint-banner">
                <span class="hint-icon">✨</span>
                <span>Kick off with a prompt chip below, or type your own trading question in the composer.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    cols = st.columns(len(suggestions))
    for col, suggestion in zip(cols, suggestions):
        if col.button(suggestion, use_container_width=True, key=f"btn_{suggestion[:20]}"):
            user_input = suggestion

    # Chat input — always rendered so user can type freely
    typed = st.chat_input("Ask about Elder's trading methods...")
    if typed:
        user_input = typed

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                with st.expander("📚 Source context used", expanded=False):
                    for i, src in enumerate(msg["sources"], 1):
                        label = src.metadata.get("label", "?")
                        q     = src.metadata.get("question", "")
                        st.markdown(f"**[{i}] {label}**")
                        st.markdown(f"> *Q: {q}*")
                        st.markdown(src.page_content)
                        if i < len(msg["sources"]):
                            st.divider()

    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving from Elder's knowledge base..."):
                try:
                    answer, sources = rag_query(user_input)
                    st.markdown(answer)
                    with st.expander("📚 Source context used", expanded=False):
                        for i, src in enumerate(sources, 1):
                            label = src.metadata.get("label", "?")
                            q     = src.metadata.get("question", "")
                            st.markdown(f"**[{i}] {label}**")
                            st.markdown(f"> *Q: {q}*")
                            st.markdown(src.page_content)
                            if i < len(sources):
                                st.divider()
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except FileNotFoundError as e:
                    err = (
                        "⚠️ **Index not built yet.**\n\n"
                        "Run this command first:\n\n"
                        "```bash\npython scripts/build_index.py\n```"
                    )
                    st.error(err)
                except EnvironmentError as e:
                    st.error(f"⚠️ **API key missing:** {e}")
                except Exception as e:
                    st.error(f"⚠️ Something went wrong: {e}")

    # Conversation actions
    if st.session_state.messages:
        c_action1, c_action2 = st.columns([1, 6])
        with c_action1:
            if st.button("🗑️ Clear", key="clear_chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with c_action2:
            st.caption("Tip: Ask focused questions to get better source passages.")


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — BACKTESTER
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Elder Triple Screen Backtester</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Design scenarios quickly, tune the strategy parameters, '
        'and inspect entry/exit behavior with clear visuals.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="glass-panel">
            <strong>Pro flow:</strong> lock trend windows first, then tune RSI bounds to balance trade
            frequency and false entries.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar-style controls inside the tab ──────────────────────────
    with st.container(border=True):
        st.markdown("#### Strategy Setup")
        st.caption("Set ticker, date range, capital, and signal thresholds before running.")
        c1, c2, c3 = st.columns(3)

        with c1:
            symbol = st.text_input(
                "Stock ticker", value="SPY",
                help="Any Yahoo Finance ticker: AAPL, MSFT, TSLA, …"
            ).upper().strip()
            initial_capital = st.number_input(
                "Starting capital ($)", min_value=1000, max_value=1_000_000,
                value=10_000, step=1000,
            )

        with c2:
            start_date = st.date_input(
                "Start date", value=date(2020, 1, 1),
                min_value=date(2000, 1, 1), max_value=date.today(),
            )
            end_date = st.date_input(
                "End date", value=date(2025, 12, 31),
                min_value=date(2000, 1, 1), max_value=date(2026, 12, 31),
            )

        with c3:
            win_short = st.slider(
                "EMA short window (win_short)", 5, 50, 15,
                help="Short-term EMA period — default: 15"
            )
            win_long = st.slider(
                "EMA long window (win_long)", 50, 300, 200,
                help="Long-term EMA period — default: 200"
            )

        c4, c5 = st.columns(2)
        with c4:
            rsi_lower = st.slider(
                "RSI lower bound (RSI_lower_bound)", 20, 60, 50,
                help="Long entry / short exit threshold — default: 50"
            )
        with c5:
            rsi_upper = st.slider(
                "RSI upper bound (RSI_upper_bound)", 40, 80, 53,
                help="Short entry / long exit threshold — default: 53"
            )

        run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)

    # ── Run and display results ─────────────────────────────────────────
    if run_btn:
        if start_date >= end_date:
            st.error("Start date must be before end date.")
        else:
            with st.spinner(f"Fetching {symbol} data and running backtest..."):
                result = run_backtest(
                    symbol          = symbol,
                    start           = str(start_date),
                    end             = str(end_date),
                    win_short       = win_short,
                    win_long        = win_long,
                    rsi_lower       = float(rsi_lower),
                    rsi_upper       = float(rsi_upper),
                    initial_capital = float(initial_capital),
                )

            if result["error"]:
                st.error(result["error"])
            else:
                data       = result["data"]
                final_eq   = result["final_equity"]
                tot_ret    = result["total_return"]
                n_long     = result["n_long_trades"]
                n_short    = result["n_short_trades"]

                # ── Metrics row ────────────────────────────────────────
                st.markdown("#### Results")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric(
                    "Final Portfolio",
                    f"${final_eq:,.2f}",
                    f"{tot_ret:+.2f}%",
                    delta_color="normal",
                )
                m2.metric("Starting Capital", f"${initial_capital:,.0f}")
                m3.metric("Long entries",  n_long)
                m4.metric("Short entries", n_short)
                st.caption(
                    f"Scenario: `{symbol}` from `{start_date}` to `{end_date}` · "
                    f"EMA {win_short}/{win_long} · RSI {rsi_lower}/{rsi_upper}"
                )

                # ── Chart 1: Price + signals ───────────────────────────
                fig1 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    subplot_titles=[
                        f"{symbol} Price + EMA + Trade Signals",
                        "MACD",
                    ],
                    vertical_spacing=0.06,
                )

                # Candlestick
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data["Close"],
                    name="Close", line=dict(color="#6B7FD7", width=1.5),
                ), row=1, col=1)

                # EMAs
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data[f"EMA{win_short}"],
                    name=f"EMA{win_short}", line=dict(color="#F4A261", width=1.2, dash="dot"),
                ), row=1, col=1)
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data[f"EMA{win_long}"],
                    name=f"EMA{win_long}", line=dict(color="#E76F51", width=1.5),
                ), row=1, col=1)

                # Buy signals
                if result["buy_dates"]:
                    fig1.add_trace(go.Scatter(
                        x=result["buy_dates"], y=result["buy_prices"],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="#2DC653"),
                        name="Long entry",
                    ), row=1, col=1)

                # Sell signals
                if result["sell_dates"]:
                    fig1.add_trace(go.Scatter(
                        x=result["sell_dates"], y=result["sell_prices"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="#E63946"),
                        name="Short entry",
                    ), row=1, col=1)

                # MACD
                macd_colors = ["#2DC653" if v >= 0 else "#E63946"
                               for v in (data["MACD"] - data["MACD_signal"])]
                fig1.add_trace(go.Bar(
                    x=data.index,
                    y=data["MACD"] - data["MACD_signal"],
                    name="MACD histogram",
                    marker_color=macd_colors,
                    opacity=0.7,
                ), row=2, col=1)
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data["MACD"],
                    name="MACD", line=dict(color="#6B7FD7", width=1),
                ), row=2, col=1)
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data["MACD_signal"],
                    name="Signal", line=dict(color="#F4A261", width=1, dash="dot"),
                ), row=2, col=1)

                fig1.update_layout(
                    height=550,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0.85)",
                )
                st.plotly_chart(fig1, use_container_width=True)

                # ── Chart 2: PnL equity curve + RSI ───────────────────
                fig2 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.65, 0.35],
                    subplot_titles=[
                        f"Portfolio Value (PnL) — {symbol}",
                        "RSI(14)",
                    ],
                    vertical_spacing=0.06,
                )

                # Equity curve
                fig2.add_trace(go.Scatter(
                    x=data.index, y=data["Equity"],
                    fill="tozeroy",
                    fillcolor="rgba(107,127,215,0.15)",
                    line=dict(color="#6B7FD7", width=2),
                    name="Portfolio value ($)",
                ), row=1, col=1)

                # Baseline
                fig2.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="#888",
                    annotation_text=f"Starting capital: ${initial_capital:,.0f}",
                    row=1, col=1,
                )

                # RSI
                fig2.add_trace(go.Scatter(
                    x=data.index, y=data["RSI"],
                    name="RSI(14)", line=dict(color="#F4A261", width=1.5),
                ), row=2, col=1)
                fig2.add_hline(y=rsi_upper, line_dash="dot", line_color="#E63946",
                               annotation_text=f"RSI upper ({rsi_upper})", row=2, col=1)
                fig2.add_hline(y=rsi_lower, line_dash="dot", line_color="#2DC653",
                               annotation_text=f"RSI lower ({rsi_lower})", row=2, col=1)
                fig2.update_yaxes(range=[0, 100], row=2, col=1)

                fig2.update_layout(
                    height=500,
                    margin=dict(l=0, r=0, t=40, b=0),
                    hovermode="x unified",
                    showlegend=True,
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0.85)",
                )
                st.plotly_chart(fig2, use_container_width=True)

                # ── Raw data expander ──────────────────────────────────
                with st.expander("🔍 View raw indicator data", expanded=False):
                    cols_show = [
                        "Close", f"EMA{win_short}", f"EMA{win_long}",
                        "MACD", "MACD_signal", "RSI", "Equity",
                    ]
                    st.dataframe(
                        data[[c for c in cols_show if c in data.columns]]
                        .tail(100)
                        .style.format("{:.2f}"),
                        use_container_width=True,
                    )
    else:
        st.markdown(
            """
            <div class="hint-banner">
                <span class="hint-icon">📊</span>
                <span><strong>Ready to run.</strong> Set ticker and dates, tune EMA/RSI, then hit <strong>Run Backtest</strong>.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )