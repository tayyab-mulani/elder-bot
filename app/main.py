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

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alexander Elder AI Assistant",
    page_icon="📈",
    layout="wide",
)

# ── Header ─────────────────────────────────────────────────────────────────
st.title("📈 Alexander Elder AI Assistant")
st.caption("RAG-powered chatbot · Triple Screen backtester · CSYE 7380 Final Project")
st.divider()

tab1, tab2 = st.tabs(["💬 Elder Chatbot", "📊 Triple Screen Backtester"])


# ══════════════════════════════════════════════════════════════════════════
#  TAB 1 — CHATBOT
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ask Alexander Elder")
    st.markdown(
        "Ask anything about Elder's trading philosophy, psychology, risk rules, "
        "strategy, or personal background."
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
        st.markdown("**Try asking:**")
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

    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
#  TAB 2 — BACKTESTER
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Elder Triple Screen Backtester")
    st.markdown(
        "Backtest Elder's Triple Screen strategy on any US stock. "
        "The strategy combines **EMA trend** (Screen 1), "
        "**MACD momentum** (Screen 2), and **RSI timing** (Screen 3)."
    )

    # ── Sidebar-style controls inside the tab ──────────────────────────
    with st.container(border=True):
        st.markdown("#### Strategy Parameters")
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