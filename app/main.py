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
 
# ── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base font size boost ── */
html, body, [class*="css"] {
    font-size: 16px !important;
}
 
/* ── Input labels (Stock ticker, Start date, etc.) ── */
label[data-testid="stWidgetLabel"] p,
label[data-testid="stWidgetLabel"],
.stTextInput label,
.stDateInput label,
.stNumberInput label,
.stSlider label {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #D4AF37 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 4px !important;
}
 
/* ── Slider value label (the number shown above thumb) ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"],
div[data-testid="stSlider"] p {
    font-size: 14px !important;
    color: #D4AF37 !important;
}
 
/* ── Text input & date input fields ── */
.stTextInput input,
.stDateInput input,
.stNumberInput input {
    font-size: 16px !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    background-color: #1a1a2e !important;
    color: #ffffff !important;
    border: 1px solid #D4AF37 !important;
}
 
/* ── Chat input box ── */
.stChatInput textarea,
div[data-testid="stChatInput"] textarea {
    font-size: 16px !important;
    min-height: 56px !important;
    padding: 14px 16px !important;
    border-radius: 8px !important;
    background-color: #1a1a2e !important;
    color: #ffffff !important;
    border: 1px solid #D4AF37 !important;
}
 
div[data-testid="stChatInput"] {
    border: 1px solid #D4AF37 !important;
    border-radius: 8px !important;
    background-color: #1a1a2e !important;
}
 
/* ── Chat messages ── */
.stChatMessage p,
div[data-testid="stChatMessageContent"] p {
    font-size: 16px !important;
    line-height: 1.7 !important;
}
 
/* ── Suggestion buttons ── */
.stButton button {
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    border: 1px solid #D4AF37 !important;
    color: #D4AF37 !important;
    background-color: transparent !important;
    transition: all 0.2s ease !important;
    white-space: normal !important;
    text-align: center !important;
    min-height: 52px !important;
}
 
.stButton button:hover {
    background-color: #D4AF37 !important;
    color: #0a0a0a !important;
}
 
/* ── Run Backtest primary button ── */
button[kind="primary"] {
    background-color: #D4AF37 !important;
    color: #0a0a0a !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    border: none !important;
    border-radius: 6px !important;
}
 
/* ── Metric cards ── */
div[data-testid="stMetric"] label {
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #999 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
 
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}
 
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 14px !important;
}
 
/* ── Section headings ── */
h1 { font-size: 2.2rem !important; color: #D4AF37 !important; }
h2 { font-size: 1.8rem !important; color: #D4AF37 !important; }
h3 { font-size: 1.4rem !important; color: #D4AF37 !important; }
h4 { font-size: 1.2rem !important; color: #D4AF37 !important; }
h5 { font-size: 1.05rem !important; color: #D4AF37 !important; }
 
/* ── Tab labels ── */
.stTabs [data-baseweb="tab"] {
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 12px 20px !important;
    letter-spacing: 0.05em !important;
}
 
/* ── Container / card borders ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #D4AF37 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}
 
/* ── Expander ── */
details summary p,
.streamlit-expanderHeader p {
    font-size: 15px !important;
    font-weight: 600 !important;
}
 
/* ── Markdown body text ── */
.stMarkdown p {
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: #cccccc !important;
}
 
/* ── Caption / tag badges under title ── */
.stCaption p {
    font-size: 14px !important;
    color: #999 !important;
}
 
/* ── Dataframe ── */
.stDataFrame {
    font-size: 14px !important;
}
 
/* ── Number input +/- buttons ── */
.stNumberInput button {
    font-size: 18px !important;
    font-weight: 700 !important;
    min-height: 36px !important;
}
 
/* ── Slider thumb label ── */
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
    font-size: 14px !important;
    color: #D4AF37 !important;
}
</style>
""", unsafe_allow_html=True)
 
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
        if col.button(suggestion, width="stretch", key=f"btn_{suggestion[:20]}"):
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
 
        run_btn = st.button("▶ Run Backtest", type="primary", width="stretch")
 
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
                data     = result["data"]
                final_eq = result["final_equity"]
                n_long   = result["n_long_trades"]
                n_short  = result["n_short_trades"]
                m        = result["metrics"]
 
                # ── Performance metrics ────────────────────────────────
                st.markdown("#### 📊 Performance Metrics")
 
                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                r1c1.metric(
                    "Final Portfolio",
                    f"${final_eq:,.2f}",
                    f"{m['total_return']:+.2f}%",
                )
                r1c2.metric("Total Return", f"{m['total_return']:+.2f}%")
                r1c3.metric(
                    "CAGR", f"{m['cagr']:+.2f}%",
                    help="Compound Annual Growth Rate — annualized return.",
                )
                r1c4.metric(
                    "Sharpe Ratio", f"{m['sharpe']:.2f}",
                    help="Annualized, risk-free rate = 0. >1 is good, >2 is very good.",
                )
 
                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                r2c1.metric(
                    "Max Drawdown", f"{m['max_drawdown']:.2f}%",
                    help="Largest peak-to-trough equity decline. Closer to 0 is better.",
                )
                r2c2.metric(
                    "Win Rate", f"{m['win_rate']:.1f}%",
                    f"{m['n_wins']} / {m['n_trades']} trades",
                )
                r2c3.metric(
                    "Profit Factor",
                    f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "∞",
                    help="Gross wins ÷ gross losses. >1 means profitable overall.",
                )
                r2c4.metric(
                    "Total Trades", f"{m['n_trades']}",
                    f"{n_long} long / {n_short} short entries",
                    delta_color="off",
                )
 
                st.markdown("##### vs. Buy & Hold Benchmark")
                b1, b2, b3 = st.columns(3)
                b1.metric("Buy & Hold Final", f"${m['bh_final']:,.2f}", f"{m['bh_return']:+.2f}%")
                b2.metric("Strategy Final", f"${final_eq:,.2f}", f"{m['total_return']:+.2f}%")
                b3.metric(
                    "Alpha (Strategy − B&H)", f"{m['alpha']:+.2f}%",
                    help="Positive = strategy beat buy-and-hold.",
                )
 
                # ── Chart 1: Price + signals ───────────────────────────
                fig1 = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    subplot_titles=[f"{symbol} Price + EMA + Trade Signals", "MACD"],
                    vertical_spacing=0.06,
                )
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data["Close"],
                    name="Close", line=dict(color="#6B7FD7", width=1.5),
                ), row=1, col=1)
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data[f"EMA{win_short}"],
                    name=f"EMA{win_short}", line=dict(color="#F4A261", width=1.2, dash="dot"),
                ), row=1, col=1)
                fig1.add_trace(go.Scatter(
                    x=data.index, y=data[f"EMA{win_long}"],
                    name=f"EMA{win_long}", line=dict(color="#E76F51", width=1.5),
                ), row=1, col=1)
                if result["buy_dates"]:
                    fig1.add_trace(go.Scatter(
                        x=result["buy_dates"], y=result["buy_prices"], mode="markers",
                        marker=dict(symbol="triangle-up", size=10, color="#2DC653"),
                        name="Long entry",
                    ), row=1, col=1)
                if result["sell_dates"]:
                    fig1.add_trace(go.Scatter(
                        x=result["sell_dates"], y=result["sell_prices"], mode="markers",
                        marker=dict(symbol="triangle-down", size=10, color="#E63946"),
                        name="Short entry",
                    ), row=1, col=1)
                macd_colors = ["#2DC653" if v >= 0 else "#E63946"
                               for v in (data["MACD"] - data["MACD_signal"])]
                fig1.add_trace(go.Bar(
                    x=data.index, y=data["MACD"] - data["MACD_signal"],
                    name="MACD histogram", marker_color=macd_colors, opacity=0.7,
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
                    height=550, margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                )
                st.plotly_chart(fig1, width="stretch")
 
                # ── Chart 2: PnL equity curve + RSI ───────────────────
                fig2 = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35],
                    subplot_titles=[f"Portfolio Value (PnL) — {symbol}", "RSI(14)"],
                    vertical_spacing=0.06,
                )
                fig2.add_trace(go.Scatter(
                    x=data.index, y=data["Equity"],
                    fill="tozeroy", fillcolor="rgba(107,127,215,0.15)",
                    line=dict(color="#6B7FD7", width=2),
                    name="Portfolio value ($)",
                ), row=1, col=1)
                fig2.add_hline(
                    y=initial_capital, line_dash="dash", line_color="#888",
                    annotation_text=f"Starting capital: ${initial_capital:,.0f}",
                    row=1, col=1,
                )
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
                    height=500, margin=dict(l=0, r=0, t=40, b=0),
                    hovermode="x unified", showlegend=True,
                )
                st.plotly_chart(fig2, width="stretch")
 
                # ── Trade log ──────────────────────────────────────────
                if result["trade_log"]:
                    with st.expander(
                        f"📋 Trade Log — {len(result['trade_log'])} closed trades",
                        expanded=False,
                    ):
                        trade_df = pd.DataFrame(result["trade_log"])
                        trade_df["entry_price"] = trade_df["entry_price"].round(2)
                        trade_df["exit_price"]  = trade_df["exit_price"].round(2)
                        trade_df["pnl"]         = trade_df["pnl"].round(2)
                        trade_df["result"]      = trade_df["pnl"].apply(
                            lambda x: "✅ Win" if x > 0 else "❌ Loss"
                        )
                        st.dataframe(
                            trade_df[["side", "entry_price", "exit_price", "pnl", "result"]],
                            width="stretch",
                        )
 
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
                        width="stretch",
                    )