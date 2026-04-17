# Alexander Elder AI Assistant

RAG-powered chatbot and Triple Screen backtesting dashboard built on Alexander Elder's trading methodology.

> CSYE 7380 — GenAI Final Project

---

## Features

**Tab 1 — Elder Chatbot**
Ask questions about Elder's trading philosophy — risk management, psychology, timing, adaptability, and personal background. Answers are grounded in a 796-row curated Q&A dataset via RAG (retrieve → context-inject → generate). Each response shows the source passages it was derived from.

**Tab 2 — Triple Screen Backtester**
Select any US ticker, date range, starting capital, and strategy parameters. The engine runs Elder's three-screen system (EMA trend → MACD momentum → RSI timing), plots price + signals + MACD and the equity curve + RSI, and reports final portfolio value and trade counts.

---

## Tech Stack

| Component | Tool |
|---|---|
| Embedding model | `paraphrase-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | Chroma — `langchain-chroma` / `langchain-community` |
| LLM | Groq — `llama-3.1-8b-instant` (free API) |
| RAG framework | LangChain |
| Market data | yfinance |
| Charts | Plotly |
| UI | Streamlit |

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd elder-bot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB) and caches it locally.

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```
GROQ_API_KEY=your_groq_api_key_here        # required — free at https://console.groq.com
HF_TOKEN=your_huggingface_token_here       # optional — suppresses rate-limit warnings
```

Get a free Groq key at https://console.groq.com  
Get a free HuggingFace token at https://huggingface.co/settings/tokens

### 3. Build the vector index (once)

```bash
python scripts/build_index.py
```

Embeds all 796 Q&A answers using `paraphrase-MiniLM-L6-v2` and persists the Chroma index to `models/chroma_db/`. Re-run this whenever you add rows to the CSV.

### 4. Launch the app

```bash
streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
elder-bot/
├── app/
│   └── main.py                 ← Streamlit app (Tab 1: chatbot, Tab 2: backtester)
├── data/
│   └── elder_qa_master.csv     ← 796 Q&A pairs labeled by topic
├── models/
│   ├── rag_pipeline.py         ← RAG engine: embed → retrieve top-4 → Groq generate
│   ├── backtest_engine.py      ← Elder Triple Screen logic (EMA + MACD + RSI)
│   └── chroma_db/              ← persisted vector index (created by build_index.py)
├── scripts/
│   └── build_index.py          ← one-time index builder
├── .env.example
├── requirements.txt
└── README.md
```

---

## Dataset

| Label | Rows |
|---|---|
| Risk Management | 200 |
| Personal Life | 196 |
| Adaptability | 149 |
| Timing | 150 |
| Psychology | 101 |
| **Total** | **796** |

---

## Trading Strategy

The backtester implements Elder's **Triple Screen Trading System**:

| Screen | Indicator | Condition | Role |
|---|---|---|---|
| 1 — Trend | EMA | `EMA(short) > EMA(long)` | Uptrend filter |
| 2 — Momentum | MACD | `MACD > MACD signal` | Trend confirmation |
| 3 — Timing | RSI(14) | `RSI < RSI_lower` | Pullback entry |

**Long entry:** all three screens align (uptrend + MACD up + RSI oversold)  
**Short entry:** all three screens flip (downtrend + MACD down + RSI overbought)  
**Exit long:** RSI crosses above `RSI_upper`  
**Exit short:** RSI crosses below `RSI_lower`

**Default parameters:** `win_short=15`, `win_long=200`, `RSI_lower=50`, `RSI_upper=53`, `capital=$10,000`

All parameters are adjustable in the UI before running a backtest.

---

## RAG Pipeline

1. User question is embedded with `paraphrase-MiniLM-L6-v2`
2. Top-4 semantically similar answers are retrieved from Chroma
3. Retrieved passages are injected into a prompt with topic labels
4. Groq (`llama-3.1-8b-instant`, temperature 0.3) generates the final answer
5. Source passages are shown in a collapsible expander below each response

Models are loaded once as module-level singletons to avoid reloading on every Streamlit rerun.
