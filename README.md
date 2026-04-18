# Alexander Elder AI Assistant

RAG-powered chatbot and Triple Screen backtesting dashboard built on Alexander Elder's trading methodology.

> CSYE 7380 — GenAI Final Project

---

## Features

**Tab 1 — Elder Chatbot**
Ask questions about Elder's trading philosophy — risk management, psychology, timing, adaptability, and personal background. Answers are grounded in a 664-row curated Q&A dataset via RAG (retrieve → context-inject → generate). Each response shows the source passages it was derived from.

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

### 3. Add raw CSV files

Place all raw label CSV files into `data/raw/`:

```
data/raw/
├── Label3_RudrangGade.csv
├── Elder_Psychology.csv
├── Elder_Personal_life_2.csv
├── Elder_Label4_RiskManagement_200QA.csv
└── Elder_Adaptability.csv
```

### 4. Preprocess the dataset (Step 1 of 2)

```bash
python scripts/preprocess.py
```

Loads all raw CSVs from `data/raw/`, standardises column names, cleans text, removes duplicates and invalid rows, and saves the unified dataset to `data/elder_qa_master.csv`.

Expected output:
```
[OK]  Label3_RudrangGade.csv                → 150 rows
[OK]  Elder_Psychology.csv                  → 101 rows
[OK]  Elder_Personal_life_2.csv             → 194 rows
[OK]  Elder_Label4_RiskManagement_200QA.csv → 200 rows
[OK]  Elder_Adaptability.csv                → 149 rows

Total rows saved  : 664
Duplicates removed: 130
```

Re-run this step whenever teammates add or update their CSV files.

### 5. Build the vector index (Step 2 of 2)

```bash
python scripts/build_index.py
```

Embeds all 664 Q&A answers using `paraphrase-MiniLM-L6-v2` and persists the Chroma index to `models/chroma_db/`. Re-run this whenever `elder_qa_master.csv` is regenerated.

### 6. Launch the app

```bash
python -m streamlit run app/main.py
```

Open http://localhost:8501 in your browser.

---

## Project Structure

```
elder-bot/
├── app/
│   └── main.py                      ← Streamlit app (Tab 1: chatbot, Tab 2: backtester)
├── data/
│   ├── raw/                         ← original CSV files per label (input to preprocess.py)
│   │   ├── Label3_RudrangGade.csv
│   │   ├── Elder_Psychology.csv
│   │   ├── Elder_Personal_life_2.csv
│   │   ├── Elder_Label4_RiskManagement_200QA.csv
│   │   └── Elder_Adaptability.csv
│   └── elder_qa_master.csv          ← cleaned & merged dataset (output of preprocess.py)
├── models/
│   ├── rag_pipeline.py              ← RAG engine: embed → retrieve top-4 → Groq generate
│   ├── backtest_engine.py           ← Elder Triple Screen logic (EMA + MACD + RSI)
│   └── chroma_db/                   ← persisted vector index (output of build_index.py)
├── scripts/
│   ├── preprocess.py                ← Step 1: clean & merge raw CSVs
│   └── build_index.py               ← Step 2: embed & index into Chroma
├── .env.example
├── requirements.txt
└── README.md
```

---

## Data Pipeline

```
data/raw/*.csv
      │
      ▼
python scripts/preprocess.py      ← clean, deduplicate, merge
      │
      ▼
data/elder_qa_master.csv
      │
      ▼
python scripts/build_index.py     ← embed with paraphrase-MiniLM-L6-v2, index into Chroma
      │
      ▼
models/chroma_db/                  ← ready for RAG queries
```

---

## Dataset

| Label | Rows |
|---|---|
| Risk Management | 200 |
| Personal Life | 194 |
| Adaptability | 149 |
| Psychology | 101 |
| Timing | 20 |
| **Total** | **664** |

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