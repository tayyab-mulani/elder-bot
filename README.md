# Alexander Elder AI Assistant

RAG chatbot and Triple Screen backtester built on Alexander Elder's trading methodology.

> CSYE 7380 — GenAI Final Project

---

## What's inside

**Tab 1 — Chatbot.** Ask questions about Elder's trading philosophy. Answers are grounded in a 1,063-row curated Q&A dataset via RAG and show the source passages used. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for how the dataset maps to Elder's Three M's framework.

**Tab 2 — Backtester.** Run Elder's Triple Screen strategy (EMA + MACD + RSI) on any US ticker. Reports Sharpe, CAGR, max drawdown, win rate, profit factor, and a buy-and-hold benchmark. See [`docs/BACKTESTER.md`](docs/BACKTESTER.md) for metric definitions and how to read them.

---

## Quick start

```bash
# 1. Install
git clone https://github.com/tayyab-mulani/elder-bot.git && cd elder-bot
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure (free key at https://console.groq.com)
cp .env.example .env                                 # then edit .env

# 3. Build the index (one-off, takes ~1 min)
python scripts/preprocess.py                         # merge & clean raw CSVs
python scripts/build_index.py                        # embed into Chroma

# 4. Run
python -m streamlit run app/main.py
```

---

## Tech stack

Streamlit UI · LangChain + Chroma vector store · `paraphrase-MiniLM-L6-v2` embeddings · Groq `llama-3.1-8b-instant` · yfinance + Plotly

---

## Dataset

Structured Q&A pairs across five labels aligned to Elder's framework:

| Label | Rows | Pillar |
|---|---|---|
| Psychology | 300 | Mind |
| Timing | 220 | Method |
| Risk Management | 200 | Money |
| Personal Life | 194 | Context |
| Adaptability | 149 | Mind |
| **Total** | **1,063** | |

Raw files live in `data/raw/`. Running `scripts/preprocess.py` merges and cleans them into `data/elder_qa_master.csv`.

---

## Project structure

```
elder-bot/
├── app/main.py                  ← Streamlit app
├── models/
│   ├── rag_pipeline.py          ← embed → retrieve → Groq generate
│   ├── backtest_engine.py       ← Triple Screen logic + metrics
│   └── chroma_db/               ← persisted vector index
├── scripts/
│   ├── preprocess.py            ← clean & merge raw CSVs
│   └── build_index.py           ← embed & index into Chroma
├── data/raw/                    ← input CSVs per label
├── docs/
│   ├── METHODOLOGY.md           ← Elder framework & dataset design
│   └── BACKTESTER.md            ← strategy, metrics, and how to read them
└── requirements.txt
```

---

## Further reading

- **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)** — Elder's Three M's framework, Triple Screen theory, and why we chose RAG over a from-scratch Transformer
- **[`docs/BACKTESTER.md`](docs/BACKTESTER.md)** — strategy rules, performance metric definitions, and the position-sizing design note that affects how metrics should be interpreted