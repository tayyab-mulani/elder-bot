# Methodology

This project is grounded in the trading framework of **Dr. Alexander Elder**, a professional trader and psychiatrist whose work — *Trading for a Living* (1993), *Come Into My Trading Room* (2002), and *The New Trading for a Living* (2014) — is built around a simple premise: successful trading rests on three pillars, the **Three M's**: **Mind, Method, and Money**.

## The Three M's — and how this project maps to them

| Pillar | Elder's Definition | How this project reflects it |
|---|---|---|
| **Mind** | Trading psychology — discipline, emotional control, handling fear and greed, avoiding the crowd | Covered in the RAG knowledge base via the *Psychology* (300 Q&A) and *Adaptability* (149 Q&A) labels. Users can ask the chatbot about mindset, emotional pitfalls, and discipline. |
| **Method** | A repeatable system for identifying trades using charts, indicators, and multi-timeframe analysis | Implemented as the **Triple Screen Trading System** in the backtester, and covered in the *Timing* label (220 Q&A) of the dataset. |
| **Money** | Risk management rules that keep a trader in the game through losing streaks | Covered in the *Risk Management* label (200 Q&A) and reflected in the backtester's capital-tracking logic. |

Together, the chatbot teaches the *why* of Elder's philosophy, and the backtester lets users test the *how* on real market data.

## The Triple Screen Trading System

Elder's signature methodology, introduced in *Trading for a Living*, was designed to solve a well-known problem in technical analysis: trend-following indicators fail in range-bound markets, while oscillators fail in trending ones. The Triple Screen combines both, filtered across multiple time horizons that Elder calls the **tide** (long-term trend), the **wave** (medium-term pullback), and the **ripple** (short-term entry).

This project's backtester implements a single-timeframe adaptation of that logic — see [`BACKTESTER.md`](BACKTESTER.md) for implementation details.

## Why RAG instead of a from-scratch Transformer

The project guidelines allowed either a hand-coded Transformer or a RAG framework. RAG was the right fit here for three reasons:

1. **Faithfulness to source material.** Elder's ideas are specific and often counterintuitive (e.g. his views on stop placement, the 2% rule, the Impulse System). A retrieval-grounded model quotes his actual positions rather than hallucinating generic trading advice.
2. **Auditability.** Every answer in the UI shows the retrieved source passages, so users can verify the chatbot isn't making things up.
3. **Dataset leverage.** The 1,063 curated Q&A pairs serve directly as the retrieval corpus — no training loop needed, and adding new material is just a re-indexing step.

## Dataset design

The knowledge base is organized around five labels that map to Elder's framework:

- **Psychology** (300) — the "Mind" pillar
- **Timing** (220) — the "Method" pillar, entry/exit execution
- **Risk Management** (200) — the "Money" pillar
- **Personal Life** (194) — Elder's background as psychiatrist-turned-trader, grounding the *why* behind his psychology-first approach
- **Adaptability** (149) — how Elder's methods evolved from 1993 to the 2014 revision

The dataset was prepared as structured Q&A pairs (matching the `Wyckoff_sample.csv` format from the course), with the answer field serving as the retrieval corpus for the RAG pipeline.

## RAG pipeline

1. User question is embedded with `paraphrase-MiniLM-L6-v2`
2. Top-4 semantically similar answers are retrieved from Chroma
3. Retrieved passages are injected into a prompt with their topic labels
4. Groq (`llama-3.1-8b-instant`, temperature 0.3) generates the final answer
5. Source passages are shown in a collapsible expander below each response

Models are loaded once as module-level singletons to avoid reloading on every Streamlit rerun.

## References

- Elder, A. (1993). *Trading for a Living: Psychology, Trading Tactics, Money Management*. John Wiley & Sons.
- Elder, A. (2014). *The New Trading for a Living: Psychology, Discipline, Trading Tools and Systems, Risk Control, Trade Management*. John Wiley & Sons.
- [Trading Strategy Guides — Alexander Elder Trading Strategy: The Triple Screen](https://tradingstrategyguides.com/alexander-elder-trading-strategy-the-triple-screen/) — tide/wave/ripple walkthrough
- [Admiral Markets — What is the Triple Screen Trading System?](https://admiralmarkets.com/education/articles/forex-strategy/triple-screen-trading-system) — concise system summary
- [QuantifiedStrategies — Triple Screen Strategy (with backtest)](https://www.quantifiedstrategies.com/alexander-elder-triple-screen-strategy/) — implementation and empirical testing