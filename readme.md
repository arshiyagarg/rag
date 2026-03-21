# Recurse — DSA Hint Engine

> A RAG-powered DSA tutor that gives hints and targeted code fixes — not full giveaways.

Paste a LeetCode-style problem and your stuck code. Recurse retrieves relevant Stack Overflow answers and CP-Algorithms articles, reranks them by true relevance, and generates a structured hint or corrected code grounded in real sources — with citations.

---

## Demo

**Hint mode** — "What's wrong with my sliding window?"
```
🔍 Pattern identified: Sliding Window
💭 Why this pattern: Your nested loop recomputes the substring set from scratch on every iteration...
🛠 What's wrong: The window never shrinks — you're missing the left pointer advancement when a duplicate is found [SOURCE: stackoverflow.com/a/...]
📖 Key concept: Two-pointer technique with a character frequency map
```

**Code fix mode** — "Fix my Dijkstra"
```
1. Pattern identified: Dijkstra's algorithm
2. What was wrong: Missing stale distance check after pq.pop() — outdated distances re-relax neighbors
3. Corrected code (C++): if(currDist > dist[currNode]) continue; // add this line
4. Why this fixes it: Ensures each node is only processed at its minimum known distance [SOURCE: cp-algorithms.com/graph/dijkstra_sparse.html]
```

---

## Architecture

```
User query
    │
    ▼
Gemini text-embedding-004     embed query (RETRIEVAL_QUERY task type)
    │
    ▼
Pinecone vector search        top-20 by cosine similarity
    │
    ▼
Jina reranker v2              rescore by true relevance → top-5
    │
    ▼
Groq llama-3.3-70b            stream structured hint with citations
    │
    ▼
Streamlit UI                  real-time streamed response + sources panel
```

---

## Data sources

| Source | Content |
|---|---|
| **Stack Overflow** | Top-voted DSA Q&A, accepted answers, vote scores | 
| **CP-Algorithms** | 87 DSA articles across 6 topic folders | 

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| Embeddings | Gemini `text-embedding-004` | Free tier, 768-dim, stable |
| Generation | Groq `llama-3.3-70b-versatile` | Free tier, fast LPU inference |
| Reranker | Jina `jina-reranker-v2-base-multilingual` | Free 10M tokens, beats cross-encoder on code |
| Vector DB | Pinecone Serverless | Free starter, metadata filtering |
| Chunking | Chonkie `SemanticChunker` | Semantic boundaries, code block protection |
| UI | Streamlit | Single-file, streaming compatible |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/yourname/ohint
cd ohint
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

| Key | Where to get it | Required |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | ✅ |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) | ✅ |
| `PINECONE_API_KEY` | [app.pinecone.io](https://app.pinecone.io) | ✅ |
| `JINA_API_KEY` | [jina.ai](https://jina.ai) | ✅ |
| `GITHUB_TOKEN` | [github.com/settings/tokens](https://github.com/settings/tokens) | Optional (higher rate limit) |
| `STACKAPPS_KEY` | [stackapps.com/apps/oauth/register](https://stackapps.com/apps/oauth/register) | Optional (higher rate limit) |

All four required keys are **free with no credit card**.

### 3. Ingest data

```bash
# Crawl, chunk, embed, and upsert to Pinecone
python ingest.py --source cp_algorithms
python ingest.py --source stackoverflow
```

> **Note:** Gemini free tier allows 1500 embed requests/day. The ingester automatically skips already-indexed chunks on re-runs — no quota wasted.

### 4. Run

```bash
# Streamlit UI
streamlit run app.py

# CLI interactive mode
python -m src.pipeline

# Single question
python -m src.pipeline --problem "two sum" --code "..."
```

---

## Usage

### Hint mode (default)
Just describe the problem — no keywords needed:
```
Problem: Given an array, find two numbers that sum to target
Code: [your O(n²) brute force]
```
O(hint) identifies the pattern, explains why, and tells you what's conceptually wrong — without writing the solution.

### Code fix mode
Include words like `fix`, `correct`, `what's wrong`, `debug`:
```
Problem: What's wrong with my Dijkstra?
Code: [your C++ implementation]
```
O(hint) identifies the specific bug and returns minimal corrected code in the same language — grounded in retrieved sources, not hallucinated.

---

## Evaluation

```bash
python run_eval.py
```
---
