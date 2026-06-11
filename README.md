# RepoMind

A fully local codebase intelligence tool. Paste a GitHub URL, ask natural language questions about the codebase, and get answers grounded in actual source code with file:line citations, including structural questions like "what calls X?" and "what breaks if I change X?"

Everything runs locally via [Ollama](https://ollama.com), no source code is sent to any external API.

---

## Demo

*[2-minute Loom demo link, coming soon]*

Example query through the UI:

> **Q: what calls add_url_rule?**
>
> Blast radius for `add_url_rule`
>
> Defined in:
> - sample/flask/src/flask/sansio/blueprints.py:412
> - sample/flask/src/flask/sansio/app.py:604
> - sample/flask/src/flask/sansio/scaffold.py:367
>
> Source code callers (4):
> - add_url_rule in sample/flask/src/flask/sansio/blueprints.py:412
> - route in sample/flask/src/flask/sansio/scaffold.py:335
> - decorator in sample/flask/src/flask/sansio/scaffold.py:634
> - create_app in sample/flask/examples/tutorial/flaskr/__init__.py:5
>
> Test coverage (31 tests reference this function)
>
> Changing this function's signature could affect 4 caller(s) listed above.

---

## Architecture

```
User question
    |
    v
Query Router (regex-based)
    |
    +-- Structural query ("what calls X", "what breaks if I change X")
    |       |
    |       v
    |   NetworkX call graph -> blast_radius()
    |       |
    |       v
    |   Source callers + test callers, file:line citations
    |
    +-- Conceptual / lookup query ("how does X work")
            |
            v
        Hybrid Retrieval
            |
            +-- Code-aware BM25 (snake_case/camelCase tokenizer, source files only)
            +-- Semantic search (ChromaDB + nomic-embed-text)
            |
            v
        Reciprocal Rank Fusion (semantic weight 2.0, BM25 weight 1.0)
            |
            v
        Custom iterative agent (llama3.2:3b via Ollama)
            |
            v
        Answer with file:line citations
```

Backend: FastAPI (`/index`, `/ask`)
Frontend: React (paste GitHub URL -> index -> ask questions)

---

## Tech Stack

- **tree-sitter** (Python + JavaScript): AST-based chunking at function/class boundaries
- **ChromaDB**: persistent local vector store
- **Ollama**:`nomic-embed-text` for embeddings, `llama3.2:3b` for generation, fully local
- **rank_bm25**: keyword retrieval with a custom code-aware tokenizer
- **NetworkX**: call graph for structural queries
- **FastAPI** + **React**: backend API and frontend UI

A custom iterative agent loop is used instead of LangChain's `AgentExecutor`. The 3B parameter model couldn't reliably follow the ReAct prompt format required by `AgentExecutor`, so the agent loop (search -> evaluate -> decide) is implemented directly.

---

## Evaluation

A 60-question benchmark was built against the [Flask](https://github.com/pallets/flask) repository, measuring Mean Reciprocal Rank (MRR), Precision@3, and Hit Rate. Ground truth includes an `also_valid` field for questions where Flask's class hierarchy makes multiple files architecturally correct answers (e.g. `app.py` vs `sansio/app.py`).

### Ablation Results

| System | MRR | Precision@3 | Hit Rate | vs Naive RAG |
|---|---|---|---|---|
| Naive RAG (fixed-size chunking + semantic search) | 0.668 | 46.1% | 95.0% | — |
| AST chunking + semantic search | 0.683 | 46.7% | 91.7% | +2.1% |
| **Hybrid: BM25 + semantic + Reciprocal Rank Fusion** | **0.732** | **37.2%** | **100%** | **+9.6%** |
| + Cross-encoder reranker (TinyBERT) | 0.481 | 25.6% | 80.0% | -27.2% |
| + Cross-encoder reranker (MiniLM, full chunks) | 0.546 | 28.3% | 95.0% | -17.4% |
| + Cross-encoder reranker (MiniLM, truncated) | 0.308 | 16.7% | 80.0% | -54.0% |

**The hybrid BM25 + semantic + RRF system is the production configuration**, improving MRR by 9.6% over a naive RAG baseline with 100% hit rate.

### Finding 1: Ground truth quality dominated early results

The first end-to-end run scored MRR 0.484. Rather than immediately changing the retrieval architecture, every zero-score question was manually inspected. Most "failures" were the system correctly retrieving an architecturally valid file that the evaluation script didn't recognize as correct, for example, Flask splits its `Flask` class across `app.py` and `sansio/app.py` via inheritance, and many questions about application-level behavior have correct answers in either file.

Adding an `also_valid` field to the evaluation set and fixing the matching logic took MRR from **0.484 to 0.671 (+39%)** without changing a single line of retrieval code. This was the single highest-impact change in the project.

### Finding 2: Test files pollute lexical retrieval

BM25 alone scored 0.588, worse than naive semantic search. The cause: test files reference every function name repeatedly (`test_redirect`, `test_redirect_with_code`, etc.), so BM25 consistently ranked test files above implementations. Excluding test files from the BM25 index (while keeping them in semantic search) was necessary for the hybrid system to outperform the baseline at all, without this fix, hybrid scored 0.536, *below* naive RAG.

### Finding 3: Cross-encoder reranking degraded results across three configurations

Three reranker configurations were tested on top of the hybrid pipeline:

1. `cross-encoder/ms-marco-TinyBERT-L-2-v2`: MRR dropped to 0.481 (-27.2%)
2. `cross-encoder/ms-marco-MiniLM-L-6-v2`, full chunk text: MRR dropped to 0.546 (-17.4%)
3. Same model, chunks truncated to 300 characters: MRR dropped to 0.308 (-54.0%)

All MS-MARCO-trained cross-encoders are trained on web search query/passage relevance, not code. They consistently over-ranked files with verbose natural-language docstrings (`app.py`) over files with terse, correct implementations (`helpers.py`). Reranking is implemented and available in `src/reranker.py` but is **not used in the production pipeline** based on this evidence.

### Finding 4: Embedding model context limits matter more than benchmark scores

`mxbai-embed-large` scores higher than `nomic-embed-text` on general MTEB benchmarks. However, its 512-token context limit required truncating code chunks to ~1000 characters, which dropped hybrid MRR from 0.732 to 0.580. `nomic-embed-text`'s effective context handled full function-level chunks without truncation. For this use case, avoiding truncation mattered more than the underlying embedding model's general benchmark score.

---

## The Symbol Graph

`graph_engine.py` builds a directed call graph from tree-sitter AST output: 1,149 nodes (functions/methods) and 3,512 edges (call relationships) for Flask.

This is a **lexical call graph, not a type-resolved static analysis graph**. Python's dynamic typing means a call like `self.processor.handle()` is matched to *every* function named `handle` in the codebase, regardless of the actual type of `processor`. This produces high recall with some false-positive edges.

For `blast_radius()`, raw caller counts for common method names can be very high (e.g. 321 for `add_url_rule`) due to this ambiguity. Test files account for the large majority of these, filtering to source-only callers reduced this to 4, all of which were manually verified as correct call relationships (decorator wrappers and inheritance overrides).

---

## Setup

```bash
# install Ollama models
ollama pull nomic-embed-text
ollama pull llama3.2:3b

# install dependencies
pip install -r requirements.txt

# index a repository and run the API
python src/api.py

# in a separate terminal, run the frontend
cd frontend
npm install
npm start
```

---

## Limitations

- Tested on one repository (Flask, Python) with a 60-question benchmark. Multi-repository and multi-language validation is in progress.
- The symbol graph has false-positive edges from ambiguous attribute calls (see "The Symbol Graph" above).
- `llama3.2:3b` was chosen for compatibility with 8GB RAM systems; larger models would likely improve answer synthesis quality. The retrieval pipeline is model-agnostic,  any Ollama-compatible model can be substituted.
- Cross-encoder reranking is implemented but disabled based on the experimental results above.

---

## Roadmap

- [ ] Multi-repository evaluation (Python + JavaScript)
- [ ] Latency benchmarks per pipeline stage
- [ ] Graph edge accuracy study (sampled precision)
- [ ] MCP server exposing the retrieval pipeline to Claude Code / Cursor