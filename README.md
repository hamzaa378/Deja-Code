# Déjà Code V2 — CodeChef-VIT AI/ML Task 2

An explainable competitive-programming problem search engine for retrieving problems that *may share an underlying algorithm*, clustering them, and measuring retrieval quality. **Python 3.12 / Windows PowerShell compatible.**

**Status:** 15 unit/regression tests passing locally on the *included synthetic demo*; the CodeChef production index must be rebuilt on your Windows computer. This distribution does **not** include your private `questions.csv` or a model index built from it. No claim of improved accuracy on CodeChef is made until annotated evaluation is completed.

## What's new in V2

| Area | V1 | V2 |
|---|---|---|
| Windows JSON output | Could crash with `UnicodeEncodeError` on `≤`, `∈` when piped | ASCII-safe JSON (`ensure_ascii=True`), Unicode preserved when parsed |
| Dataset representations | First 14,000 characters including instructions | Problem-focused, first 270 words; cuts recognized input/output boilerplate, **full statement preserved** |
| Lexical retrieval | TF-IDF (word + char) | TF-IDF + new **BM25** baseline |
| Hybrid ranking | Raw score mixture (0.65 cosine + 0.35 TF-IDF) | **Weighted reciprocal rank fusion** of BM25, TF-IDF, sentence embeddings, with limited transparent high-precision text cues |
| Hard examples | First-pass retrieval only | **Optional CrossEncoder rerank** of top 30 candidates (`--rerank`) |
| Clustering | K-Means, PCA and silhouette | Extra Davies–Bouldin and Calinski–Harabasz metrics; optional `--auto-k`; generic-text filtered from keyword labels |
| UI | Search, clusters | Search, retrieval comparison, optional reranking, clusters, method notes |
| Evaluation | Did not score non-TF-IDF methods without semantic encoder | Supports BM25 / hybrid on CPU without neural model; weak tags restricted to algorithm-family tags |
| Query comparison | Six CLI invocations | One `compare` command loads each model once and omits huge statements |

**Important:** Improvements to the pipeline do not guarantee better ranking accuracy. Dataset relevance judgments, review of false positives, and baseline comparison decide whether V2 is actually better.

## Windows: upgrade your current project

1. Keep your original `DejaCode_AI_ML_Task2` folder and `artifacts/codechef` index intact. Extract the new ZIP to a **new** folder e.g. `C:\Users\hamza\Downloads\DejaCode_Task2_v2`.
2. In the new folder activate Python 3.12:

```powershell
cd "C:\Users\hamza\Downloads\DejaCode_Task2_v2"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,semantic,test,parquet]"
python -m pytest -q
```

3. Your known path to the real dataset (no need to copy the data):

```powershell
$dataset = "C:\Users\hamza\Downloads\codechef competetive coding\questions.csv"
python -m dejacode.cli inspect --data $dataset
```

4. Build a new V2 index without overwriting the original V1 results:

```powershell
python -m dejacode.cli build --data $dataset --semantic --clusters 10 --out artifacts/codechef_v2
```

To experiment with several cluster counts, add `--auto-k` (slower). The Sentence-BERT weights should already be in your Hugging Face cache from V1. On your machine the first version encoded 1,474 problems in ~30 seconds; V2 requires fresh embeddings of cleaned text.

5. **Compare all available retrieval methods with one command:**

```powershell
python -m dejacode.cli compare --index artifacts/codechef_v2 --top-k 5 --query "Find the minimum cost path between two vertices in a weighted graph"
python -m dejacode.cli compare --index artifacts/codechef_v2 --top-k 5 --query "Choose values to maximize their sum without selecting adjacent array elements"
```

No `Select-String` pipeline is required, and all printed JSON is safe for older Windows encodings. To save a comparison report, add `--out artifacts/codechef_v2/comparison.json`.

6. Test more intensive ranking on the most difficult queries:

```powershell
python -m dejacode.cli compare --index artifacts/codechef_v2 --top-k 5 --rerank --query "Choose values to maximize their sum without selecting adjacent array elements"
```

`--rerank` downloads `cross-encoder/ms-marco-MiniLM-L-6-v2` on first use. The first call can take longer; repeat calls reuse it within the same process. This is a general retrieval reranker, not an algorithm-specific supervised model. It cannot be assumed to improve results.

7. Run the interactive browser interface:

```powershell
streamlit run app.py
```

Use index directory `artifacts/codechef_v2` in the sidebar. Switch between **Search**, **Compare retrieval methods**, and **Concept clusters**.

## Understanding the model

`questions.csv` -> sanitize statements -> preserve full statement + compact *index-only* representation -> word and character TF-IDF + BM25 + Sentence-BERT -> RRF hybrid search -> optional neural cross-encoder re-ranking top candidates -> clustered map. Tags/solutions are **not used in retrieval** (no label leakage). Rule-based clues from the *text* are displayed explicitly and given only a small hybrid bonus. They are **not algorithm labels**.

RRF scores (`~0.01`) are rank-fusion values, not similarity percentages. Sentence-BERT is cosine similarity. BM25 and CrossEncoder scores have different scales. **Do not compare raw score magnitudes across methods.**

## Evaluation

The shipped evaluation uses *35 fictional demo problems* and *7 synthetic judged queries*, only as a reproducibility check:

```powershell
python -m dejacode.cli build --data data/demo/demo_problems.csv --clusters 7 --out artifacts/demo_v2
python -m dejacode.cli evaluate --index artifacts/demo_v2 --judgments evaluation/demo_judgments.jsonl --out artifacts/demo_v2/eval.json
```

For real CodeChef, prepare `evaluation/codechef_judgments.jsonl` with manually checked relevant problem IDs before calling:

```powershell
python -m dejacode.cli evaluate --index artifacts/codechef_v2 --judgments evaluation/codechef_judgments.jsonl --top-k 5 --out artifacts/codechef_v2/eval_manually_verified.json
```

Each JSONL line is `{"query":"...","relevant_codes":["auto-id-1", "auto-id-2"], "query_code":"optional-excluded-id"}`. All codes must appear in the V2 index. Capture errors from both *different-story / same-algorithm* and *same-story / different-algorithm* queries. Also document queries with no relevant match in this 1,474-problem corpus; otherwise comparisons can be misleading.

An **exploratory proxy** evaluation is available: 

```powershell
python -m dejacode.cli evaluate --index artifacts/codechef_v2 --weak-tags --max-queries 100 --out artifacts/codechef_v2/WEAK_tag_proxy.json
```

Shared algorithm tags are a limited proxy, **not human ground truth**. They can yield optimistic/incorrect results; competition/author tags are filtered. It is not fair to report them as final accuracy.

## Folder guide

- `src/dejacode/data.py`: import, clean, deduplicate and generate compact search text
- `src/dejacode/engine.py`: TF-IDF, BM25, Sentence-BERT, fusion, clustering, optional reranker, persistence
- `src/dejacode/cli.py`: `inspect`, `build`, `search`, `compare`, `evaluate` and Windows-safe output
- `src/dejacode/evaluation.py`: human judgment loader, precision/recall/MRR/MAP/NDCG; weak tag proxy
- `app.py`: Streamlit visual interface
- `tests/`: unit/regression tests, including Windows Unicode output test
- `data/demo/`, `evaluation/demo_judgments.jsonl`: **synthetic demonstration only**
- `evaluation/judgments_template.jsonl`: format hint, requires adapting real IDs
- `UPGRADE_GUIDE.md`: short commands for your current setup
- `RESEARCH_EVALUATION.md`: experimental design and candidate next improvements
- `HANDOVER.md`: implementation checkpoint and architecture continuation notes

## Known limitations

- Input heading detection is heuristic and may omit meaningful text after a heading; full statements are still accessible, and cleaning should be manually audited on a real sample.
- General-purpose embeddings and rerankers understand language similarity, not proven algorithm equivalence.
- Small RRF fusion weights are **not optimized on real CodeChef relevance data** to avoid leakage.
- Clusters may group stories rather than solution strategies; geometry scores alone are not sufficient.
- Real measured improvements on 1,474 CodeChef problems are **pending** and must not be fabricated.
- `joblib` artifacts should be loaded only from trusted sources; Python pickle loading is unsafe on untrusted files.

## Source / task

This implements AI/ML Task 2, **Déjà Code**, from pages 13–14 of the CodeChef-VIT 2026–27 Promotional Cycle task sheet: preprocessing, embeddings, search, clustering, visualization, evaluation, error analysis, and an interactive search. The CodeChef dataset used by the local build belongs to its respective owners. The ZIP does **not** bundle any third-party CodeChef data or external transformer weights.
