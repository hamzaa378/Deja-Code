# Déjà Code — CodeChef-VIT AI/ML Task 2

**Goal:** Search for competitive programming problems that require similar underlying algorithms, even when the statements describe different stories. Group problems into related concepts and rigorously evaluate whether search results are actually relevant.

This project addresses **all eight Expected Functionality items** and the six **Required Deliverables** listed on pages 13–14 of the CodeChef-VIT 2026–27 Promotional Cycle task sheet. It also implements the optional TF-IDF baseline and weighted **hybrid search** bonus. The dashboard is optional.

> **Data honesty:** The archive includes 35 *original synthetic demonstration statements* and an evaluation on those statements for immediate, offline testing. **It does not include or claim evaluation on the CodeChef Kaggle dataset.** For actual submission, download that dataset, build a Sentence-BERT index, create and review real relevance judgments, then regenerate the report.

## 1. Architecture

```text
CodeChef statement dataset (CSV / TSV / JSON / JSONL / parquet / text directory)
                         |
                         v
   ingestion -> HTML & whitespace cleaning -> statement/code deduplication
                         |
                         v
                    normalized corpus
                      /       \
                     /         \
       TF-IDF word + char   Sentence-BERT embeddings
                 \           /
                 configurable fusion
                         |
                  cosine retrieval
                         |
            Top-K related problems in UI

   normalized embeddings -> K-Means -> cluster IDs + descriptive keywords
                         |             |
                        PCA       silhouette score
                         |
                    concept explorer

   human relevance labels -> Precision, Recall, MAP, MRR, NDCG @ K
                         |
             comparison + false-positive review
```

**Retrieval choices:**

- `tfidf`: Word unigram+bigram TF-IDF (70%) and character 3–5-gram TF-IDF (30%). Strong, fast, reproducible lexical baseline.
- `semantic`: Pretrained `sentence-transformers/all-MiniLM-L6-v2`; normalized sentence embeddings and cosine similarity. Different terminology can still map to similar representations, though the generic model may not understand algorithms.
- `hybrid`: 65% semantic cosine score + 35% lexical score. This simple, untuned blend is a starting point and **not guaranteed** to outperform either component. When no semantic model is built, the CLI's hybrid mode transparently falls back to the lexical baseline; the UI only offers TF-IDF.

**Clustering:** K-Means in embedding space (Sentence-BERT if available, otherwise TF-IDF compressed by TruncatedSVD), PCA for a two-dimensional plot, and distinctive top TF-IDF terms as approximate labels. K is configurable; we do **not** assume K is the correct number of algorithm families.

**No label leakage into retrieval:** The index searches problem titles and problem statements. Existing algorithm tags, solutions, and test answers are not part of the vectorized input. Tags can optionally support a clearly marked weak-proxy evaluation.

## 2. Installation — Windows PowerShell

From inside the extracted `DejaCode_Task2` directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[ui,semantic,test,parquet]"
```

Python 3.10+ supported; **3.11** is a practical choice. `sentence-transformers` and PyTorch increase download size; first use downloads the pretrained model from Hugging Face. Internet access is required for installation and first model download. A CPU is sufficient for ~1,000 problem statements, though GPU is faster.

For **minimal TF-IDF-only use**, install:

```powershell
python -m pip install -e ".[ui,test]"
```

On macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate` then the same `python -m pip install ...` command.

If PowerShell blocks activation, call `.\.venv\Scripts\python.exe` in commands instead of activating the environment. The offline verification environment used Python's existing installed packages and `PYTHONPATH=src`, as package downloading is unavailable there.

## 3. Run the included working demonstration

```powershell
# Optional: regenerate the sample data from original written cases
python scripts/make_demo_data.py

# Check the ingest and schema
python -m dejacode.cli inspect --data data/demo/demo_problems.csv

# Fit lexical search + clustering; create cluster_map.png
python -m dejacode.cli build --data data/demo/demo_problems.csv --clusters 7 --out artifacts

# Try a query
python -m dejacode.cli search --index artifacts --method tfidf --top-k 5 --query "Roads charge different tolls. Find the route with minimum total cost."

# Evaluate against the INCLUDED SYNTHETIC DEMO judgments
python -m dejacode.cli evaluate --index artifacts --judgments evaluation/demo_judgments.jsonl --out artifacts/demo_evaluation.json

# Open UI
streamlit run app.py

# Run tests
pytest -q
```

An initial **demo index and clustering plot** are already in `artifacts/`. The commands above reproduce them. This is a ready-to-run example even before you download CodeChef data.

## 4. Use the actual CodeChef dataset (required for submission)

**Dataset source:** [Arjoonn Sharma, CodeChef Competitive Programming, Kaggle](https://www.kaggle.com/datasets/arjoonn/codechef-competitive-programming). This is the matching named source in the task sheet; Kaggle describes about 1,000 scraped CodeChef statements, **plus a much larger collection of solutions**. Only index problem statements — not submitted solutions. Do not place the entire solutions corpus in an import folder.

1. Download the dataset on Kaggle (a Kaggle account/API authentication may be needed). If you use the Kaggle CLI, the command is:

   ```powershell
   python -m pip install kaggle
   kaggle datasets download -d arjoonn/codechef-competitive-programming -p data/raw --unzip
   ```

2. Inspect the extracted filenames, choose the **statement/problem** file or folder, and run:

   ```powershell
   python -m dejacode.cli inspect --data "data/raw/PATH_TO_STATEMENT_FILE_OR_FOLDER"
   ```

3. The loader accepts common statement fields like `problem_statement`, `description`, `statement`, `content`, and code fields like `problem_code` or `code`. If the downloaded dataset uses different column names, pass the explicit mapping:

   ```powershell
   python -m dejacode.cli inspect --data "data/raw/problems.csv" --statement-col "ACTUAL_STATEMENT_COLUMN" --id-col "ACTUAL_CODE_COLUMN" --title-col "ACTUAL_TITLE_COLUMN"
   ```

4. **Build the semantic and hybrid index:**

   ```powershell
   python -m dejacode.cli build --data "data/raw/PATH_TO_STATEMENT_FILE_OR_FOLDER" --semantic --clusters 10 --out artifacts/codechef
   ```

   If explicit column mappings were needed in Step 3, repeat them here. First run downloads `all-MiniLM-L6-v2`; cache it for later offline use. This is pretrained **embedding inference**, not training a language model from scratch. It is a legitimate NLP approach for the task.

5. **Launch the real-data dashboard** with `streamlit run app.py`, and change *Index directory* in the sidebar to `artifacts/codechef`. Use `semantic`, `tfidf`, and `hybrid` for the same query, then inspect differences.

6. **Evaluate real retrieval**, not the demo. Duplicate `evaluation/judgments_template.jsonl` into a new file and replace the placeholders with **actual CodeChef problem codes and manually verified relevant pairs**. Keep one JSON object per line. See section 6.

   ```powershell
   python -m dejacode.cli evaluate --index artifacts/codechef --judgments evaluation/codechef_judgments.jsonl --top-k 5 --out artifacts/codechef/evaluation.json
   ```

**Dataset caveat:** We could verify Kaggle's public dataset listing, but did not download the original full dataset inside this environment, so its specific column names/files have *not* been independently validated here. The flexible loader allows `CSV`, `TSV`, `JSON`, `JSONL`, `parquet` (needs `pyarrow`), or a directory of statement text files; when in doubt, use `inspect` and the explicit mapping flags. Avoid indexing source code, solutions, or generic documents as if they were statements.

## 5. Required features and concrete files

| Requirement | Implementation / proof |
|---|---|
| 1. Explore/clean statements | `src/dejacode/data.py`, `inspect` command, `artifacts/problems.json` |
| 2. Represent statements numerically | `src/dejacode/engine.py` word/char TF-IDF and optional MiniLM embeddings |
| 3. Semantic search | `ProblemIndex.search`; cosine similarity for neural embeddings |
| 4. Group related problems | `ProblemIndex.cluster`; K-Means |
| 5. Visualize and interpret groups | `artifacts/cluster_map.png`, `artifacts/cluster_report.json`, Streamlit plot |
| 6. User submits an arbitrary statement | `app.py` multiline query and top-K results |
| 7. Measure retrieval quality | `src/dejacode/evaluation.py`, CLI `evaluate` with per-query judgments |
| 8. Examine false matches | Evaluation JSON's `error_cases`, including missed and false-positive codes |
| TF-IDF vs semantic bonus | Three selectable retrieval modes |
| Hybrid-search bonus | Weighted lexical + semantic scoring |
| Cluster explorer bonus | Streamlit interactive plot and click-through cluster listing |

## 6. Evaluation protocol — no misleading metrics

**Create independent relevance judgments** by selecting 20–50 real query problems (preferably more), and asking annotators to find other problems solved by essentially the same technique. Check statements and editorials/solutions as needed. Keep problems with the same story but **different algorithms** labeled irrelevant. Relevance here means underlying algorithm, *not* merely overlapping words, topics, or difficulties.

Example judgment schema (replace all values with real IDs):

```json
{"query":"<entire query problem statement>","query_code":"CODE_OF_QUERY_PROBLEM","relevant_codes":["OTHER_CODE_A","OTHER_CODE_B"],"notes":"Confirmed same dynamic programming recurrence"}
```

**Important:** The target problem's `query_code` is excluded from retrieved results; otherwise copying a corpus statement would trivially retrieve itself and falsely inflate scores. Problems in `relevant_codes` must exist in the index. The evaluator excludes missing codes from scoring. It reports:

- `Precision@K`: fraction of returned top-K items judged relevant.
- `Recall@K`: fraction of known relevant problems found in top K.
- `MRR@K`: reciprocal rank of first relevant result, averaged over queries.
- `MAP@K`: mean truncated average precision.
- `NDCG@K`: rank-sensitive discounted relevance (binary relevance labels).
- `Hit@K`: at least one known relevant item in the top K.

For the evaluation to be credible, judge candidates from **more than one retrieval method** (pooling), not only the semantic top results; otherwise judgments may be biased toward the system that produced them. Do not label all unjudged results as known negatives when reporting human evaluation: the included evaluator treats other retrieved results as nonrelevant for the current *judged set*, which can underestimate systems finding unjudged true matches. Expand judgments and discuss this limitation.

An **optional proxy** is available where metadata tags overlap:

```powershell
python -m dejacode.cli evaluate --index artifacts/codechef --weak-tags --max-queries 100 --out artifacts/codechef/weak_tag_proxy.json
```

This output is labeled `WEAK_SHARED_TAG_PROXY_NOT_GROUND_TRUTH`; it is for quick sanity checks only. The original Kaggle dataset may not include algorithm tags.

## 7. Demonstration results (original synthetic data only)

Verified by running the included commands against the 35 generated statements (7 concept families, 5 statements each), with 7 author-labeled queries:

| Test | Measured outcome |
|---|---:|
| Parsed problem statements | 35 |
| Missing statements | 0 |
| Duplicate IDs | 0 |
| TF-IDF Precision@5 | 0.40000 |
| TF-IDF Recall@5 | 0.50000 |
| TF-IDF MRR@5 | 0.71429 |
| TF-IDF MAP@5 | 0.42500 |
| TF-IDF NDCG@5 | 0.52438 |
| K-Means clusters requested | 7 |
| Clustering silhouette (TruncatedSVD TF-IDF) | -0.00034 |
| Offline automated unit/integration tests | 7 passed |

**Interpretation:** The near-zero/negative silhouette score means the lexical features do **not** form clearly separated groups on this tiny dataset. The search also confuses unrelated story words with algorithm structure. This is a meaningful failure case, not a result to hide. These figures **must not** be copied as the results of a CodeChef experiment. No Sentence-BERT benchmark has been run in the offline verification environment, because that package and the pretrained model are not installed there.

Example test failure: The query `Subway Transfers` requires unweighted shortest paths, but lexical retrieval favors different tasks using shared terms like *minimum* and *connected*, missing all four author-labeled related problems in the top five. That is precisely why the semantic comparison matters. See `artifacts/demo_evaluation.json` for exact returned codes and misses.

## 8. Design decisions and limitations

- The generic MiniLM model is **not algorithm-specific**. It can confuse story similarity with computational equivalence. Improving this requires high-quality labeled positive/negative problem pairs and metric-learning fine-tuning or reranking.
- TF-IDF is an honest baseline, and it can outperform neural embeddings when rare variable names, terminology, or special mathematical symbols matter.
- K-Means assumes Euclidean clusters with a chosen K. Some problems require multiple concepts; hard clusters can be misleading. Avoid claiming cluster labels are ground truth.
- Character n-grams aid typo/paraphrase robustness but do not prove algorithmic similarity.
- PCA is for visualization; it can distort distance and separate concepts imperfectly.
- Truncating and stripping sample input/output prevents superficial matching and keeps index lightweight, but may discard clues in tricky problems.
- Public CodeChef dataset may contain older/sparse statements. The raw scraper's content and license should be checked before redistributing any data.
- `artifacts/tfidf.joblib` uses pickle-based serialization. Load saved files only from trusted local builds; do not accept arbitrary uploaded index directories.
- Performance depends on annotator coverage and the number/diversity of CodeChef problems. A tiny synthetic dataset is not evidence of generalization.

**Next improvements if time permits:** (a) create manually checked same-algorithm/different-story challenge pairs; (b) train a domain-specific bi-encoder with in-batch negatives; (c) add a cross-encoder reranker; (d) compare Recall@10 + human judgment of hard false positives; (e) add relevance explanations citing the algorithmic mechanism, not generic shared words.

## 9. Reproducibility / submission checklist

- [x] Source-code project, executable CLI, tests, dashboard implementation, documentation
- [x] Synthetic proof-of-function dataset and judged example queries
- [x] Local TF-IDF baseline, clustering, evaluation metrics, static PNG and results
- [x] Semantic embedding and hybrid retrieval **implementation** (not executed offline)
- [ ] Download and preprocess **actual CodeChef dataset**
- [ ] Build actual Sentence-BERT embeddings and rerun searches
- [ ] Make independent algorithm-level relevance annotations on real problems
- [ ] Compare all three methods with measured real-data Precision/Recall/MRR/MAP/NDCG
- [ ] Inspect real failure cases and produce screenshots/video of live UI
- [ ] Upload project to **your own GitHub repository** and add the link to your submission

See `DECISIONS.md` for your technical defense and `DEMO_RESULTS.md` for the verified offline benchmark.
