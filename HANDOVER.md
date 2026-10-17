# Déjà Code V2 — developer handover / continuity checkpoint

**Date:** 2026-10-17. Task: CodeChef-VIT AI/ML Task 2, Déjà Code. V2 source ZIP lives separately from V1; the user's original V1 artifacts must not be overwritten.

## User environment / data (reported by user)

- Windows PowerShell, Python **3.12.0**, venv `.venv`.
- Old project: `C:\Users\hamza\Downloads\DejaCode_AI_ML_Task2`.
- Real source: `C:\Users\hamza\Downloads\codechef competetive coding\questions.csv` (do not copy `solutions.csv`).
- Loader diagnostics: **1474** accepted problems, **0** missing statements, **0** duplicates, **1469** tagged.
- V1 model: `sentence-transformers/all-MiniLM-L6-v2`, indexed all 1474, embedding time 30sec, 10 clusters, silhouette 0.0498523; stored at old `artifacts/codechef` (user machine only).
- V1 quality issue: generic graph retrieval, poor nonadjacent DP matches. Separate defect: `UnicodeEncodeError` on cp1252 when JSON printing special symbols to piped PowerShell.

## V2 code responsibilities

`src/dejacode/data.py`: robust ingestion, `search_representation()` truncates input/format noise while retaining original statement. `engine.py`: TF-IDF, BM25 from sparse counts, Sentence-BERT, weighted RRF, concept text-clue detection, optional CrossEncoder rerank top candidates, clustering metrics and joblib/NumPy persistence. `cli.py`: inspect/build/search/compare/evaluate, ASCII-safe JSON. `evaluation.py`: manual rankings and algorithm-tag-only proxy. `app.py`: Streamlit comparison and concept map. `tests/test_v2.py`: regression cases, including cp1252 failure. `README.md`: complete setup. `UPGRADE_GUIDE.md`: user-specific steps. `RESEARCH_EVALUATION.md`: experimental protocol.

## Validation on included offline demo

- 35 **fictional** labeled test statements; 7 **synthetic** test queries, not CodeChef benchmark.
- Latest locally verified: **15 tests pass**. TF-IDF Precision@5 0.42857, BM25 0.42857, Hybrid 0.42857; MRR@5 TF-IDF 0.67857, BM25 0.71429, Hybrid 0.64286. Hybrid is not always better; show all baselines and compare carefully.
- V2 real-data results **not yet measured**. No user's CodeChef CSV or model weights are bundled.
- Fresh V2 demo artifacts in `artifacts/demo_v2/` may be omitted from ZIP because reproducible by README.

## Remaining work

1. User extracts V2 project, installs with Python 3.12, runs 15 tests.
2. Rebuild from user's `questions.csv` to **new** `artifacts/codechef_v2`; inspect preprocessing and cluster report.
3. Run `compare` for the two prior queries. Optionally `--rerank` to test second-stage matching.
4. Create human-verified CodeChef query/relevant-problem annotations; run evaluation across all methods. Do not claim performance boosts from rankings without labels.
5. Consider additional retrieval training / solution-technique verification only after diagnosis from a reliable evaluation set.

## Limitations / safeguards

- CrossEncoder model is general-domain, optional and downloaded separately; no proof of algorithm equivalence.
- RRF weight choices are unoptimized; rule cues are weak and based on problem text, not tags.
- Weak tag judgments are not proper ground truth; concept labels are hints only.
- Some relevant constraints may appear in omitted input sections; check samples.
- joblib uses pickle and must only load trusted artifacts.
