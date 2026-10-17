# Research evaluation protocol — Déjà Code V2

## Known preliminary issues from user-run V1

- 1,474 CodeChef problem statements successfully indexed; 0 missing and 0 duplicates; 1,469 with tags.
- V1 Sentence-BERT/all-MiniLM-L6-v2 + 10 KMeans clusters: **silhouette 0.0498523** (low separation).
- Shortest-path query returned some graph-related but algorithmically different problems (e.g. Queries on tree again!, Maximal Score Path), alongside stronger matches (Two Closest).
- Nonadjacent maximum-reward query returned reward maximization and array problems but no clearly equivalent nonadjacent DP recurrence among the top 5.
- V1 comparison command crashed due to PowerShell cp1252 Unicode encoding when printing statements containing ≤ and ∈. This was an **output-only encoding failure**, not a model inference failure.

## Proposed experiment (not yet executed on the user's 1,474 problems)

1. Freeze V1 baseline and save the original `artifacts/codechef` intact.
2. Build V2 in `artifacts/codechef_v2`. Model remains MiniLM to **isolate the effects of statement filtering, lexical retrieval and rank fusion**.
3. Create at least 20–40 manually checked queries with a mix of graph, shortest path, trees, DP, greedy, combinatorics, strings, geometry and binary search. Annotate relevant problems using official statements and, when needed, verified solution explanations; do *not* use the retrieval ranking as annotation ground truth.
4. Include positive controls (different stories, same algorithm) and confounders (same story, different algorithm). Check that relevant problems actually exist in the corpus.
5. Evaluate TF-IDF, BM25, semantic, hybrid (RRF), and optionally hybrid+reranker on identical queries and identical judgments. Record Precision@5, Recall@5, MRR@5, MAP@5, NDCG@5, Hit@5 and example errors. Use the same index IDs.
6. Note that the shipped evaluator treats unlabeled documents as nonrelevant: for incomplete manual pools, these metrics are *lower-bound/biased exploratory scores*, not definitive. Use an exhaustive small pool or depth-k pooled judging across all methods for better estimates.
7. Inspect at least 10 false positives and 10 false negatives. Check cleaning truncation and whether tags/contest boilerplate are inadvertently used.
8. Compare clustering silhouette, Davies–Bouldin and Calinski–Harabasz, **and** manually audit clusters for coherent algorithm families. Do not conclude semantic validity from silhouette alone.
9. Optionally evaluate a stronger retrieval encoder or pairwise reranker, keeping the judged set held out. Tune ranking weights on a development subset rather than the evaluation subset.

## Ablation table to fill in only with real outcomes

| Version | Core-only preprocessing | BM25 | RRF | Reranker | P@5 | MRR@5 | NDCG@5 |
|---|---|---|---|---|---|---|---|
| V1 baseline | No | No | No | No | TBD | TBD | TBD |
| V2 TF-IDF | Yes | No | No | No | TBD | TBD | TBD |
| V2 BM25 | Yes | Yes | No | No | TBD | TBD | TBD |
| V2 Sentence-BERT | Yes | No | No | No | TBD | TBD | TBD |
| V2 hybrid | Yes | Yes | Yes | No | TBD | TBD | TBD |
| V2 reranked | Yes | Yes | Yes | Yes | TBD | TBD | TBD |

**No numerical CodeChef retrieval evaluation has been run in this archive.**
