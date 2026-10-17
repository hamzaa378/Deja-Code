# V2 synthetic demo results (NOT CodeChef results)

- Dataset: 35 made-up demonstration problems.
- Evaluation: 7 synthetic judgments, k=5.
- CPU baseline, no Sentence-BERT evaluated in this environment.

| Method | Precision@5 | Recall@5 | MRR@5 | MAP@5 | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.42857 | 0.53571 | 0.67857 | 0.43750 | 0.53479 |
| BM25 | 0.42857 | 0.53571 | 0.71429 | 0.43988 | 0.55227 |
| Hybrid (lexical-only RRF) | 0.42857 | 0.53571 | 0.64286 | 0.42202 | 0.53169 |

This small synthetic test does NOT show consistent superiority of hybrid. For CodeChef evaluation, create manually verified relevance labels. The Windows output regression passes with escaped ≤ / ∈.
