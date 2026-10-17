# Offline experiment record — synthetic demonstration ONLY

**Run conditions:** Python (local execution environment), NumPy/Pandas/scikit-learn, random seed 42 for cluster reduction and K-Means. CodeChef Kaggle dataset **not** used for these numbers.

## Corpus and test methodology

`python scripts/make_demo_data.py` generates 35 *original* sample problems in seven algorithm groups: unweighted shortest paths, nonnegative weighted shortest paths, 0/1 knapsack, binary search on answer, sliding window, DSU/dynamic connectivity, and interval scheduling. One query problem per group is held as the query; its own code is excluded. The four remaining items in its group are designated relevant by the sample author.

`python -m dejacode.cli build --data data/demo/demo_problems.csv --clusters 7 --out artifacts` fits the TF-IDF retrieval model and K-Means clusters and saves PCA data and a PNG scatter plot. With no Sentence-BERT download available in the runtime, **this is a lexical-only experiment**.

`python -m dejacode.cli evaluate --index artifacts --judgments evaluation/demo_judgments.jsonl --top-k 5 --out artifacts/demo_evaluation.json` uses the held-query judgments to compute the score table:

| Metric | TF-IDF @ 5 |
|---|---:|
| Precision | 0.40000 |
| Recall | 0.50000 |
| Hit rate | 0.85714 |
| MRR | 0.71429 |
| MAP | 0.42500 |
| NDCG | 0.52438 |

`pytest -q` result: **7 passed**. Preprocessing accepted 35 rows, no missing statements, no duplicate IDs. For K-Means (`k=7`), silhouette = **-0.00034**, indicating weak separation in this deliberately small lexical feature space.

## Error analysis

- **Unweighted shortest path:** The `Subway Transfers` query retrieved no designated same-algorithm counterparts in the top five. Its false positives include weighted routing problems and other cases about networks or counts. Different story vocabulary masked the same BFS pattern.
- **Weighted shortest path:** `Fuel Efficient Roads` found some weighted-road counterparts but still missed `Delivery Charges` and `Mountain Trails` in the top five. The lexical model was distracted by common words also present in non-routing problems.
- **0/1 knapsack:** `Festival Backpack` missed several other capacity/subset maximization statements. Domain-specific story words such as *souvenir* are too prominent in sparse TF-IDF.

**Conclusions limited to this test:** Search, ingestion, error reporting and plotting operate as intended. TF-IDF alone does not reliably discover shared algorithms through divergent narratives in this example. The next experiment must run Sentence-BERT **and** human-annotated real CodeChef data before claiming semantic improvement. These synthetic metrics are not suitable as a CodeChef performance result.
