# Upgrade for Windows Python 3.12

Existing environment is in `C:\Users\hamza\Downloads\DejaCode_AI_ML_Task2`. Dataset lives at `C:\Users\hamza\Downloads\codechef competetive coding\questions.csv` and had **1,474** accepted rows (0 missing, 0 duplicates, 1,469 with tags). You already built an older `artifacts/codechef` Sentence-BERT/KMeans index successfully.

## Keep existing index! Build into `artifacts/codechef_v2`

```powershell
# Extract new project archive to C:\Users\hamza\Downloads\DejaCode_Task2_v2
cd "C:\Users\hamza\Downloads\DejaCode_Task2_v2"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui,semantic,test,parquet]"
python -m pytest -q
$dataset = "C:\Users\hamza\Downloads\codechef competetive coding\questions.csv"
python -m dejacode.cli build --data $dataset --semantic --clusters 10 --out artifacts/codechef_v2
python -m dejacode.cli compare --index artifacts/codechef_v2 --top-k 5 --query "Find the minimum cost path between two vertices in a weighted graph"
python -m dejacode.cli compare --index artifacts/codechef_v2 --top-k 5 --query "Choose values to maximize their sum without selecting adjacent array elements"
streamlit run app.py
```

Optional neural refinement:

```powershell
python -m dejacode.cli compare --index artifacts/codechef_v2 --rerank --query "Choose values to maximize their sum without selecting adjacent array elements"
```

The reranker requires an additional model download the first time. The JSON output is now Windows-encoding-safe even when redirected through PowerShell pipes.

### Troubleshooting

- If `ModuleNotFoundError: dejacode`, run `python -m pip install -e .` from the folder with `pyproject.toml`.
- If `torch.cuda.is_available()` is `False`, use CPU; no GPU is required for 1,474 problems.
- If weights are not available offline, omit `--semantic` temporarily to build BM25/TF-IDF; run the semantic build when online.
- Keep V1 and V2 indexes separate; compare methods on the same manually checked queries before claiming gains.
