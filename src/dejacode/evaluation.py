"""Information-retrieval evaluation. Ground truth must be independently created."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_judgments(path: str | Path) -> list[dict]:
    out = []
    with Path(path).open(encoding='utf8') as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry.get('query'), str) or not entry['query'].strip():
                raise ValueError(f'Invalid query on line {line_no}')
            rel = entry.get('relevant_codes', [])
            if not isinstance(rel, list) or not all(isinstance(v, str) for v in rel):
                raise ValueError(f'Expected relevant_codes list of strings on line {line_no}')
            if not rel:
                continue
            out.append(entry)
    if not out:
        raise ValueError('No nonempty judgments. Use evaluation/judgments_template.jsonl')
    return out


def _per_query(found: list[str], relevant: set[str], k: int) -> dict[str, float]:
    retrieved = found[:k]
    hits = [int(code in relevant) for code in retrieved]
    hit_count = sum(hits)
    precision = hit_count / k
    recall = hit_count / len(relevant)
    rr = next((1 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    ap = sum(sum(hits[:i + 1]) / (i + 1) for i, h in enumerate(hits) if h) / min(len(relevant), k)
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return {'precision': precision, 'recall': recall, 'mrr': rr,
            'map': ap, 'ndcg': dcg / idcg if idcg else 0.0, 'hit': float(hit_count > 0)}


def evaluate(index, judgments: list[dict], *, k: int = 5,
             methods: tuple[str, ...] = ('tfidf', 'semantic', 'hybrid')) -> dict:
    codes = set(index.documents['code'].astype(str))
    report = {'k': k, 'source': 'explicit_judgments', 'n_queries': 0, 'methods': {},
              'warnings': ['Relevance labels must be manually verified. Neither shared tags nor lexical overlap proves algorithmic similarity.']}
    for method in methods:
        if method == 'semantic' and not index.has_semantic:
            continue
        if method == 'bm25' and not index.has_bm25:
            continue
        scores = defaultdict(list)
        mistakes = []
        checked = 0
        for entry in judgments:
            own = str(entry.get('query_code', '')).strip()
            relevant = set(map(str, entry['relevant_codes'])) & codes
            relevant.discard(own)
            if not relevant:
                continue
            hits = index.search(entry['query'], method=method, top_k=k, exclude_code=own or None)
            retrieved = [str(item['code']) for item in hits]
            result = _per_query(retrieved, relevant, k)
            for metric, score in result.items():
                scores[metric].append(score)
            checked += 1
            if result['recall'] < 1.0:
                mistakes.append({'query': entry['query'], 'query_code': own,
                                 'relevant_codes': sorted(relevant),
                                 'returned_codes': retrieved,
                                 'false_positives': [c for c in retrieved if c not in relevant],
                                 'missed': sorted(relevant - set(retrieved))})
        report['n_queries'] = checked
        report['methods'][method] = {'metrics': {name + f'@{k}': round(float(np.mean(values)), 5)
                                                  for name, values in scores.items()},
                                      'error_cases': mistakes[:15]}
    return report


# Only recognizable algorithm families for exploratory weak-label evaluation.
# Contest names, handle names, difficulty labels and years are excluded.
ALGORITHM_PROXY_TAGS = frozenset({
    'graph', 'graphs', 'tree', 'trees', 'dfs', 'bfs', 'shortest-path',
    'dijkstra', 'bellman-ford', 'floyd-warshall', 'heavy-light', 'segment-tree',
    'fenwick-tree', 'binary-search', 'dynamic-programming', 'dp', 'greedy',
    'knapsack', 'prefix-sum', 'two-pointers', 'sliding-window', 'strings',
    'string', 'combinatorics', 'number-theory', 'bitmask', 'bitmasking',
    'sorting', 'disjoint-set-union', 'dsu', 'union-find', 'matching',
    'geometry', 'trie', 'hashing', 'matrix', 'flow', 'fft', 'backtracking',
})


def build_weak_tag_judgments(index, *, max_queries: int = 100) -> list[dict]:
    """Weak, imperfect proxy using *only* recognizable algorithm tags.

    Never present these as human-verified correctness scores. Restricting to
    algorithm families avoids accidentally calling shared author/difficulty or
    contest labels a relevant solution-technique pair.
    """
    df = index.documents
    tag_to_codes: dict[str, set] = defaultdict(set)
    for _, row in df.iterrows():
        for tag in row['tags'] if isinstance(row['tags'], list) else []:
            t = str(tag).lower()
            if t in ALGORITHM_PROXY_TAGS:
                tag_to_codes[t].add(str(row['code']))
    entries = []
    for _, row in df.iterrows():
        code = str(row['code'])
        tags = row['tags'] if isinstance(row['tags'], list) else []
        rel = set().union(*(tag_to_codes[str(t).lower()] for t in tags
                            if str(t).lower() in tag_to_codes)) if tags else set()
        rel.discard(code)
        if rel:
            entries.append({'query': row['search_text'], 'query_code': code,
                            'relevant_codes': sorted(rel)})
        if len(entries) >= max_queries:
            break
    return entries
