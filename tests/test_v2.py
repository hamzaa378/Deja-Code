"""V2 regression coverage including the reported Windows cp1252 pipe crash."""
import io
import json
from pathlib import Path

import numpy as np
import pytest

from dejacode.cli import output_json
from dejacode.data import load_problems, search_representation
from dejacode.engine import ProblemIndex, detected_concepts, reciprocal_rank_fusion
from dejacode.evaluation import evaluate, read_judgments

ROOT = Path(__file__).resolve().parents[1]


def demo():
    return ProblemIndex(load_problems(ROOT / 'data/demo/demo_problems.csv'))


def test_boilerplate_is_filtered_but_original_is_preserved():
    raw = ('All submissions for this problem are available. Given a weighted graph, '
           'find the shortest path between two nodes. Input The first line gives N. '
           'Output Print a number. Constraints 1 <= N <= 100.')
    ix = search_representation('Road Route', raw)
    assert 'shortest path' in ix
    assert 'The first line' not in ix
    assert 'All submissions' not in ix
    assert 'Input' not in ix
    assert len(ix.split()) < len(raw.split())


def test_bm25_and_hybrid_work_without_semantic_embeddings():
    ix = demo()
    assert ix.has_bm25
    for method in ('tfidf', 'bm25', 'hybrid'):
        rows = ix.search('Find the shortest path along weighted roads', method=method,
                         include_statement=False)
        assert len(rows) == 5
        assert all(np.isfinite(row['score']) for row in rows)
        assert all('statement' not in row for row in rows)
    assert ix.search(' ', method='bm25') == []


def test_v2_roundtrip_and_metadata(tmp_path):
    ix = demo()
    ix.cluster(7)
    ix.save(tmp_path)
    loaded = ProblemIndex.load(tmp_path)
    assert loaded.version == 2
    assert loaded.has_bm25
    assert len(loaded.search('Shortest path', method='hybrid')) == 5
    assert (tmp_path / 'meta.json').exists()


def test_ascii_safe_json_even_with_cp1252():
    # UTF-8 source statements may include ≤, ∈, em dashes etc. A redirected
    # Windows cp1252 output stream otherwise raises UnicodeEncodeError.
    bio = io.BytesIO()
    stream = io.TextIOWrapper(bio, encoding='cp1252')
    import unittest.mock
    with unittest.mock.patch('sys.stdout', stream):
        output_json({'statement': 'a ≤ b, x ∈ A, café ☃'})
        stream.flush()
    contents = bio.getvalue().decode('cp1252')
    assert '\\u2264' in contents and '\\u2208' in contents
    assert json.loads(contents)['statement'].startswith('a ≤ b')


def test_rrf_ranks_not_raw_scale():
    a = np.array([1000., 0., 0.])
    b = np.array([0.0, 0.1, 0.3])
    fused = reciprocal_rank_fusion({'a': a, 'b': b})
    assert len(fused) == 3
    assert max(fused) < 1


def test_heuristics_are_opt_in_hints_only():
    assert 'nonadjacent_selection' in detected_concepts('select maximum without choosing adjacent elements')
    assert 'shortest_path' in detected_concepts('find the shortest path')
    assert not detected_concepts('choose a thing')


def test_evaluation_does_not_skip_nonsemantic_methods():
    ix = demo()
    judgments = read_judgments(ROOT / 'evaluation/demo_judgments.jsonl')
    report = evaluate(ix, judgments, k=5, methods=('tfidf', 'bm25', 'hybrid'))
    assert set(report['methods']) == {'tfidf', 'bm25', 'hybrid'}
    assert report['n_queries'] == 7


def test_k_auto_candidates():
    ix = demo()
    report = ix.cluster(k=7, auto_k=True)
    assert len(report['auto_k_trials']) >= 3
    assert report['davies_bouldin'] > 0
    assert sum(report['cluster_sizes'].values()) == 35
