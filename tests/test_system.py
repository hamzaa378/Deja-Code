import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from dejacode.data import load_problems, clean_text, parse_tags
from dejacode.engine import ProblemIndex
from dejacode.evaluation import _per_query, evaluate, read_judgments

ROOT = Path(__file__).resolve().parents[1]


def build_demo():
    df = load_problems(ROOT / 'data/demo/demo_problems.csv')
    return ProblemIndex(df)


def test_ingestion_and_cleaning():
    df = load_problems(ROOT / 'data/demo/demo_problems.csv')
    assert len(df) == 35
    assert df['code'].is_unique
    assert all(len(x.split()) >= 12 for x in df['statement'])
    assert parse_tags('["graph", "bfs"]') == ['bfs', 'graph']
    assert clean_text('<p>Hello</p> more text') == 'Hello more text'


def test_search_exclusion_and_recovery():
    ix = build_demo()
    q = ix.documents.iloc[0]['search_text']
    results = ix.search(q, method='tfidf', top_k=5, exclude_code='demo-01-01')
    assert len(results) == 5
    assert all(r['code'] != 'demo-01-01' for r in results)
    assert all(np.isfinite(r['score']) for r in results)
    assert ix.search('   ') == []


def test_search_has_neural_guard():
    ix = build_demo()
    with pytest.raises(ValueError):
        ix.search('Some network and paths', method='semantic')


def test_clustering_roundtrip(tmp_path):
    ix = build_demo()
    report = ix.cluster(k=7)
    assert sum(report['cluster_sizes'].values()) == 35
    ix.save(tmp_path)
    ix2 = ProblemIndex.load(tmp_path)
    assert ix2.cluster_table is not None
    assert len(ix2.search('Find a cheapest weighted road route', method='tfidf')) == 5
    assert len(ix2.cluster_terms) >= 2


def test_evaluation_math():
    val = _per_query(['A', 'X', 'B'], {'A', 'B'}, k=3)
    assert val['precision'] == pytest.approx(2 / 3)
    assert val['recall'] == pytest.approx(1)
    assert val['mrr'] == 1
    assert val['map'] == pytest.approx((1 + 2 / 3) / 2)
    assert val['ndcg'] > .9


def test_demo_judgments_and_no_self_leakage():
    ix = build_demo()
    judgments = read_judgments(ROOT / 'evaluation/demo_judgments.jsonl')
    report = evaluate(ix, judgments, k=5)
    assert report['n_queries'] == 7
    assert report['methods']['tfidf']['metrics']['precision@5'] >= 0
    for entry in report['methods']['tfidf']['error_cases']:
        assert entry['query_code'] not in entry['returned_codes']


def test_nested_json_and_deduplication(tmp_path):
    obj = {'easy': {'ABC': {'problem_name': 'A', 'problem_statement': ' '.join(['Find the value'] * 20),
                            'user_tags': ['math']},
                    'DEF': {'problem_name': 'B', 'problem_statement': ' '.join(['Take graph edge'] * 20)}}}
    file = tmp_path / 'nested.json'
    file.write_text(json.dumps(obj), encoding='utf8')
    df = load_problems(file)
    assert set(df.code) == {'ABC', 'DEF'}
    assert df.set_index('code').loc['ABC', 'tags'] == ['math']
