"""Cross-platform CLI, with ASCII-safe JSON for Windows cp1252 consoles."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .data import load_problems
from .engine import DEFAULT_MODEL, DEFAULT_RERANKER, ProblemIndex, draw_clusters
from .evaluation import build_weak_tag_judgments, evaluate, read_judgments


def output_json(obj: object):
    # Important on Windows: PowerShell 5 / redirected pipes may use cp1252.
    # JSON escapes all non-ASCII characters; no change to the underlying data.
    print(json.dumps(obj, indent=2, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(prog='dejacode', description='Deja Code V2: competitive-programming search')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Inspect normalized dataset and inferred schema')
    inspect.add_argument('--data', required=True)
    inspect.add_argument('--statement-col')
    inspect.add_argument('--id-col')
    inspect.add_argument('--title-col')
    inspect.add_argument('--limit', type=int)

    build = sub.add_parser('build', help='Build lexical indexes, semantic embeddings and clusters')
    build.add_argument('--data', required=True)
    build.add_argument('--out', default='artifacts/v2')
    build.add_argument('--semantic', action='store_true')
    build.add_argument('--model', default=DEFAULT_MODEL)
    build.add_argument('--batch-size', type=int, default=32)
    build.add_argument('--clusters', type=int, default=10)
    build.add_argument('--auto-k', action='store_true', help='Explore nearby k values via sample silhouette')
    build.add_argument('--limit', type=int)
    build.add_argument('--statement-col')
    build.add_argument('--id-col')
    build.add_argument('--title-col')

    search = sub.add_parser('search', help='Query an indexed corpus')
    search.add_argument('--index', default='artifacts/v2')
    search.add_argument('--query', required=True)
    search.add_argument('--method', choices=('tfidf', 'bm25', 'semantic', 'hybrid'), default='hybrid')
    search.add_argument('--top-k', type=int, default=5)
    search.add_argument('--exclude-code')
    search.add_argument('--compact', action='store_true', help='Omit long full statements from JSON')
    search.add_argument('--rerank', action='store_true', help='Optional cross-encoder on shortlisted candidates')
    search.add_argument('--rerank-pool', type=int, default=30)
    search.add_argument('--rerank-model', default=DEFAULT_RERANKER)

    compare = sub.add_parser('compare', help='Run all available methods in one process for a query')
    compare.add_argument('--index', default='artifacts/v2')
    compare.add_argument('--query', required=True)
    compare.add_argument('--top-k', type=int, default=5)
    compare.add_argument('--rerank', action='store_true', help='Also include hybrid + cross encoder')
    compare.add_argument('--rerank-pool', type=int, default=30)
    compare.add_argument('--rerank-model', default=DEFAULT_RERANKER)
    compare.add_argument('--out', help='Optional file for full comparison JSON')

    ev = sub.add_parser('evaluate', help='Score retrieval against manual judgments or proxy tags')
    ev.add_argument('--index', default='artifacts/v2')
    ev.add_argument('--judgments')
    ev.add_argument('--weak-tags', action='store_true')
    ev.add_argument('--max-queries', type=int, default=100)
    ev.add_argument('--top-k', type=int, default=5)
    ev.add_argument('--out', default='artifacts/v2/retrieval_evaluation.json')

    args = parser.parse_args()
    start = time.perf_counter()
    if args.command in ('inspect', 'build'):
        data = load_problems(args.data, statement_col=args.statement_col,
                             id_col=args.id_col, title_col=args.title_col, limit=args.limit)
        output_json({'problems': len(data), 'input_diagnostics': data.attrs['diagnostics'],
                     'columns': list(data.columns), 'has_tags': int(data['tags'].map(bool).sum()),
                     'sample': data[['code', 'title', 'difficulty']].head(4).to_dict('records')})
        if args.command == 'inspect':
            return
        if len(data) < 3:
            raise ValueError('Need at least 3 usable statements for clustering')
        ingestion_seconds = time.perf_counter() - start
        clock = time.perf_counter()
        index = ProblemIndex(data)
        lexical_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        if args.semantic:
            index.attach_embeddings(args.model, batch_size=args.batch_size)
        neural_seconds = time.perf_counter() - clock
        clock = time.perf_counter()
        report = index.cluster(k=min(args.clusters, len(data) - 1), auto_k=args.auto_k)
        cluster_seconds = time.perf_counter() - clock
        target = Path(args.out)
        index.save(target)
        draw_clusters(index, target / 'cluster_map.png')
        (target / 'cluster_report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
        lengths = index.documents['search_text'].astype(str).str.split().map(len)
        original_lengths = index.documents['statement'].astype(str).str.split().map(len)
        build_report = {'n_problems': len(index.documents),
                        'mean_index_words': round(float(lengths.mean()), 2),
                        'median_index_words': int(lengths.median()),
                        'min_index_words': int(lengths.min()),
                        'possible_overtruncation_count': int(((lengths < 25) & (original_lengths >= 80)).sum()),
                        'durations_seconds': {'ingest': round(ingestion_seconds, 2),
                            'lexical_indexes': round(lexical_seconds, 2),
                            'semantic_embeddings': round(neural_seconds, 2),
                            'clustering': round(cluster_seconds, 2),
                            'total': round(time.perf_counter() - start, 2)}}
        (target / 'build_report.json').write_text(json.dumps(build_report, indent=2), encoding='utf8')
        output_json({'index': str(target), 'semantic': index.has_semantic,
                     'build_report': build_report, 'clusters': report})
    elif args.command == 'search':
        index = ProblemIndex.load(args.index)
        if args.method == 'semantic' and not index.has_semantic:
            parser.error('Semantic embeddings not built; rerun build --semantic')
        if args.method == 'bm25' and not index.has_bm25:
            parser.error('BM25 not built; rerun build to a new v2 index')
        output_json(index.search(args.query, method=args.method, top_k=args.top_k,
                                 exclude_code=args.exclude_code,
                                 rerank=args.rerank, rerank_pool=args.rerank_pool,
                                 rerank_model=args.rerank_model,
                                 include_statement=not args.compact))
    elif args.command == 'compare':
        index = ProblemIndex.load(args.index)
        methods = ['tfidf']
        if index.has_bm25:
            methods.append('bm25')
        if index.has_semantic:
            methods.append('semantic')
        methods.append('hybrid')
        table = {}
        for method in methods:
            table[method] = index.search(args.query, method=method, top_k=args.top_k,
                                         include_statement=False)
        if args.rerank:
            table['hybrid_reranked'] = index.search(args.query, method='hybrid',
                                                     top_k=args.top_k, rerank=True,
                                                     rerank_pool=args.rerank_pool,
                                                     rerank_model=args.rerank_model,
                                                     include_statement=False)
        report = {'query': args.query, 'results': table,
                  'warning': 'Scores are method-specific ranking signals; cross-method scores are not comparable.'}
        if args.out:
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
        output_json(report)
    elif args.command == 'evaluate':
        index = ProblemIndex.load(args.index)
        if bool(args.judgments) == bool(args.weak_tags):
            parser.error('Choose --judgments FILE or --weak-tags')
        judgments = read_judgments(args.judgments) if args.judgments else \
            build_weak_tag_judgments(index, max_queries=args.max_queries)
        if not judgments:
            parser.error('No relevance pairs. Supply manual judgments or tag proxy.')
        report = evaluate(index, judgments, k=args.top_k,
                          methods=tuple(['tfidf'] + (['bm25'] if index.has_bm25 else []) +
                                        (['semantic', 'hybrid'] if index.has_semantic else ['hybrid'])))
        if args.weak_tags:
            report['source'] = 'WEAK_SHARED_TAG_PROXY_NOT_GROUND_TRUTH'
            report['warnings'].insert(0, 'WARNING: Shared tags are an imperfect proxy, not validated algorithm similarity.')
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf8')
        output_json({'source': report['source'], 'queries': report['n_queries'],
                     'results': {m: res['metrics'] for m, res in report['methods'].items()},
                     'saved': str(target)})


if __name__ == '__main__':
    main()
