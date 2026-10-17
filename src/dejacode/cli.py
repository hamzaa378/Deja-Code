"""CLI entrypoint. Run `dejacode --help` after pip install -e . ."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_problems
from .engine import DEFAULT_MODEL, ProblemIndex, draw_clusters
from .evaluation import build_weak_tag_judgments, evaluate, read_judgments


def main():
    parser = argparse.ArgumentParser(prog='dejacode', description='Déjà Code: semantic search and clustering')
    sub = parser.add_subparsers(dest='command', required=True)
    inspect = sub.add_parser('inspect', help='Inspect normalized dataset and inferred schema')
    inspect.add_argument('--data', required=True, help='Source CSV/JSON/JSONL/parquet or folder')
    inspect.add_argument('--statement-col')
    inspect.add_argument('--id-col')
    inspect.add_argument('--title-col')
    inspect.add_argument('--limit', type=int)

    train = sub.add_parser('build', help='Preprocess, fit retrieval indexes, cluster and save')
    train.add_argument('--data', required=True)
    train.add_argument('--out', default='artifacts')
    train.add_argument('--semantic', action='store_true', help='Download and encode with Sentence-BERT')
    train.add_argument('--model', default=DEFAULT_MODEL)
    train.add_argument('--clusters', type=int, default=8)
    train.add_argument('--limit', type=int)
    train.add_argument('--statement-col')
    train.add_argument('--id-col')
    train.add_argument('--title-col')

    find = sub.add_parser('search', help='Query the index (statement text or short description)')
    find.add_argument('--index', default='artifacts')
    find.add_argument('--query', required=True)
    find.add_argument('--method', choices=('tfidf', 'semantic', 'hybrid'), default='hybrid')
    find.add_argument('--top-k', type=int, default=5)
    find.add_argument('--exclude-code')

    ev = sub.add_parser('evaluate', help='Score retrieval against human judgments or weak tags')
    ev.add_argument('--index', default='artifacts')
    ev.add_argument('--judgments', help='JSONL with query, relevant_codes[, query_code]')
    ev.add_argument('--weak-tags', action='store_true', help='Proxy relevance ONLY; not human ground truth')
    ev.add_argument('--max-queries', type=int, default=100)
    ev.add_argument('--top-k', type=int, default=5)
    ev.add_argument('--out', default='artifacts/retrieval_evaluation.json')

    args = parser.parse_args()
    if args.command in ('inspect', 'build'):
        data = load_problems(args.data, statement_col=args.statement_col,
                             id_col=args.id_col, title_col=args.title_col, limit=args.limit)
        print(json.dumps({'problems': len(data), 'input_diagnostics': data.attrs['diagnostics'],
                          'columns': list(data.columns), 'has_tags': int(data['tags'].map(bool).sum()),
                          'sample': data[['code', 'title', 'difficulty']].head(4).to_dict('records')}, indent=2))
        if args.command == 'inspect':
            return
        if len(data) < 3:
            raise ValueError('At least 3 usable problem statements are needed for clustering.')
        index = ProblemIndex(data)
        if args.semantic:
            index.attach_embeddings(args.model)
        cluster_report = index.cluster(k=min(args.clusters, len(data) - 1))
        target = Path(args.out)
        index.save(target)
        draw_clusters(index, target / 'cluster_map.png')
        (target / 'cluster_report.json').write_text(json.dumps(cluster_report, indent=2), encoding='utf8')
        print(json.dumps({'index': str(target), 'semantic': index.has_semantic,
                          'clusters': cluster_report}, indent=2))
    elif args.command == 'search':
        index = ProblemIndex.load(args.index)
        if args.method == 'semantic' and not index.has_semantic:
            parser.error('Semantic model not built. Re-run `build --semantic` first.')
        print(json.dumps(index.search(args.query, method=args.method, top_k=args.top_k,
                                      exclude_code=args.exclude_code), indent=2, ensure_ascii=False))
    elif args.command == 'evaluate':
        index = ProblemIndex.load(args.index)
        if bool(args.judgments) == bool(args.weak_tags):
            parser.error('Choose one of --judgments FILE or --weak-tags')
        judgments = read_judgments(args.judgments) if args.judgments else \
                    build_weak_tag_judgments(index, max_queries=args.max_queries)
        if not judgments:
            parser.error('No valid relevant pairs. Provide human annotation or tags with shared concepts.')
        report = evaluate(index, judgments, k=args.top_k)
        if args.weak_tags:
            report['source'] = 'WEAK_SHARED_TAG_PROXY_NOT_GROUND_TRUTH'
            report['warnings'].insert(0, 'WARNING: This evaluation is tag overlap, not real algorithmic relevance.')
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2), encoding='utf8')
        print(json.dumps({'source': report['source'], 'queries': report['n_queries'],
                          'results': {m: res['metrics'] for m, res in report['methods'].items()},
                          'saved': str(target)}, indent=2))


if __name__ == '__main__':
    main()
