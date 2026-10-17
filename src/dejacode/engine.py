"""Déjà Code V2: lexical baselines, semantic retrieval, weighted rank fusion,
optional cross-encoder reranking, and unsupervised concept exploration.

Metadata tags and solutions are NEVER retrieval features. Rules are transparent
heuristics over statement text only; they are not verified algorithm labels.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from .data import search_representation

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
DEFAULT_RERANKER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
DOMAIN_STOPWORDS = frozenset({
    'input', 'output', 'test', 'tests', 'case', 'cases', 'chef', 'codechef', 'problem',
    'statement', 'subtask', 'subtasks', 'example', 'examples', 'print', 'line',
    'lines', 'read', 'integer', 'integers', 'given', 'contains', 'consists',
    'respectively', 'constraints', 'denoting', 'denotes', 'space', 'separated',
    'points', 'available', 'submission', 'submissions', 'yes',
})
# Do NOT blindly discard content words such as array from the actual embeddings.
# Generic terms are filtered only for lexical matching and cluster keyword labels.
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | DOMAIN_STOPWORDS)

# Precision-oriented phrase hints, not a substitute for algorithm annotations.
# All signals come exclusively from problem descriptions and submitted queries.
CONCEPT_PATTERNS = {
    'shortest_path': [r'\bshortest\s+paths?\b', r'\bminimum\s+(?:total\s+)?(?:cost|distance)\s+paths?\b',
                      r'\bminimum\s+(?:cost|distance)\b.{0,55}\b(?:graph|path|route)\b'],
    'nonadjacent_selection': [r'\b(?:without|cannot|must not|no)\b.{0,65}\b(?:adjacent|consecutive|neighboring)\b',
                              r'\b(?:skip|avoiding)\b.{0,30}\b(?:adjacent|neighbors?)\b'],
    'dynamic_programming': [r'\bdynamic programming\b', r'\boptimal substructure\b', r'\bmemoization\b'],
    'range_queries': [r'\brange (?:minimum|maximum|sum|query|queries|update)\b', r'\bsegment tree\b',
                      r'\brange updates?\b'],
    'tree': [r'\b(?:rooted|binary|spanning) tree\b', r'\b(?:ancestor|lca)\b', r'\btrees?\b'],
    'graph': [r'\bgraphs?\b', r'\bvertices\b', r'\bedges?\b'],
    'string': [r'\bsubstrings?\b', r'\bpalindrom\w*\b', r'\bstrings?\b'],
    'knapsack': [r'\bknapsack\b', r'\b(?:capacity|budget)\b.{0,75}\b(?:maximum|maximize|profit|value)\b'],
    'binary_search': [r'\bbinary search\b', r'\bsearch space\b', r'\bmonotonic\b'],
    'prefix_sum': [r'\bprefix sums?\b', r'\bcumulative sum\b'],
    'minimum_spanning_tree': [r'\bminimum spanning tree\b', r'\b(?:kruskal|prim.s algorithm)\b'],
}
COMPILED_PATTERNS = {k: tuple(re.compile(p, re.I) for p in expressions)
                     for k, expressions in CONCEPT_PATTERNS.items()}


def detected_concepts(text: str) -> set[str]:
    if not isinstance(text, str):
        return set()
    return {name for name, patterns in COMPILED_PATTERNS.items()
            if any(p.search(text) for p in patterns)}


def reciprocal_rank_fusion(score_sets: dict[str, np.ndarray], *,
                           weights: dict[str, float] | None = None, constant: int = 60) -> np.ndarray:
    """Fuse ranks without treating lexical and cosine scores as comparable units."""
    if not score_sets:
        raise ValueError('No ranking sources supplied')
    scores = next(iter(score_sets.values()))
    total = np.zeros(len(scores), dtype=np.float64)
    weights = weights or {method: 1.0 for method in score_sets}
    for method, raw in score_sets.items():
        ranking = np.argsort(-np.asarray(raw), kind='stable')
        positions = np.empty(len(ranking), dtype=np.float64)
        positions[ranking] = np.arange(1, len(ranking) + 1)
        total += max(0.0, weights.get(method, 0.0)) / (constant + positions)
    return total


class ProblemIndex:
    def __init__(self, documents: pd.DataFrame, *, neural_model: str | None = None,
                 neural_embeddings: np.ndarray | None = None):
        self.documents = documents.reset_index(drop=True).copy()
        self.documents['search_text'] = [
            search_representation(str(row['title']), str(row['statement']))
            for _, row in self.documents.iterrows()
        ]
        texts = self.documents['search_text'].tolist()
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=65000, sublinear_tf=True,
            strip_accents='unicode', stop_words=STOP_WORDS)
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb', ngram_range=(3, 5), min_df=1, max_features=50000,
            sublinear_tf=True)
        self.bm25_vectorizer = CountVectorizer(
            ngram_range=(1, 2), min_df=1, max_features=65000,
            strip_accents='unicode', stop_words=STOP_WORDS)
        self.word_matrix = self.word_vectorizer.fit_transform(texts)
        self.char_matrix = self.char_vectorizer.fit_transform(texts)
        self.bm25_matrix = self.bm25_vectorizer.fit_transform(texts).tocsr()
        self.doc_lengths = np.asarray(self.bm25_matrix.sum(axis=1)).ravel().astype(np.float64)
        self.mean_doc_length = max(1e-8, float(self.doc_lengths.mean()))
        dfs = np.asarray((self.bm25_matrix > 0).sum(axis=0)).ravel()
        self.bm25_idf = np.log1p((len(self.documents) - dfs + 0.5) / (dfs + 0.5))
        self.concept_signals = [detected_concepts(t) for t in texts]
        self.model_name = neural_model
        self.dense = neural_embeddings
        self.cluster_table: pd.DataFrame | None = None
        self.cluster_terms: dict[int, str] = {}
        self.version = 2

    def attach_embeddings(self, model: str = DEFAULT_MODEL, batch_size: int = 32):
        from sentence_transformers import SentenceTransformer
        print(f'Loading semantic model {model} (download only needed on first run)...')
        self._encoder = SentenceTransformer(model)
        vectors = self._encoder.encode(self.documents['search_text'].tolist(),
                                       batch_size=batch_size, normalize_embeddings=True,
                                       show_progress_bar=True, convert_to_numpy=True)
        self.dense = np.asarray(vectors, dtype=np.float32)
        self.model_name = model

    @property
    def has_semantic(self) -> bool:
        return self.dense is not None and self.model_name is not None

    @property
    def has_bm25(self) -> bool:
        return hasattr(self, 'bm25_matrix')

    def _tfidf_scores(self, query: str) -> np.ndarray:
        words = (self.word_matrix @ self.word_vectorizer.transform([query]).T).toarray().ravel()
        chars = (self.char_matrix @ self.char_vectorizer.transform([query]).T).toarray().ravel()
        return np.asarray(0.75 * words + 0.25 * chars, dtype=np.float64)

    def _bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        if not self.has_bm25:
            raise ValueError('BM25 not available in old artifact. Build a v2 index.')
        q = self.bm25_vectorizer.transform([query])
        if q.nnz == 0:
            return np.zeros(len(self.documents), dtype=np.float64)
        scores = np.zeros(len(self.documents), dtype=np.float64)
        normalizer = k1 * (1 - b + b * self.doc_lengths / self.mean_doc_length)
        for col in q.indices:
            counts = self.bm25_matrix.getcol(int(col)).toarray().ravel()
            scores += self.bm25_idf[col] * (counts * (k1 + 1)) / (counts + normalizer + 1e-12)
        return scores

    def _dense_scores(self, query: str) -> np.ndarray:
        if not self.has_semantic:
            raise ValueError('No semantic embeddings. Build with --semantic.')
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, '_encoder'):
            self._encoder = SentenceTransformer(self.model_name)
        vec = self._encoder.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True)[0]
        return np.asarray(self.dense @ vec, dtype=np.float64)

    def _base_scores(self, query: str, method: str) -> np.ndarray:
        if method == 'tfidf':
            return self._tfidf_scores(query)
        if method == 'bm25':
            return self._bm25_scores(query)
        if method == 'semantic':
            return self._dense_scores(query)
        # V2 hybrid; older indexes continue to work with legacy score fusion.
        if method != 'hybrid':
            raise ValueError('Choose tfidf, bm25, semantic or hybrid')
        if not self.has_bm25:
            lexical = self._tfidf_scores(query)
            return 0.65 * self._dense_scores(query) + 0.35 * lexical if self.has_semantic else lexical
        sources = {'bm25': self._bm25_scores(query), 'tfidf': self._tfidf_scores(query)}
        weights = {'bm25': 0.43, 'tfidf': 0.17}
        if self.has_semantic:
            sources['semantic'] = self._dense_scores(query)
            weights['semantic'] = 0.40
        scores = reciprocal_rank_fusion(sources, weights=weights)
        # Small explanatory agreement bonus, ONLY if both query and doc
        # explicitly contain the same algorithm cue. Capped to protect ranking.
        qconcept = detected_concepts(query)
        if qconcept:
            signals = getattr(self, 'concept_signals', None)
            if signals is None:
                signals = [detected_concepts(t) for t in self.documents['search_text']]
            overlaps = np.array([
                len((qconcept & concepts) - {'graph', 'tree', 'string'})
                for concepts in signals], dtype=np.float64)
            scores += 0.0006 * np.clip(overlaps, 0, 2)
        return scores

    def search(self, query: str, *, method: str = 'hybrid', top_k: int = 5,
               exclude_code: str | None = None, rerank: bool = False,
               rerank_pool: int = 30, rerank_model: str = DEFAULT_RERANKER,
               include_statement: bool = True) -> list[dict]:
        if not query.strip():
            return []
        if method not in ('tfidf', 'bm25', 'semantic', 'hybrid'):
            raise ValueError('Choose tfidf, bm25, semantic or hybrid')
        scores = np.asarray(self._base_scores(query, method), dtype=np.float64).copy()
        if exclude_code is not None:
            scores[self.documents['code'].astype(str).to_numpy() == str(exclude_code)] = -np.inf
        count = min(max(1, int(top_k)), len(scores))
        if rerank:
            # A cross-encoder examines the query and candidate statement jointly;
            # this is an OPTIONAL second stage requiring a model download.
            from sentence_transformers import CrossEncoder
            if not hasattr(self, '_rerankers'):
                self._rerankers = {}
            if rerank_model not in self._rerankers:
                self._rerankers[rerank_model] = CrossEncoder(rerank_model)
            shortlist = np.argsort(-scores, kind='stable')[:max(count, int(rerank_pool))]
            shortlist = [int(i) for i in shortlist if np.isfinite(scores[i])]
            pairs = [(query, str(self.documents.iloc[i]['search_text'])) for i in shortlist]
            neural = np.asarray(self._rerankers[rerank_model].predict(pairs), dtype=np.float64).reshape(-1)
            order_local = np.argsort(-neural, kind='stable')
            order = np.array(shortlist, dtype=int)[order_local][:count]
            display_scores = {int(shortlist[j]): float(neural[j]) for j in range(len(shortlist))}
        else:
            order = np.argsort(-scores, kind='stable')[:count]
            display_scores = {}
        qconcept = detected_concepts(query)
        out = []
        for i in order:
            i = int(i)
            if not np.isfinite(scores[i]):
                continue
            row = self.documents.iloc[i]
            cluster_label = None
            if self.cluster_table is not None:
                cluster = int(self.cluster_table.iloc[i]['cluster'])
                cluster_label = {'id': cluster, 'keywords': self.cluster_terms.get(cluster, '')}
            signals = self.concept_signals[i] if hasattr(self, 'concept_signals') else detected_concepts(row['search_text'])
            result = {
                'code': str(row['code']), 'title': str(row['title']),
                'score': round(float(display_scores.get(i, scores[i])), 6),
                'score_type': 'cross_encoder_logit' if rerank else (
                    'rrf_rank_score' if method == 'hybrid' and self.has_bm25 else method + '_score'),
                'matched_concept_hints': sorted(qconcept & signals),
                'tags': row['tags'], 'difficulty': row['difficulty'],
                'url': row['url'], 'cluster': cluster_label,
            }
            if include_statement:
                result['statement'] = row['statement']
            out.append(result)
        return out

    def _cluster_matrix(self) -> np.ndarray:
        n = len(self.documents)
        if self.has_semantic:
            return np.asarray(self.dense, dtype=np.float32)
        if n <= 2:
            return np.asarray(self.word_matrix.toarray(), dtype=np.float32)
        max_components = min(60, n - 1, self.word_matrix.shape[1] - 1)
        if max_components < 2:
            return np.asarray(self.word_matrix.toarray(), dtype=np.float32)
        return TruncatedSVD(n_components=max_components, random_state=42).fit_transform(self.word_matrix)

    def cluster(self, k: int = 8, *, auto_k: bool = False) -> dict:
        n = len(self.documents)
        if n < 3:
            raise ValueError('Need at least 3 problem statements for clustering')
        k = min(int(k), n - 1)
        if not 2 <= k < n:
            raise ValueError(f'k must be between 2 and {n - 1}')
        matrix = self._cluster_matrix()
        trials = []
        if auto_k:
            choices = sorted(set(max(2, min(n - 1, q)) for q in
                                 [max(2, k-4), max(2, k-2), k, k+2, k+4]))
            best_score, best_labels, best_k = -np.inf, None, None
            for candidate in choices:
                labels = KMeans(n_clusters=candidate, n_init=5, random_state=42).fit_predict(matrix)
                if len(set(labels)) < 2:
                    continue
                s = float(silhouette_score(matrix, labels, sample_size=min(n, 700), random_state=42))
                trials.append({'k': candidate, 'silhouette_sample': round(s, 6)})
                if s > best_score:
                    best_score, best_labels, best_k = s, labels, candidate
            if best_labels is None:
                raise ValueError('All clusters collapsed into one')
            labels, k = best_labels, best_k
        else:
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)
        if len(set(labels)) < 2:
            raise ValueError('Clustering collapsed into one group')
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
        self.cluster_table = pd.DataFrame([{
            'code': str(self.documents.iloc[i]['code']),
            'title': str(self.documents.iloc[i]['title']),
            'cluster': int(labels[i]), 'x': float(coords[i, 0]),
            'y': float(coords[i, 1])} for i in range(n)])
        term_array = self.word_vectorizer.get_feature_names_out()
        self.cluster_terms = {}
        for c in sorted(set(labels)):
            subset = self.word_matrix[labels == c]
            mean_weights = np.asarray(subset.mean(axis=0)).ravel()
            terms = [str(term_array[i]) for i in np.argsort(-mean_weights)[:6] if mean_weights[i] > 0]
            self.cluster_terms[int(c)] = ', '.join(terms)
        sil = float(silhouette_score(matrix, labels, sample_size=min(n, 1000), random_state=42))
        report = {'n_problems': n, 'k': k, 'silhouette': sil,
                  'davies_bouldin': float(davies_bouldin_score(matrix, labels)),
                  'calinski_harabasz': float(calinski_harabasz_score(matrix, labels)),
                  'cluster_sizes': {int(key): int(value) for key, value in
                                    pd.Series(labels).value_counts().sort_index().items()},
                  'cluster_keywords': self.cluster_terms,
                  'auto_k_trials': trials,
                  'important_note': 'Scores quantify embedding geometry, not correctness of algorithm labels. PCA and keywords are exploratory.'}
        return report

    def save(self, directory: str | Path):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.documents.to_json(path / 'problems.json', orient='records', indent=2, force_ascii=False)
        blocks = {name: getattr(self, name) for name in (
            'word_vectorizer', 'char_vectorizer', 'word_matrix', 'char_matrix',
            'bm25_vectorizer', 'bm25_matrix', 'doc_lengths', 'mean_doc_length', 'bm25_idf')
                  if hasattr(self, name)}
        joblib.dump(blocks, path / 'tfidf.joblib', compress=3)
        if self.has_semantic:
            np.save(path / 'semantic.npy', self.dense)
        else:
            (path / 'semantic.npy').unlink(missing_ok=True)
        if self.cluster_table is not None:
            self.cluster_table.to_csv(path / 'clusters.csv', index=False)
            (path / 'cluster_keywords.json').write_text(
                json.dumps(self.cluster_terms, indent=2), encoding='utf8')
        else:
            (path / 'clusters.csv').unlink(missing_ok=True)
            (path / 'cluster_keywords.json').unlink(missing_ok=True)
        meta = {'model': self.model_name if self.has_semantic else None,
                'n_problems': len(self.documents), 'version': 2,
                'retrieval': 'BM25 + TF-IDF + Sentence-BERT weighted RRF + optional cross encoder',
                'source_fingerprint': hashlib.sha256(
                    self.documents[['code', 'search_text']].to_json(orient='records').encode()).hexdigest()}
        (path / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf8')

    @classmethod
    def load(cls, directory: str | Path) -> 'ProblemIndex':
        """Load ONLY artifacts produced by your own trusted run; joblib uses pickle."""
        path = Path(directory)
        meta = json.loads((path / 'meta.json').read_text(encoding='utf8'))
        df = pd.read_json(path / 'problems.json', orient='records').fillna('')
        if len(df) != meta['n_problems']:
            raise ValueError('Artifact metadata count mismatch')
        obj = cls.__new__(cls)
        obj.documents = df
        blocks = joblib.load(path / 'tfidf.joblib')
        for name, value in blocks.items():
            setattr(obj, name, value)
        obj.model_name = meta.get('model')
        obj.dense = np.load(path / 'semantic.npy', allow_pickle=False) if obj.model_name else None
        obj.cluster_table = pd.read_csv(path / 'clusters.csv') if (path / 'clusters.csv').exists() else None
        obj.cluster_terms = {int(k): str(v) for k, v in
                             json.loads((path / 'cluster_keywords.json').read_text(encoding='utf8')).items()} \
                            if (path / 'cluster_keywords.json').exists() else {}
        obj.version = int(meta.get('version', 1))
        obj.concept_signals = [detected_concepts(t) for t in obj.documents['search_text']]
        return obj


def draw_clusters(index: ProblemIndex, output: str | Path):
    if index.cluster_table is None:
        raise ValueError('Run cluster() first')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tab = index.cluster_table
    fig, ax = plt.subplots(figsize=(11, 8))
    for c, part in tab.groupby('cluster'):
        ax.scatter(part['x'], part['y'], label=f'Cluster {c} ({len(part)})', s=30, alpha=.75)
    ax.set_title('Déjà Code V2 — PCA projection of problem embeddings')
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.legend(loc='best', fontsize=8, ncol=2)
    fig.tight_layout()
    path = Path(output)
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
