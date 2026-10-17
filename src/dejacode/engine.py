"""Explainable retrieval baselines, optional sentence embeddings, and unsupervised clusters."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


class ProblemIndex:
    def __init__(self, documents: pd.DataFrame, *, neural_model: str | None = None,
                 neural_embeddings: np.ndarray | None = None):
        self.documents = documents.reset_index(drop=True).copy()
        self.word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                               max_features=65000, sublinear_tf=True,
                                               strip_accents='unicode', stop_words='english')
        self.char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5),
                                               min_df=1, max_features=50000,
                                               sublinear_tf=True)
        self.word_matrix = self.word_vectorizer.fit_transform(self.documents['search_text'])
        self.char_matrix = self.char_vectorizer.fit_transform(self.documents['search_text'])
        self.model_name = neural_model
        self.dense = neural_embeddings
        self.cluster_table: pd.DataFrame | None = None
        self.cluster_terms: dict[int, str] = {}

    def attach_embeddings(self, model: str = DEFAULT_MODEL, batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Sentence Transformers not installed. Run: pip install -e .[semantic]') from exc
        print(f'Loading semantic model {model} (first run downloads model weights)...')
        encoder = SentenceTransformer(model)
        vectors = encoder.encode(self.documents['search_text'].tolist(), batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=True,
                                 convert_to_numpy=True)
        self.dense = np.asarray(vectors, dtype=np.float32)
        self.model_name = model

    @property
    def has_semantic(self) -> bool:
        return self.dense is not None and self.model_name is not None

    def _tfidf_scores(self, query: str):
        words = (self.word_matrix @ self.word_vectorizer.transform([query]).T).toarray().ravel()
        chars = (self.char_matrix @ self.char_vectorizer.transform([query]).T).toarray().ravel()
        return 0.70 * words + 0.30 * chars

    def _dense_scores(self, query: str):
        if not self.has_semantic:
            raise ValueError('No semantic embeddings. Train again with --semantic.')
        from sentence_transformers import SentenceTransformer
        # Cache model in process; each query should not reload hundreds of MB.
        if not hasattr(self, '_encoder'):
            self._encoder = SentenceTransformer(self.model_name)
        vec = self._encoder.encode([query], normalize_embeddings=True,
                                   convert_to_numpy=True)[0]
        return np.asarray(self.dense @ vec, dtype=float)

    def search(self, query: str, *, method: str = 'hybrid', top_k: int = 5,
               exclude_code: str | None = None) -> list[dict]:
        if not query.strip():
            return []
        if method not in ('tfidf', 'semantic', 'hybrid'):
            raise ValueError('Choose tfidf, semantic or hybrid')
        lexical = self._tfidf_scores(query)
        if method == 'tfidf':
            scores = lexical
        elif method == 'semantic':
            scores = self._dense_scores(query)
        else:
            scores = 0.65 * self._dense_scores(query) + 0.35 * lexical if self.has_semantic else lexical
        scores = np.asarray(scores, dtype=float).copy()
        if exclude_code is not None:
            scores[self.documents['code'].astype(str).to_numpy() == str(exclude_code)] = -np.inf
        count = min(max(1, int(top_k)), len(scores))
        # Stable score sort makes results reproducible on tied scores.
        order = np.argsort(-scores, kind='stable')[:count]
        out = []
        for i in order:
            if not np.isfinite(scores[i]):
                continue
            row = self.documents.iloc[int(i)]
            label = None
            if self.cluster_table is not None:
                cluster = int(self.cluster_table.iloc[int(i)]['cluster'])
                label = {'id': cluster, 'keywords': self.cluster_terms.get(cluster, '')}
            out.append({'code': row['code'], 'title': row['title'],
                        'score': round(float(scores[i]), 5), 'tags': row['tags'],
                        'difficulty': row['difficulty'], 'url': row['url'],
                        'statement': row['statement'], 'cluster': label})
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

    def cluster(self, k: int = 8) -> dict:
        n = len(self.documents)
        if n < 3:
            raise ValueError('Need at least three problem statements for clustering')
        k = int(k)
        if not 2 <= k < n:
            raise ValueError(f'k must be between 2 and {n - 1}')
        matrix = self._cluster_matrix()
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)
        if len(set(labels)) < 2:
            raise ValueError('Clustering collapsed into one group; try another k or larger data')
        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
        values = []
        for idx in range(n):
            values.append({'code': str(self.documents.iloc[idx]['code']),
                           'title': str(self.documents.iloc[idx]['title']),
                           'cluster': int(labels[idx]), 'x': float(coords[idx, 0]),
                           'y': float(coords[idx, 1])})
        self.cluster_table = pd.DataFrame(values)
        term_array = self.word_vectorizer.get_feature_names_out()
        self.cluster_terms = {}
        for c in sorted(set(labels)):
            subset = self.word_matrix[labels == c]
            mean_weights = np.asarray(subset.mean(axis=0)).ravel()
            terms = [term_array[i] for i in np.argsort(-mean_weights)[:5] if mean_weights[i] > 0]
            self.cluster_terms[int(c)] = ', '.join(terms)
        sil = None
        if len(set(labels)) < n:
            sample = min(n, 1000)
            sil = float(silhouette_score(matrix, labels, sample_size=sample if sample < n else None,
                                         random_state=42))
        return {'n_problems': n, 'k': k, 'silhouette': sil,
                'cluster_sizes': {int(key): int(val) for key, val in
                                  pd.Series(labels).value_counts().sort_index().items()},
                'cluster_keywords': self.cluster_terms,
                'important_note': 'PCA shows approximate distances. Cluster keywords are extracted text terms, not verified algorithm labels.'}

    def save(self, directory: str | Path):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.documents.to_json(path / 'problems.json', orient='records', indent=2, force_ascii=False)
        joblib.dump({'word_vectorizer': self.word_vectorizer, 'char_vectorizer': self.char_vectorizer,
                     'word_matrix': self.word_matrix, 'char_matrix': self.char_matrix},
                    path / 'tfidf.joblib', compress=3)
        if self.has_semantic:
            np.save(path / 'semantic.npy', self.dense)
        else:
            (path / 'semantic.npy').unlink(missing_ok=True)
        if self.cluster_table is not None:
            self.cluster_table.to_csv(path / 'clusters.csv', index=False)
            (path / 'cluster_keywords.json').write_text(json.dumps(self.cluster_terms, indent=2), encoding='utf8')
        else:
            (path / 'clusters.csv').unlink(missing_ok=True)
            (path / 'cluster_keywords.json').unlink(missing_ok=True)
        meta = {'model': self.model_name if self.has_semantic else None,
                'n_problems': len(self.documents), 'version': 1,
                'source_fingerprint': hashlib.sha256(
                    self.documents[['code', 'search_text']].to_json(orient='records').encode()).hexdigest()}
        (path / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, directory: str | Path) -> 'ProblemIndex':
        """Load artifacts only when they were generated by your own trusted code/data."""
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
    ax.set_title('Déjà Code — PCA projection of problem embeddings')
    ax.set_xlabel('Principal component 1')
    ax.set_ylabel('Principal component 2')
    ax.legend(loc='best', fontsize=8, ncol=2)
    fig.tight_layout()
    path = Path(output)
    path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)
