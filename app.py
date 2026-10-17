"""Streamlit V2 UI. Run from project root: streamlit run app.py."""
from pathlib import Path

import pandas as pd
import streamlit as st

from dejacode.engine import ProblemIndex

st.set_page_config(page_title='Déjà Code V2', page_icon='🔎', layout='wide')
st.title('Déjà Code · Algorithmic Problem Discovery')
st.caption('Search, compare, and explore CodeChef problems. Similarity is evidence for review, not proof of the same solution algorithm.')


@st.cache_resource(show_spinner='Loading retrieval index…')
def load_index(path: str, mtime: float):
    return ProblemIndex.load(path)


index_dir = st.sidebar.text_input('Index directory', 'artifacts/codechef_v2')
meta = Path(index_dir) / 'meta.json'
if not meta.exists():
    st.warning('Index not found. Build the V2 index and enter its directory in the sidebar.')
    st.code('python -m dejacode.cli build --data "C:\\path\\to\\questions.csv" --semantic --clusters 10 --out artifacts/codechef_v2')
    st.stop()

index = load_index(index_dir, meta.stat().st_mtime)
st.sidebar.success(f'{len(index.documents):,} indexed problems · V{index.version}')
st.sidebar.info('Semantic embeddings available' if index.has_semantic else 'Lexical-only index (no semantic embeddings)')
if not index.has_bm25:
    st.sidebar.warning('Legacy index. Rebuild V2 to enable BM25 and rank fusion.')
page = st.sidebar.radio('Explore', ['Search', 'Compare retrieval methods', 'Concept clusters', 'Methods and limitations'])


def show_matches(rows):
    if not rows:
        st.info('No results; enter a problem statement or shorter concept description.')
        return
    for rank, r in enumerate(rows, 1):
        with st.container(border=True):
            st.subheader(f'{rank}. {r["title"]}')
            st.caption(f'ID: {r["code"]}  ·  Ranking signal: {r["score"]:.5f} ({r.get("score_type", "score")})'
                       + (f'  ·  Difficulty: {r["difficulty"]}' if r.get('difficulty') else ''))
            if r.get('matched_concept_hints'):
                st.write('Matching text clues (not verified tags): ' + ', '.join(r['matched_concept_hints']))
            if isinstance(r.get('tags'), list) and r['tags']:
                st.caption('Dataset tags (not used for ranking): ' + ', '.join(r['tags']))
            if r.get('cluster'):
                st.caption('Cluster terms: ' + r['cluster']['keywords'])
            url = str(r.get('url', ''))
            if url.startswith('/') and url.count('/') >= 2:
                url = 'https://www.codechef.com' + url
            if url.startswith('https://'):
                st.markdown(f'[View problem]({url})')
            if 'statement' in r:
                with st.expander('Read full statement'):
                    st.write(r['statement'])


methods = ['tfidf']
if index.has_bm25:
    methods.append('bm25')
if index.has_semantic:
    methods.append('semantic')
methods.append('hybrid')

if page == 'Search':
    q = st.text_area('Describe a problem or paste its statement',
        value='Find the minimum cost path between two vertices in a weighted graph', height=140)
    c1, c2, c3 = st.columns([2, 1, 1])
    method = c1.selectbox('Retrieval method', methods, index=len(methods) - 1)
    top_k = c2.slider('Top matches', 1, 20, 5)
    rerank = c3.checkbox('Cross-encoder rerank', value=False,
        help='Downloads an additional model on first use. Slower but may improve fine-grained relevance.')
    if st.button('Find related problems', type='primary'):
        with st.spinner('Scoring…'):
            try:
                matches = index.search(q, method=method, top_k=top_k, rerank=rerank)
                show_matches(matches)
            except Exception as exc:
                st.error(f'Search failed: {exc}')
    st.caption('Methods have different score scales. Do not compare raw scores across methods.')

elif page == 'Compare retrieval methods':
    q = st.text_area('Use exactly the same query for each retrieval method',
                     'Choose elements to maximize their sum without selecting neighboring indices', height=110)
    k = st.slider('Top results per method', 2, 10, 5)
    rerank = st.checkbox('Also compare cross-encoder reranked hybrid results', value=False)
    if st.button('Compare methods', type='primary'):
        variants = list(methods) + (['hybrid + reranker'] if rerank else [])
        cols = st.columns(min(3, len(variants)))
        for j, method in enumerate(variants):
            with cols[j % len(cols)]:
                st.markdown(f'**{method}**')
                with st.spinner('Ranking…'):
                    try:
                        rows = index.search(q, method='hybrid' if method.startswith('hybrid +') else method,
                                            rerank=method.startswith('hybrid +'), top_k=k,
                                            include_statement=False)
                        st.dataframe(pd.DataFrame([{'Rank': i, 'Title': x['title'], 'Score': x['score']}
                                                  for i, x in enumerate(rows, 1)]),
                                     hide_index=True, use_container_width=True)
                    except Exception as exc:
                        st.error(str(exc))
    st.warning('Compare the retrieved algorithms manually or using the evaluation judgments. High scores alone cannot establish a correct match.')

elif page == 'Concept clusters':
    if index.cluster_table is None:
        st.warning('Index has no clusters; build an index first.')
    else:
        import plotly.express as px
        frame = index.cluster_table.copy()
        frame['cluster'] = frame['cluster'].astype(str)
        frame['keywords'] = frame['cluster'].map(lambda x: index.cluster_terms.get(int(x), ''))
        fig = px.scatter(frame, x='x', y='y', color='cluster', hover_data=['code', 'title', 'keywords'],
                         title='Cluster map (PCA projection)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('Overlapping groups are expected. The display is a 2D projection, not a correctness test.')
        counts = frame['cluster'].value_counts().sort_index()
        st.dataframe(pd.DataFrame({'Cluster': counts.index, 'Problems': counts.values,
                                   'Possible topic words': [index.cluster_terms.get(int(c), '') for c in counts.index]}),
                     hide_index=True, use_container_width=True)
        choice = st.selectbox('Inspect one cluster', sorted(index.cluster_table['cluster'].unique()))
        st.dataframe(index.cluster_table[index.cluster_table['cluster'] == choice][['code', 'title']],
                     hide_index=True, use_container_width=True)
else:
    st.markdown('''
### Pipeline
1. Extract and clean problem descriptions, preserving original statement separately.
2. Build TF-IDF, BM25 and optional Sentence-BERT indexes from **statement text only**.
3. Combine rank positions using weighted reciprocal rank fusion, with limited high-precision text clues.
4. Optionally cross-encode shortlisted query/problem pairs for deeper matching.
5. Cluster dense embeddings using K-Means; show geometry and tentative frequent terms.

### Limitations
- No result is guaranteed to share the same algorithm as the query.
- Regex concept clues can miss paraphrases and can produce false matches.
- Cross-encoders trained for general search do not inherently know algorithmic equivalence.
- Human-judged retrieval metrics are required; dataset tags are a **weak proxy** only.
- This tool does not use submitted solutions for ranking.
- Ranking scores use different scales; do not interpret them as probabilities.
''')
