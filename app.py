"""Optional interactive Streamlit interface; run: streamlit run app.py."""
from pathlib import Path
import pandas as pd
import streamlit as st

from dejacode.engine import ProblemIndex

st.set_page_config(page_title='Déjà Code', page_icon='🔍', layout='wide')
st.title('Déjà Code')
st.caption('Find programming problems with similar underlying ideas. Clusters are exploratory, not proven algorithm labels.')

@st.cache_resource(show_spinner='Loading retrieval index...')
def load_index(path: str):
    return ProblemIndex.load(path)

index_dir = st.sidebar.text_input('Index directory', 'artifacts')
if not (Path(index_dir) / 'meta.json').exists():
    st.warning('No index found. First run `dejacode build --data data/demo/demo_problems.csv --out artifacts`.')
    st.stop()

index = load_index(index_dir)
st.sidebar.success(f'{len(index.documents):,} indexed problems')
st.sidebar.info('Semantic search available' if index.has_semantic else 'TF-IDF only. Add --semantic for Sentence-BERT.')
page = st.sidebar.radio('Explore', ['Search', 'Concept clusters', 'Dataset and methods'])

if page == 'Search':
    sample = ('There are intersections connected by undirected streets, all taking equal travel time. '
              'Find the minimum number of streets to cross to reach a destination.')
    query = st.text_area('Paste a full programming problem statement or describe the concept',
                         value=sample, height=145)
    left, right = st.columns([2, 1])
    method = left.selectbox('Retrieval method', ['hybrid', 'semantic', 'tfidf']
                            if index.has_semantic else ['tfidf'])
    top_k = right.slider('Top matches', 1, 20, 5)
    if st.button('Find similar problems', type='primary'):
        with st.spinner('Ranking statements...'):
            results = index.search(query, method=method, top_k=top_k)
        if not results:
            st.info('Enter a problem description to search.')
        for rank, result in enumerate(results, start=1):
            with st.container(border=True):
                st.subheader(f"{rank}. {result['title']}  ·  {result['code']}")
                st.caption(f"Similarity score: {result['score']:.4f} (relative ranking score, not probability)"
                           + (f" | Difficulty: {result['difficulty']}" if result['difficulty'] else ''))
                if result['tags']:
                    st.write('Metadata tags: ' + ', '.join(result['tags']))
                if result['cluster']:
                    st.write('Cluster: ' + result['cluster']['keywords'])
                if str(result['url']).startswith('https://'):
                    st.markdown(f"[View original problem]({result['url']})")
                with st.expander('Problem statement'):
                    st.write(result['statement'])

elif page == 'Concept clusters':
    if index.cluster_table is None:
        st.warning('Build a clustered index to explore the concept map.')
    else:
        import plotly.express as px
        frame = index.cluster_table.copy()
        frame['cluster'] = frame['cluster'].astype(str)
        frame['keywords'] = frame['cluster'].map(lambda c: index.cluster_terms.get(int(c), ''))
        fig = px.scatter(frame, x='x', y='y', color='cluster', hover_data=['code', 'title', 'keywords'],
                         title='2D PCA projection of problems (visualization only)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('PCA compresses high-dimensional vectors. Apparent overlaps/distances are not proof of semantic equivalence.')
        counts = index.cluster_table['cluster'].value_counts().sort_index()
        st.dataframe(pd.DataFrame({'cluster': counts.index, 'count': counts.values,
                                   'textual_theme': [index.cluster_terms.get(int(c), '') for c in counts.index]}),
                     hide_index=True, use_container_width=True)
        chosen = st.selectbox('Inspect problems in cluster', sorted(index.cluster_table['cluster'].unique()))
        selected = index.cluster_table[index.cluster_table['cluster'] == chosen]
        st.dataframe(selected[['code', 'title']], hide_index=True, use_container_width=True)
else:
    st.write('The index searches title + cleaned statement only; tags are used for display or optional weak-label evaluation, not as search input.')
    st.write('TF-IDF combines word/bigram and character n-gram similarity. Semantic search uses normalized Sentence-BERT sentence embeddings. Hybrid search blends both scores.')
    st.write('K-Means clusters embeddings. PCA projects them to two dimensions. Top TF-IDF terms provide tentative descriptions of each cluster.')
    st.write('For rigorous retrieval evaluation, manually judge related problem pairs and compare Precision@K, Recall@K, MRR@K, MAP@K, and NDCG@K.')
    st.caption('Do not treat a high cosine similarity or a shared tag as proof that two problems use the same algorithm.')
