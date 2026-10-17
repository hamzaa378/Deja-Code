# Déjà Code V2: Implementation decisions

1. **Most challenging part:** General Sentence-BERT preferred graph-related text over truly equivalent algorithms. We revised problem-focused indexing, added BM25 and rank-based fusion, and made neural re-ranking optional. This is a hypothesis to test, not proof of improvement.
2. **Main design choice:** Weighted reciprocal rank fusion rather than summing raw BM25, TF-IDF and cosine values. Their scores have different scales; ranks are more comparable. This still requires manual relevance validation.
3. **Known failure:** Problems can share story vocabulary and still require completely different algorithms. Rules and a generic reranker cannot prove algorithmic equivalence. Preprocessing can omit important late details. Clusters may not represent solution concepts.
4. **Next priority:** Annotate relevant pairs from the user's 1,474 CodeChef problems and compare V1 vs V2 with unbiased evaluations. Only then consider domain-specific fine-tuning.

This summary documents project decisions; the student should verify and rewrite it in their own words for the CodeChef-VIT optional DECISIONS.md submission.
