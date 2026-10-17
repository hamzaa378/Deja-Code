# Project technical decisions — to customize in your own words

The task sheet suggests explaining four decisions. These are **starting points** for your own reflection; personalize them after using the system, especially after seeing the CodeChef results.

## 1. What was technically hardest?

The hardest part is defining similarity: similar story text is not necessarily the same algorithm. TF-IDF finds familiar vocabulary, but two stories about a route may require different methods (BFS vs Dijkstra) based on edge weights. I handled that by implementing a lexical baseline and an optional semantic embedding model separately, so I can inspect whether semantic ranking actually makes better problem-solving matches. The evaluation explicitly excludes the query problem itself.

## 2. What design decision mattered most?

I use normalized sentence embeddings plus cosine ranking rather than attempting supervised end-to-end training on the small available corpus. A pretrained encoder allows useful retrieval without having hundreds of hand-labeled examples. The hybrid model adds a lexical component because rare technical phrases can matter. I chose a simple, auditable weighted blend rather than a complicated model whose gains cannot yet be verified.

## 3. What limitation remains?

Semantic language similarity does not ensure identical asymptotic complexity, constraints, or algorithmic technique. K-Means groups sometimes depend on problem narratives instead of technical ideas, and their labels are generated from text terms. The included small demonstration has weak cluster separation and should not be confused with a real-data benchmark.

## 4. What would I improve first?

I would create a manually judged set of same-algorithm/different-story problem pairs, together with hard negative pairs that sound alike but need different algorithms. Then I would run TF-IDF, MiniLM, and hybrid search against those exact labels, examine false positives, and fine-tune or rerank only if the comparison shows a measurable benefit.

## Questions you should be ready to answer

1. **Why normalize embeddings?** Unit vectors make the dot product equal cosine similarity, ensuring magnitude does not dominate rankings.
2. **Why exclude the query code?** Otherwise a problem statement retrieved from its own corpus creates artificial perfect matches.
3. **Is cosine 0.8 an 80% probability?** No. Cosine is a ranking similarity score, not a calibrated probability.
4. **How is retrieval different from clustering?** Retrieval ranks relevant neighbors for one query; clustering assigns corpus-wide group membership, sometimes oversimplifying multi-concept problems.
5. **Why train K-Means after computing embeddings?** Embeddings create continuous representations; K-Means fits centers in that space to find broad groupings.
6. **What does a negative silhouette indicate?** Under the chosen distance metric, many examples are closer to another cluster than their assigned one, suggesting overlapping or inappropriate groups.
7. **Why isn't shared-tag evaluation ground truth?** Tags can be missing, inaccurate, or broadly conceptual; sharing a tag does not prove the same core algorithm.
8. **What is the key risk with MiniLM?** General-language models can be sensitive to narrative similarity and miss programming-specific constraints or algorithms.
