from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def rerank(self, query: str, chunks: list, top_n: int = 5) -> list:
        if not chunks:
            return chunks
        truncated = [c[:300] for c in chunks]
        pairs = [(query, c) for c in truncated]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, chunks), reverse=True)
        return [chunk for _, chunk in ranked[:top_n]]