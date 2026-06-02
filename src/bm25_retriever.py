from rank_bm25 import BM25Okapi
from tokenizer import tokenize

class BM25Retriever:
    def __init__(self, chunks):
        # only index source files, not tests
        self.chunks = [c for c in chunks if "/tests/" not in c.file and "test_" not in c.file]
        tokenized = [tokenize(chunk.func_code) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query: str, n: int = 20) -> list:
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [self.chunks[i] for i in top_indices]