import chromadb
import ollama
from chromadb import Documents, EmbeddingFunction, Embeddings
from chunker import chunk_repo

class OllamaEmbeddings(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            truncated = text[:1000]
            response = ollama.embeddings(model="nomic-embed-text", prompt=truncated)
            embeddings.append(response["embedding"])
        return embeddings

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="db")
    collection = client.get_or_create_collection(
        "codebase",
        embedding_function=OllamaEmbeddings()
    )

    chunks = chunk_repo("sample/flask")

    if collection.count() == 0:
        collection.add(
            documents=[chunk.func_code for chunk in chunks],
            metadatas=[{"name": chunk.name, "file": chunk.file, "start_line": chunk.start_line, "end_line": chunk.end_line} for chunk in chunks],
            ids=[f"{chunk.file}::{chunk.name}::{chunk.start_line}" for chunk in chunks]
        )

    results = collection.query(
        query_texts=["how does routing work"],
        n_results=5
    )

    for i, doc in enumerate(results["documents"][0]):
        print(f"--- Result {i+1} ---")
        print(results["metadatas"][0][i])
        print(doc[:200])
        print()