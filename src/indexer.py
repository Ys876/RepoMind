import chromadb # type: ignore
from chunker import chunk_repo
#stores data locally on disk
client = chromadb.PersistentClient(path="db")

# codebase
collection = client.get_or_create_collection("codebase")
chunks = chunk_repo("sample/flask")
if collection.count() == 0:
    collection.add(
        documents=[chunk.func_code for chunk in chunks],
        metadatas=[{"name": chunk.name, "file": chunk.file, "start_line": chunk.start_line, "end_line": chunk.end_line} for chunk in chunks],
        ids=[f"{chunk.file}::{chunk.name}::{chunk.start_line}" for chunk in chunks]
    )

results = collection.query(
    query_texts=["how does routing work"],
    n_results=5 #chromadb converts to vector and returns 5 most similar chunks
)

for i, doc in enumerate(results["documents"][0]):
    print(f"--- Result {i+1} ---")
    print(results["metadatas"][0][i]) #returned chromadb results 
    print(doc[:200])
    print()