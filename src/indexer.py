import chromadb # type: ignore
from chunker import chunk_repo
#stores data locally on disk
client = chromadb.PersistentClient(path="db")

# codebase
collection = client.get_or_create_collection("codebase")
chunks = chunk_repo("sample/flask")
collection.add(
    documents=[chunk.func_code for chunk in chunks],
    metadatas=[{"name": chunk.name, "file": chunk.file, "start_line": chunk.start_line, "end_line": chunk.end_line} for chunk in chunks],
    ids=[f"{chunk.file}::{chunk.name}::{chunk.start_line}" for chunk in chunks]
)


print(f"Stored {collection.count()} chunks")