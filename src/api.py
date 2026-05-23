from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import git
import os
import shutil
import chromadb
from chunker import chunk_repo
from agent import run_agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
class IndexRequest(BaseModel):
    github_url: str

class AskRequest(BaseModel):
    question: str

@app.post("/index")
def index_repo(request: IndexRequest):
    repo_path = "sample/temp_repo"
    
    # delete old temp repo if exists
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)
    
    # clone the repo
    try:
        git.Repo.clone_from(request.github_url, repo_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to clone repo: {str(e)}")
    
    # chunk it
    chunks = chunk_repo(repo_path)
    
    if not chunks:
        raise HTTPException(status_code=400, detail="No code chunks found in repo")
    
    # clear old collection and reindex
    client = chromadb.PersistentClient(path="db")
    client.delete_collection("codebase")
    
    from indexer import OllamaEmbeddings
    new_collection = client.create_collection(
        "codebase",
        embedding_function=OllamaEmbeddings()
    )
    
    new_collection.add(
        documents=[chunk.func_code for chunk in chunks],
        metadatas=[{"name": chunk.name, "file": chunk.file, "start_line": chunk.start_line, "end_line": chunk.end_line} for chunk in chunks],
        ids=[f"{chunk.file}::{chunk.name}::{chunk.start_line}" for chunk in chunks]
    )
    
    return {"message": f"Indexed {len(chunks)} chunks from {request.github_url}"}

@app.post("/ask")
def ask_question(request: AskRequest):
    answer = run_agent(request.question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)