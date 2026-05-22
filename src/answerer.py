import chromadb
import ollama
from indexer import OllamaEmbeddings

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(
    "codebase",
    embedding_function=OllamaEmbeddings()
)

def answer(question):
    results = collection.query(
        query_texts=[question],
        n_results=5
    )
    
    chunks_text = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        chunks_text += f"\n--- {meta['file']}:{meta['start_line']} ---\n{doc}\n"
    
    prompt = f"""You are a code assistant. Use the code chunks below to answer the question.
Cite file and line numbers like (file.py:L34) for every claim.
Only use information from the chunks. If something is absent, say so.

CODE CHUNKS:
{chunks_text}

QUESTION: {question}
ANSWER:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3}
    )
    
    return response["message"]["content"]


print(answer("explain simply what the entire repository does"))