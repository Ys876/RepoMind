import chromadb
from langchain.tools import tool
from langchain_ollama import OllamaLLM
from indexer import OllamaEmbeddings

client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(
    "codebase",
    embedding_function=OllamaEmbeddings()
)

llm = OllamaLLM(model="llama3.2:3b", temperature=0.3)

@tool
def search_code(query: str) -> str:
    """Search the codebase for relevant code chunks."""
    results = collection.query(query_texts=[query], n_results=3)
    output = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        output += f"\n--- {meta['file']}:L{meta['start_line']} ---\n{doc}\n"
    return output

def run_agent(question: str) -> str:
    context = ""

    for i in range(5):
        if i == 0:
            # force a short search query only
            search_prompt = f"Reply with only 2-4 words to search for in a codebase to answer: '{question}'. No explanation, just the search terms."
            query = llm.invoke(search_prompt).strip().strip('"')
        else:
            # ask if we have enough or need to search more
            decide_prompt = f"""Question: {question}

Retrieved code so far:
{context}

Do you have enough code to answer with specific file and line citations?
Reply with SEARCH: followed by 2-4 word query if you need more.
Reply with DONE if you have enough."""
            
            response = llm.invoke(decide_prompt).strip()
            
            if "DONE" in response:
                break
            elif "SEARCH:" in response:
                query = response.split("SEARCH:")[1].strip()
            else:
                break

        print(f"Iteration {i+1} — searching: {query}")
        results = search_code.invoke(query)
        context += f"\n=== Search {i+1}: '{query}' ===\n{results}"

    # force answer using only retrieved code
    final_prompt = f"""You only know what is in the code chunks below. You have no other knowledge.
Answer ONLY using these chunks. Cite exact file and line numbers like (file.py:L34) for every sentence.
If something is not in the chunks say "not found in retrieved code."

RETRIEVED CODE:
{context}

QUESTION: {question}
ANSWER:"""

    return llm.invoke(final_prompt)

if __name__ == "__main__":
    print(run_agent("how does routing work in flask?"))