import chromadb
from langchain_ollama import OllamaLLM
from indexer import OllamaEmbeddings
import re
from graph_engine import build_graph, blast_radius

GRAPH_KEYWORDS = ["what breaks", "what calls", "what depends on", "what uses", "callers of", "who calls"]

def is_structural_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in GRAPH_KEYWORDS)

def extract_function_name(question: str) -> str:
    q = question.lower()
    for kw in GRAPH_KEYWORDS:
        q = q.replace(kw, "")
    
    tokens = re.findall(r'\b[a-z_][a-z0-9_]*\b', q)
    stopwords = {"the", "a", "an", "is", "are", "to", "of", "in", "on", "if", 
                  "i", "this", "function", "method", "does", "do"}
    candidates = [t for t in tokens if t not in stopwords]
    
    if not candidates:
        return None
    
    # prefer snake_case identifiers (likely real function names)
    underscored = [c for c in candidates if "_" in c]
    if underscored:
        return underscored[0]
    
    return candidates[0]

llm = OllamaLLM(model="llama3.2:3b", temperature=0.3)

def get_collection():
    client = chromadb.PersistentClient(path="db")
    return client.get_or_create_collection(
        "codebase",
        embedding_function=OllamaEmbeddings()
    )

def search(query: str) -> str:
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    output = ""
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        output += f"\n--- {meta['file']}:L{meta['start_line']} ---\n{doc}\n"
    return output


def format_blast_radius(result: dict) -> str:
    output = f"Blast radius for `{result['function']}`\n\n"
    
    output += "Defined in:\n"
    for loc in result["found_in"]:
        output += f"  - {loc['file']}:{loc['line']}\n"
    
    output += f"\nSource code callers ({result['source_count']}):\n"
    if result["source_callers"]:
        for c in result["source_callers"]:
            output += f"  - {c['name']} in {c['file']}:{c['line']}\n"
    else:
        output += "  (none found)\n"
    
    output += f"\nTest coverage ({result['test_count']} tests reference this function)\n"
    
    if result["source_count"] > 0:
        output += f"\nChanging this function's signature could affect {result['source_count']} caller(s) listed above."
    else:
        output += "\nNo direct source callers found — likely safe to modify, but verify test coverage."
    
    return output

def run_agent(question: str) -> str:
    if is_structural_query(question):
        func_name = extract_function_name(question)
        if func_name:
            G = build_graph("sample/flask")
            result = blast_radius(G, func_name)
            if "error" not in result:
                return format_blast_radius(result)
    context = ""

    for i in range(5):
        if i == 0:
            search_prompt = f"Reply with only 2-4 words to search for in a codebase to answer: '{question}'. No explanation, just the search terms."
            query = llm.invoke(search_prompt).strip().strip('"')
        else:
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
        results = search(query)
        context += f"\n=== Search {i+1}: '{query}' ===\n{results}"

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