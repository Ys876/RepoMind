import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from indexer import OllamaEmbeddings

REPO_PATH = "sample/flask"
QUESTIONS_PATH = "eval/questions.json"

def load_questions():
    with open(QUESTIONS_PATH) as f:
        return json.load(f)

def get_ast_collection():
    client = chromadb.PersistentClient(path="db")
    return client.get_or_create_collection(
        "codebase",
        embedding_function=OllamaEmbeddings()
    )

def setup_naive_collection():
    client = chromadb.PersistentClient(path="db_naive")
    collection = client.get_or_create_collection(
        "naive",
        embedding_function=OllamaEmbeddings()
    )
    if collection.count() == 0:
        print("Building naive index...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs, metas, ids = [], [], []
        i = 0
        for root, dirs, files in os.walk(REPO_PATH):
            for file in files:
                if not file.endswith((".py", ".js")):
                    continue
                path = os.path.join(root, file)
                try:
                    with open(path, "r", errors="ignore") as f:
                        content = f.read()
                    for chunk in splitter.split_text(content):
                        docs.append(chunk)
                        metas.append({"file": path})
                        ids.append(f"naive_{i}")
                        i += 1
                except:
                    continue
        collection.add(documents=docs, metadatas=metas, ids=ids)
        print(f"Naive index built: {len(docs)} chunks")
    return collection

def query_collection(collection, question, n=5):
    results = collection.query(query_texts=[question], n_results=n)
    return [meta["file"] for meta in results["metadatas"][0]]

def is_match(cited_file, question):
    expected = question["expected_file"]
    also_valid = question.get("also_valid", [])
    all_valid = [expected] + also_valid
    return any(valid in cited_file for valid in all_valid)

def reciprocal_rank(cited_files, question):
    for i, f in enumerate(cited_files):
        if is_match(f, question):
            return 1.0 / (i + 1)
    return 0.0

def precision_at_3(cited_files, question):
    top3 = cited_files[:3]
    hits = sum(1 for f in top3 if is_match(f, question))
    return hits / 3

def run_eval():
    questions = load_questions()
    print("Loading collections...")
    ast_col = get_ast_collection()
    naive_col = setup_naive_collection()

    ast_rr, naive_rr = [], []
    ast_p3, naive_p3 = [], []
    ast_hits, naive_hits = 0, 0
    total = len(questions)

    print(f"\nRunning eval on {total} questions...\n")
    print(f"{'Q':<4} {'AST RR':<10} {'Naive RR':<10} {'AST P@3':<10} {'Naive P@3':<12} Question")
    print("-" * 80)

    for i, q in enumerate(questions):
        question = q["question"]

        ast_files = query_collection(ast_col, question, n=5)
        naive_files = query_collection(naive_col, question, n=5)

        a_rr = reciprocal_rank(ast_files, q)
        n_rr = reciprocal_rank(naive_files, q)
        a_p3 = precision_at_3(ast_files, q)
        n_p3 = precision_at_3(naive_files, q)

        ast_rr.append(a_rr)
        naive_rr.append(n_rr)
        ast_p3.append(a_p3)
        naive_p3.append(n_p3)

        if a_rr > 0:
            ast_hits += 1
        if n_rr > 0:
            naive_hits += 1

        print(f"Q{i+1:<3} {a_rr:<10.2f} {n_rr:<10.2f} {a_p3:<10.2f} {n_p3:<12.2f} {question[:45]}")

    ast_mrr = sum(ast_rr) / total
    naive_mrr = sum(naive_rr) / total
    ast_avg_p3 = sum(ast_p3) / total * 100
    naive_avg_p3 = sum(naive_p3) / total * 100
    mrr_improvement = ((ast_mrr - naive_mrr) / naive_mrr * 100) if naive_mrr > 0 else 0
    p3_improvement = ((ast_avg_p3 - naive_avg_p3) / naive_avg_p3 * 100) if naive_avg_p3 > 0 else 0

    print(f"\n{'='*60}")
    print("FAILURE ANALYSIS")
    print(f"{'='*60}")
    print("\nAST COMPLETE MISSES (RR = 0):")
    for i, q in enumerate(questions):
        if ast_rr[i] == 0:
            print(f"\nQ{i+1}: {q['question']}")
            print(f"  Expected: {q['expected_file']}")
            files = query_collection(ast_col, q['question'], n=5)
            for f in files[:3]:
                print(f"    {f}")

    print(f"\n{'='*60}")
    print(f"{'Metric':<25} {'AST':<15} {'Naive':<15} {'Improvement'}")
    print(f"{'-'*60}")
    print(f"{'MRR':<25} {ast_mrr:<15.3f} {naive_mrr:<15.3f} {mrr_improvement:+.1f}%")
    print(f"{'Precision@3':<25} {ast_avg_p3:<15.1f}% {naive_avg_p3:<15.1f}% {p3_improvement:+.1f}%")
    print(f"{'Hit Rate':<25} {ast_hits/total*100:<15.1f}% {naive_hits/total*100:<15.1f}%")
    print(f"{'='*60}")
    print(f"\nResume line:")
    print(f"AST-aware chunking achieved MRR of {ast_mrr:.2f} vs {naive_mrr:.2f} baseline")
    print(f"and Precision@3 of {ast_avg_p3:.1f}% vs {naive_avg_p3:.1f}% on a {total}-question")
    print(f"benchmark across open-source repositories.")
if __name__ == "__main__":
    run_eval()
