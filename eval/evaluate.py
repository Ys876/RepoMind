import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from chunker import chunk_repo
from bm25_retriever import BM25Retriever
from fusion import reciprocal_rank_fusion
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

def setup_bm25_retriever():
    chunks = chunk_repo(REPO_PATH)
    return BM25Retriever(chunks)

def query_bm25(retriever, question, n=20):
    results = retriever.search(question, n=n)
    return [chunk.file for chunk in results]

def query_collection(collection, question, n=5):
    results = collection.query(query_texts=[question], n_results=n)
    return [meta["file"] for meta in results["metadatas"][0]]
def query_hybrid(ast_col, bm25_retriever, question, n=5):
    semantic_files = query_collection(ast_col, question, n=20)
    bm25_raw = query_bm25(bm25_retriever, question, n=20)
    # filter test files from BM25 — they pollute fusion
    bm25_files = [f for f in bm25_raw if "/tests/" not in f and "test_" not in f]
    fused = reciprocal_rank_fusion([semantic_files, bm25_files], weights=[2.0, 1.0])
    return fused[:n]

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
    bm25_retriever = setup_bm25_retriever()

    ast_rr, naive_rr, bm25_rr, hybrid_rr = [], [], [], []
    ast_p3, naive_p3, bm25_p3, hybrid_p3 = [], [], [], []
    ast_hits, naive_hits, bm25_hits, hybrid_hits = 0, 0, 0, 0
    total = len(questions)

    print(f"\nRunning eval on {total} questions...\n")
    print(f"{'Q':<4} {'AST':<8} {'Naive':<8} {'BM25':<8} {'Hybrid':<8} Question")
    print("-" * 80)

    for i, q in enumerate(questions):
        question = q["question"]

        ast_files    = query_collection(ast_col, question, n=5)
        naive_files  = query_collection(naive_col, question, n=5)
        bm25_files   = query_bm25(bm25_retriever, question, n=5)
        hybrid_files = query_hybrid(ast_col, bm25_retriever, question, n=5)

        a_rr  = reciprocal_rank(ast_files, q)
        n_rr  = reciprocal_rank(naive_files, q)
        b_rr  = reciprocal_rank(bm25_files, q)
        h_rr  = reciprocal_rank(hybrid_files, q)

        a_p3  = precision_at_3(ast_files, q)
        n_p3  = precision_at_3(naive_files, q)
        b_p3  = precision_at_3(bm25_files, q)
        h_p3  = precision_at_3(hybrid_files, q)

        ast_rr.append(a_rr);    naive_rr.append(n_rr)
        bm25_rr.append(b_rr);   hybrid_rr.append(h_rr)
        ast_p3.append(a_p3);    naive_p3.append(n_p3)
        bm25_p3.append(b_p3);   hybrid_p3.append(h_p3)

        if a_rr > 0: ast_hits += 1
        if n_rr > 0: naive_hits += 1
        if b_rr > 0: bm25_hits += 1
        if h_rr > 0: hybrid_hits += 1

        print(f"Q{i+1:<3} {a_rr:<8.2f} {n_rr:<8.2f} {b_rr:<8.2f} {h_rr:<8.2f} {question[:40]}")

    ast_mrr    = sum(ast_rr)    / total
    naive_mrr  = sum(naive_rr)  / total
    bm25_mrr   = sum(bm25_rr)   / total
    hybrid_mrr = sum(hybrid_rr) / total

    ast_p3_avg    = sum(ast_p3)    / total * 100
    naive_p3_avg  = sum(naive_p3)  / total * 100
    bm25_p3_avg   = sum(bm25_p3)   / total * 100
    hybrid_p3_avg = sum(hybrid_p3) / total * 100

    print(f"\n{'='*65}")
    print("FAILURE ANALYSIS — HYBRID COMPLETE MISSES")
    print(f"{'='*65}")
    for i, q in enumerate(questions):
        if hybrid_rr[i] == 0:
            files = query_hybrid(ast_col, bm25_retriever, q["question"], n=5)
            print(f"\nQ{i+1}: {q['question']}")
            print(f"  Expected: {q['expected_file']}")
            for f in files[:3]:
                print(f"    {f}")

    print(f"\n{'='*65}")
    print(f"{'Metric':<25} {'AST':<12} {'Naive':<12} {'BM25':<12} {'Hybrid'}")
    print(f"{'-'*65}")
    print(f"{'MRR':<25} {ast_mrr:<12.3f} {naive_mrr:<12.3f} {bm25_mrr:<12.3f} {hybrid_mrr:.3f}")
    print(f"{'Precision@3':<25} {ast_p3_avg:<12.1f}% {naive_p3_avg:<12.1f}% {bm25_p3_avg:<12.1f}% {hybrid_p3_avg:.1f}%")
    print(f"{'Hit Rate':<25} {ast_hits/total*100:<12.1f}% {naive_hits/total*100:<12.1f}% {bm25_hits/total*100:<12.1f}% {hybrid_hits/total*100:.1f}%")
    print(f"{'='*65}")

    best = max(ast_mrr, naive_mrr, hybrid_mrr)
    improvement = ((hybrid_mrr - naive_mrr) / naive_mrr * 100) if naive_mrr > 0 else 0
    print(f"\nHybrid vs Naive RAG baseline: {improvement:+.1f}% MRR improvement")
    print(f"\nResume line:")
    print(f"Hybrid BM25+semantic retrieval via RRF achieved MRR of {hybrid_mrr:.2f}")
    print(f"vs naive RAG baseline of {naive_mrr:.2f} (+{improvement:.1f}%) on a {total}-question")
    print(f"benchmark — AST chunking alone: {ast_mrr:.2f}, BM25 alone: {bm25_mrr:.2f}")

if __name__ == "__main__":
    run_eval()