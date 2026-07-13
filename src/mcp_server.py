import asyncio
import os
import sys

# make sibling modules (chunker, bm25_retriever, fusion, graph_engine, indexer) importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import chromadb
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from bm25_retriever import BM25Retriever
from chunker import chunk_repo
from fusion import reciprocal_rank_fusion
from graph_engine import blast_radius, build_graph
from indexer import OllamaEmbeddings

REPO_PATH = "sample/flask"

server = Server("repomind")

_collection = None
_bm25 = None
_graph = None


def initialize():
    global _collection, _bm25, _graph

    client = chromadb.PersistentClient(path="db")
    _collection = client.get_or_create_collection(
        "codebase", embedding_function=OllamaEmbeddings()
    )

    # chunk_repo/build_graph parse every file with tree-sitter, which prints
    # each parse tree to stdout. On a stdio MCP server stdout is reserved for
    # JSON-RPC, so that debug output must not leak there — divert it while
    # these calls run.
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        chunks = chunk_repo(REPO_PATH)
        _bm25 = BM25Retriever(chunks)
        _graph = build_graph(REPO_PATH)
    finally:
        sys.stdout.close()
        sys.stdout = real_stdout

    print("RepoMind MCP server ready", file=sys.stderr)


def _chunk_id(file: str, name: str, start_line) -> str:
    return f"{file}::{name}::{start_line}"


def _semantic_search(query: str, n: int = 20):
    results = _collection.query(query_texts=[query], n_results=n)
    ids = []
    info = {}
    for meta in results["metadatas"][0]:
        cid = _chunk_id(meta["file"], meta["name"], meta["start_line"])
        ids.append(cid)
        info[cid] = meta
    return ids, info


def _bm25_search(query: str, n: int = 20):
    # BM25Retriever already excludes "/tests/" and "test_" files from its index
    chunks = _bm25.search(query, n=n)
    ids = []
    info = {}
    for chunk in chunks:
        cid = _chunk_id(chunk.file, chunk.name, chunk.start_line)
        ids.append(cid)
        info[cid] = {"file": chunk.file, "name": chunk.name, "start_line": chunk.start_line}
    return ids, info


def search_codebase(query: str) -> str:
    semantic_ids, semantic_info = _semantic_search(query, n=20)
    bm25_ids, bm25_info = _bm25_search(query, n=20)

    fused = reciprocal_rank_fusion([semantic_ids, bm25_ids], weights=[2.0, 1.0])
    top = fused[:5]

    combined_info = {**semantic_info, **bm25_info}

    lines = [f"Top results for: {query}"]
    if not top:
        lines.append("(no results found)")
    else:
        for i, cid in enumerate(top, start=1):
            meta = combined_info[cid]
            lines.append(f"{i}. {meta['file']}:{meta['start_line']} — {meta['name']}")

    return "\n".join(lines)


def blast_radius_check(function_name: str) -> str:
    result = blast_radius(_graph, function_name)

    if "error" in result:
        return result["error"]

    lines = [f"Blast radius for `{function_name}`"]

    defined_in = ", ".join(f"{loc['file']}:{loc['line']}" for loc in result["found_in"])
    lines.append(f"Defined in: {defined_in}")

    if result["source_callers"]:
        callers_str = ", ".join(
            f"{c['name']} in {c['file']}:{c['line']}" for c in result["source_callers"]
        )
    else:
        callers_str = "(none found)"
    lines.append(f"Source callers ({result['source_count']}): {callers_str}")

    lines.append(f"Test coverage: {result['test_count']} tests reference this function")

    if result["source_count"] > 0:
        lines.append(
            f"Impact: changing this function could affect {result['source_count']} caller(s)"
        )
    else:
        lines.append("Impact: no direct source callers found — likely safe to modify")

    return "\n".join(lines)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_codebase",
            description=(
                "Hybrid semantic + keyword search over the indexed codebase. Use this "
                "to find where a piece of functionality, class, or concept is "
                "implemented. Returns up to 5 file:line results ranked by combined "
                "relevance."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Natural language or keyword description of what to find, "
                            "e.g. 'how does routing work'"
                        ),
                    }
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="blast_radius_check",
            description=(
                "Given an exact function name, find every place in the source code "
                "and tests that calls it (via the codebase's call graph). Use this "
                "before changing, renaming, or removing a function to see what could "
                "break."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "function_name": {
                        "type": "string",
                        "description": "Exact function name to check, e.g. 'add_url_rule'",
                    }
                },
                "required": ["function_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_codebase":
        text = search_codebase(arguments["query"])
    elif name == "blast_radius_check":
        text = blast_radius_check(arguments["function_name"])
    else:
        raise ValueError(f"Unknown tool: {name}")

    return [TextContent(type="text", text=text)]


async def main():
    initialize()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
