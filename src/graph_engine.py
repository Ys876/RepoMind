from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import networkx as nx
from chunker import chunk_repo

PY_LANGUAGE = Language(tspython.language())


def build_graph(repo_path: str) -> nx.DiGraph:
    chunks = chunk_repo(repo_path)
    G = nx.DiGraph()
    
    # add all functions as nodes first
    for chunk in chunks:
        node_id = f"{chunk.file}::{chunk.name}"
        G.add_node(node_id, name=chunk.name, file=chunk.file, 
                    start_line=chunk.start_line, end_line=chunk.end_line)
    
    # build a lookup: function name -> list of node_ids with that name
    name_to_nodes = {}
    for chunk in chunks:
        node_id = f"{chunk.file}::{chunk.name}"
        name_to_nodes.setdefault(chunk.name, []).append(node_id)
    
    # add edges: caller -> callee
    for chunk in chunks:
        caller_id = f"{chunk.file}::{chunk.name}"
        calls = extract_calls(chunk.func_code)
        for called_name in calls:
            if called_name in name_to_nodes:
                for callee_id in name_to_nodes[called_name]:
                    if callee_id != caller_id:
                        G.add_edge(caller_id, callee_id)
    
    return G



def blast_radius(G: nx.DiGraph, function_name: str, max_hops: int = 1) -> dict:
    targets = [n for n in G.nodes if G.nodes[n]["name"] == function_name]
    
    if not targets:
        return {"error": f"No function named '{function_name}' found"}
    
    callers = set()
    for target in targets:
        callers.update(G.predecessors(target))
        if max_hops == 2:
            for c in list(callers):
                callers.update(G.predecessors(c))
    
    source_callers = []
    test_callers = []
    for c in callers:
        info = {"name": G.nodes[c]["name"], "file": G.nodes[c]["file"], "line": G.nodes[c]["start_line"]}
        if "/tests/" in G.nodes[c]["file"] or "test_" in G.nodes[c]["file"]:
            test_callers.append(info)
        else:
            source_callers.append(info)
    
    return {
        "function": function_name,
        "found_in": [{"file": G.nodes[t]["file"], "line": G.nodes[t]["start_line"]} for t in targets],
        "source_callers": source_callers,
        "test_callers": test_callers,
        "source_count": len(source_callers),
        "test_count": len(test_callers)
    }


def extract_calls(func_code: str) -> list[str]:
    """Find all function names called inside this function's code."""
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(bytes(func_code, "utf-8"))
    calls = []
    
    def walk(node):
        if node.type == "call":
            func_node = node.children[0]
            if func_node.type == "identifier":
                calls.append(func_node.text.decode("utf-8"))
            elif func_node.type == "attribute":
                # e.g. self.processor.handle() -> "handle"
                attr = func_node.children[-1]
                calls.append(attr.text.decode("utf-8"))
        for child in node.children:
            walk(child)
    
    walk(tree.root_node)
    return list(set(calls))