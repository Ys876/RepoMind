from tree_sitter import Language, Parser  # type: ignore
import tree_sitter_python as tspython  # type: ignore
import tree_sitter_javascript as tsjavascript #type: ignore
from dataclasses import dataclass #struct in python for chunks
import os
@dataclass
class Chunk:
    name: str
    func_code: str
    start_line: str
    end_line: str
    file: str 
LANGUAGES = {
    ".py": Language(tspython.language()),
    ".js": Language(tsjavascript.language()),
}  



def chunk_file(filepath):
    chunks=[]
    with open(filepath, "rb" ) as f:
        code = f.read()
    extension = filepath.split(".")[-1]
    extension = "." + extension
    language = LANGUAGES.get(extension, None)
    if language is None:
        return[]
    parser = Parser(language)
    tree = parser.parse(code)
    print(tree.root_node)

    #recursive function, finds all functions, creates chunk objs, puts in list
    def find_func(node):
        
        if node.type in ("function_definition", "function_declaration"):
            name = node.children[1].text.decode("utf-8")#extract only func name, print w/varwidth encoding
            func_code = code[node.start_byte:node.end_byte].decode("utf-8")#extract function text
            if len(func_code) > 2000:
                func_code = func_code[:2000]
            chunks.append(Chunk(name, func_code, node.start_point[0], node.end_point[0], filepath))
    
        for children in node.children:
            find_func(children)

    find_func(tree.root_node) #call recursive on root
    return chunks
#print(chunk_file("sample/sample.js"))

def chunk_repo(repo_path):
    results = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            fullpath = os.path.join(root, file)
            if file.endswith(".py") or file.endswith(".js"):
                res = chunk_file(fullpath)
                results.extend(res)
    return results
