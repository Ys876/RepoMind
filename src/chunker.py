from tree_sitter import Language, Parser # type: ignore
import tree_sitter_python as tspython # type: ignore
from dataclasses import dataclass #struct in python for chunks
@dataclass
class Chunk:
    name: str
    func_code: str
    start_line: str
    end_line: str
    file: str 

    
PY_LANGUAGE = Language(tspython.language())

parser = Parser(PY_LANGUAGE)


def chunk_file(filepath):
    chunks=[]
    with open(filepath, "rb" ) as f:
        code = f.read()
    tree = parser.parse(code)

    #recursive function, finds all functions, creates chunk objs, puts in list
    def find_func(node):
    
        if node.type == "function_definition":
            name = node.children[1].text.decode("utf-8")#extract only func name, print w/varwidth encoding
            func_code = code[node.start_byte:node.end_byte].decode("utf-8")#extract function text
            chunks.append(Chunk(name, func_code, node.start_point[0], node.end_point[0], filepath))
    
        for children in node.children:
            find_func(children)

    find_func(tree.root_node) #call recursive on root
    return chunks
print(chunk_file("sample/sample.py"))
