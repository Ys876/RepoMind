import re

def tokenize(text: str) -> list[str]:
    tokens = re.findall(r'[a-zA-Z0-9_]+', text)
    expanded = []
    
    for token in tokens:
        expanded.append(token)
        sub = re.findall(r'[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z]|\b)', token)
        if len(sub) > 1:
            expanded.extend([s.lower() for s in sub])
    
    return list(set(expanded))