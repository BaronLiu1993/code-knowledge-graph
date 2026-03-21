import os
import json
from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
from dotenv import load_dotenv
from openai import OpenAI
from utils.graph.builder import add_symbol_node, add_import_edge, add_semantic_edge

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

_LANGUAGES = {
    ".py":  Language(tspython.language()),
    ".js":  Language(tsjavascript.language()),
    ".jsx": Language(tsjavascript.language()),
    ".ts":  Language(tsjavascript.language()),
    ".tsx": Language(tsjavascript.language()),
}

_SYMBOL_NODE_TYPES = {
    ".py": {"function_definition", "class_definition"},
    ".js": {"function_declaration", "class_declaration"},
}
_SYMBOL_NODE_TYPES[".jsx"] = _SYMBOL_NODE_TYPES[".js"]
_SYMBOL_NODE_TYPES[".ts"]  = _SYMBOL_NODE_TYPES[".js"]
_SYMBOL_NODE_TYPES[".tsx"] = _SYMBOL_NODE_TYPES[".js"]

_IMPORT_NODE_TYPES = {
    ".py": {"import_statement", "import_from_statement"},
    ".js": {"import_statement"},
}
_IMPORT_NODE_TYPES[".jsx"] = _IMPORT_NODE_TYPES[".js"]
_IMPORT_NODE_TYPES[".ts"]  = _IMPORT_NODE_TYPES[".js"]
_IMPORT_NODE_TYPES[".tsx"] = _IMPORT_NODE_TYPES[".js"]


def _get_extension(file_path):
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def _walk(node):
    yield node
    for child in node.children:
        yield from _walk(child)


def _extract_name(node):
    for child in node.children:
        if child.type == "identifier":
            return child.text.decode("utf-8")
    return None


def _extract_import_module(node, ext):
    if ext == ".py":
        for child in node.children:
            if child.type in {"dotted_name", "relative_module"}:
                return child.text.decode("utf-8")
    elif ext in {".js", ".jsx", ".ts", ".tsx"}:
        for child in _walk(node):
            if child.type == "string_fragment":
                return child.text.decode("utf-8")
    return None


def _is_require_call(node):
    if node.type != "call_expression":
        return False
    fn = node.children[0] if node.children else None
    return fn and fn.type == "identifier" and fn.text == b"require"


def extract_symbols_from_file(file_path, content, nodes, adj_list):
    ext = _get_extension(file_path)
    language = _LANGUAGES.get(ext)
    if not language:
        return

    parser = Parser(language)
    tree = parser.parse(bytes(content, "utf-8"))

    symbol_types = _SYMBOL_NODE_TYPES.get(ext, set())
    import_types = _IMPORT_NODE_TYPES.get(ext, set())

    for node in _walk(tree.root_node):
        if node.type in symbol_types:
            name = _extract_name(node)
            if name:
                symbol_type = "class" if "class" in node.type else "function"
                add_symbol_node(file_path, name, symbol_type, nodes, adj_list)

        elif node.type in import_types or _is_require_call(node):
            module = _extract_import_module(node, ext)
            if module:
                add_import_edge(file_path, module, adj_list)


ADD_RELATIONSHIP_TOOL = {
    "type": "function",
    "function": {
        "name": "add_relationship",
        "description": "Record a semantic relationship between two files or symbols. The predicate describes what the source does for the target.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_path": {"type": "string"},
                "to_path": {"type": "string"},
                "predicate": {
                    "type": "string",
                    "description": "Natural language: what source does for target, e.g. 'delegates auth requests to'",
                },
            },
            "required": ["from_path", "to_path", "predicate"],
        },
    },
}


def _call_openai(prompt, tools):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto",
    )
    calls = []
    for tool_call in response.choices[0].message.tool_calls or []:
        calls.append((tool_call.function.name, json.loads(tool_call.function.arguments)))
    return calls


def analyze_module_relationships(module_name, file_paths, adj_list, nodes):
    path_list = "\n".join(f"- {p}" for p in file_paths)
    prompt = (
        f"Analyze the '{module_name}' module. Use add_relationship to record how "
        f"these files relate to each other. The predicate must describe what the "
        f"source does for the target (e.g. 'delegates auth logic to'). "
        f"Only record relationships you are confident about.\n\n"
        f"Files:\n{path_list}"
    )

    for name, args in _call_openai(prompt, [ADD_RELATIONSHIP_TOOL]):
        if name == "add_relationship" and args["from_path"] in nodes and args["to_path"] in nodes:
            add_semantic_edge(args["from_path"], args["to_path"], args["predicate"], adj_list)


def analyze_cross_module_relationships(module_names, adj_list, nodes):
    module_list = "\n".join(f"- {m}" for m in module_names)
    prompt = (
        f"Analyze the top-level architecture. Use add_relationship to identify "
        f"how these modules depend on each other. The predicate must describe "
        f"what the source module does for the target.\n\n"
        f"Modules:\n{module_list}"
    )

    for name, args in _call_openai(prompt, [ADD_RELATIONSHIP_TOOL]):
        if name == "add_relationship":
            add_semantic_edge(args["from_path"], args["to_path"], args["predicate"], adj_list)
