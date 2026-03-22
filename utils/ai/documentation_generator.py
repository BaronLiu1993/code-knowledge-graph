import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are an expert documentation agent. Your job is to generate CLAUDE.md files "
    "that give AI assistants maximum context about a codebase. Write deeply descriptive "
    "documentation that covers what every file does, how they connect, what data flows "
    "through them, and why each dependency exists."
)


def _create_llm():
    return ChatOpenAI(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"), max_tokens=4000)


def generate_shared_context(graph, file_contents):
    llm = _create_llm()
    repo_summary = _build_repo_summary(graph)
    top_nodes = _get_top_nodes_by_pagerank(graph, limit=15)
    cross_deps = _get_cross_folder_dependencies(graph)
    top_code = _build_top_files_code_context(graph, file_contents, limit=5)

    prompt = (
        "Deeply analyze this repository and produce a comprehensive context summary. "
        "This summary will be shared with multiple sub-agents that each generate "
        "documentation for one folder. Capture everything they need:\n\n"
        "1. What this project does and its overall purpose\n"
        "2. The architecture pattern and how modules connect\n"
        "3. The role of each folder\n"
        "4. Which files are most critical and why (PageRank = importance)\n"
        "5. How data flows through the system end-to-end\n"
        "6. Key shared dependencies and what they provide\n"
        "7. External services and env vars needed\n"
        "8. Naming conventions and patterns used across the codebase\n\n"
        "Be thorough — sub-agents cannot see each other's work, so this is their "
        "only source of cross-folder awareness.\n\n"
        f"## Repository Structure\n{repo_summary}\n\n"
        f"## Top Files by PageRank\n{top_nodes}\n\n"
        f"## Cross-Folder Dependencies\n{cross_deps}\n\n"
        f"## Source Code of Most Critical Files\n{top_code}"
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


def generate_root_claude_md(graph, file_contents, shared_context):
    llm = _create_llm()
    top_nodes = _get_top_nodes_by_pagerank(graph, limit=15)
    cross_folder_deps = _get_cross_folder_dependencies(graph)
    top_file_code = _build_top_files_code_context(graph, file_contents, limit=8)
    full_deps = _build_full_dependency_chain(graph, file_contents)

    prompt = (
        "Generate the root CLAUDE.md for the entire repository. This is the single "
        "most important file — an AI reading only this should understand everything.\n\n"
        "Include these sections:\n"
        "## What This Repository Does\n"
        "Detailed description of the product, who uses it, what problem it solves.\n\n"
        "## Architecture Overview\n"
        "Pattern used, how requests flow, middleware, module communication.\n\n"
        "## Folder Structure\n"
        "Each folder: what it does, most important file, dependencies.\n\n"
        "## Critical Files (by PageRank)\n"
        "Most depended-upon files, what each does, why it's central.\n\n"
        "## Module Dependency Map\n"
        "For each cross-folder dependency: what, why, how, what breaks without it.\n\n"
        "## Request Flow\n"
        "Trace a typical request from entry to response, naming every file.\n\n"
        "## Environment & Configuration\n"
        "Env vars, external services, configs.\n\n"
        "---\n"
        f"## Shared Repository Context\n{shared_context}\n\n"
        f"## Top Files by PageRank\n{top_nodes}\n\n"
        f"## Cross-Folder Dependencies\n{cross_folder_deps}\n\n"
        f"## Full Dependency Chain\n{full_deps}\n\n"
        f"## Critical Source Code\n{top_file_code}"
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


def generate_folder_claude_md(folder_path, folder_nodes, graph, file_contents, shared_context):
    llm = _create_llm()
    relationship_map = _build_deep_relationship_map(folder_nodes, graph, file_contents)
    source_code = _build_source_code_context(folder_nodes, file_contents)
    inbound = _build_inbound_dependencies(folder_path, graph)

    prompt = (
        f"Generate CLAUDE.md for the folder: {folder_path}/\n\n"
        "Include these sections:\n"
        "## Purpose\n"
        "What this folder handles. What breaks if deleted. How it fits in the architecture.\n\n"
        "## Files\n"
        "For EACH file: what it does (from the code), exports, PageRank, why it matters.\n\n"
        "## Dependency Map\n"
        "For EACH import: WHAT is imported, WHY it's needed, HOW it's used. "
        "Format: `source` → `target`: detailed explanation.\n\n"
        "## Inbound Dependencies\n"
        "Which files outside this folder depend on files here, and what they use.\n\n"
        "## Data Flow\n"
        "How data enters, transforms, and exits this folder.\n\n"
        "## Key Patterns\n"
        "Conventions, patterns, or anti-patterns.\n\n"
        "---\n"
        f"## Shared Repository Context\n{shared_context}\n\n"
        f"## Relationship Map\n{relationship_map}\n\n"
        f"## Inbound Dependencies\n{inbound}\n\n"
        f"## Source Code\n{source_code}"
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content


def _build_inbound_dependencies(folder_path, graph):
    inbound = []
    for node in graph["nodes"]:
        if node["type"] != "file":
            continue
        node_folder = node["name"].rsplit("/", 1)[0] if "/" in node["name"] else "root"
        if node_folder == folder_path:
            continue
        for rel in node["relationships"]:
            if rel["predicate"] == "imports" and folder_path in rel["target"]:
                inbound.append(f"{node['name']} imports {rel['target']}")

    if not inbound:
        return "No external files depend on this folder."
    return "\n".join(f"- {dep}" for dep in inbound)


def _build_deep_relationship_map(folder_nodes, graph, file_contents):
    all_nodes_by_name = {n["name"]: n for n in graph["nodes"]}
    lines = []
    char_budget = 12000

    for node in sorted(folder_nodes, key=lambda n: n["pagerank"], reverse=True):
        imports = [r for r in node["relationships"] if r["predicate"] == "imports"]
        if not imports:
            continue

        lines.append(f"### {node['name']}")
        source_code = file_contents.get(node["name"], "")
        if source_code:
            lines.append(f"Source file preview:\n```\n{source_code[:1000]}\n```\n")
            char_budget -= min(len(source_code), 1000)

        for rel in imports:
            target_path = rel["target"]
            target_code = file_contents.get(target_path, "")
            target_node = all_nodes_by_name.get(target_path, {})
            target_defines = [
                r["target"].split("::")[-1]
                for r in target_node.get("relationships", [])
                if r["predicate"] == "defines"
            ]

            lines.append(f"**→ imports `{target_path}`**")
            if target_defines:
                lines.append(f"  Target exports: {', '.join(target_defines)}")
            lines.append(f"  Target pagerank: {rel.get('target_pagerank', 0)}")
            if target_code and char_budget > 0:
                preview = target_code[:800]
                lines.append(f"  Target code preview:\n```\n{preview}\n```")
                char_budget -= len(preview)
            lines.append("")

        if char_budget <= 0:
            break

    return "\n".join(lines) if lines else "No import relationships in this folder."


def _build_full_dependency_chain(graph, file_contents):
    lines = []
    char_budget = 10000
    file_nodes = [n for n in graph["nodes"] if n["type"] == "file"]
    all_nodes_by_name = {n["name"]: n for n in graph["nodes"]}

    for node in file_nodes[:10]:
        imports = [r for r in node["relationships"] if r["predicate"] == "imports"]
        if not imports:
            continue

        lines.append(f"### {node['name']} (pagerank: {node['pagerank']})")
        for rel in imports:
            target_path = rel["target"]
            target_node = all_nodes_by_name.get(target_path, {})
            target_defines = [
                r["target"].split("::")[-1]
                for r in target_node.get("relationships", [])
                if r["predicate"] == "defines"
            ]
            target_code = file_contents.get(target_path, "")

            lines.append(f"  → `{target_path}` (pagerank: {rel.get('target_pagerank', 0)})")
            if target_defines:
                lines.append(f"    Exports: {', '.join(target_defines)}")
            if target_code and char_budget > 0:
                preview = target_code[:600]
                lines.append(f"    Code:\n```\n{preview}\n```")
                char_budget -= len(preview)
        lines.append("")

        if char_budget <= 0:
            break

    return "\n".join(lines) if lines else "No dependency chains found."


def _build_source_code_context(folder_nodes, file_contents):
    lines = []
    char_budget = 10000
    for node in sorted(folder_nodes, key=lambda n: n["pagerank"], reverse=True):
        path = node["name"]
        content = file_contents.get(path)
        if not content:
            continue
        truncated = content[:2500]
        lines.append(f"### {path} (pagerank: {node['pagerank']})\n```\n{truncated}\n```\n")
        char_budget -= len(truncated)
        if char_budget <= 0:
            break
    return "\n".join(lines) if lines else "No source code available."


def _build_top_files_code_context(graph, file_contents, limit=8):
    lines = []
    file_nodes = [n for n in graph["nodes"] if n["type"] == "file"]
    for node in file_nodes[:limit]:
        content = file_contents.get(node["name"])
        if not content:
            continue
        truncated = content[:2000]
        defines = [
            r["target"].split("::")[-1]
            for r in node["relationships"]
            if r["predicate"] == "defines"
        ]
        imports = [r["target"] for r in node["relationships"] if r["predicate"] == "imports"]

        lines.append(f"### {node['name']} (pagerank: {node['pagerank']})")
        if defines:
            lines.append(f"Exports: {', '.join(defines)}")
        if imports:
            lines.append(f"Imports: {', '.join(imports)}")
        lines.append(f"```\n{truncated}\n```\n")
    return "\n".join(lines) if lines else "No source code available."


def _build_repo_summary(graph):
    folders = set()
    file_count = 0
    for node in graph["nodes"]:
        if node["type"] == "folder":
            folders.add(node["name"])
        elif node["type"] == "file":
            file_count += 1

    folder_list = "\n".join(f"- {f}/" for f in sorted(folders))
    return f"Files: {file_count}\n\nFolders:\n{folder_list}"


def _get_top_nodes_by_pagerank(graph, limit=15):
    file_nodes = [n for n in graph["nodes"] if n["type"] == "file"]
    top = file_nodes[:limit]
    lines = []
    for n in top:
        import_count = sum(1 for r in n["relationships"] if r["predicate"] == "imports")
        define_count = sum(1 for r in n["relationships"] if r["predicate"] == "defines")
        lines.append(
            f"- {n['name']} (pagerank: {n['pagerank']}, "
            f"exports {define_count} symbols, imports {import_count} modules)"
        )
    return "\n".join(lines)


def _get_cross_folder_dependencies(graph):
    deps = {}
    for node in graph["nodes"]:
        if node["type"] != "file":
            continue
        source_folder = node["name"].rsplit("/", 1)[0] if "/" in node["name"] else "root"
        for rel in node["relationships"]:
            if rel["predicate"] != "imports":
                continue
            target = rel["target"]
            target_folder = target.rsplit("/", 1)[0] if "/" in target else "root"
            if source_folder != target_folder and target_folder != "root":
                key = f"{source_folder} → {target_folder}"
                deps.setdefault(key, []).append(f"{node['name']} imports {target}")

    lines = []
    for key, examples in sorted(deps.items(), key=lambda x: -len(x[1])):
        lines.append(f"### {key} ({len(examples)} imports)")
        for ex in examples[:5]:
            lines.append(f"  - {ex}")
        lines.append("")
    return "\n".join(lines) if lines else "No cross-folder dependencies detected."
