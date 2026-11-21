"""Isolated access to the production E2 normalisation and course-eligibility logic (whole-pipeline evaluation, Step 2).

WHY NOT IMPORT THE MODULES
  src/parser.py runs load_dotenv() at import and imports PyPDF2, langchain and langchain_groq; frontend/display.py
  imports streamlit and requests. Importing either would read .env and load heavy, unrelated dependencies.
  This module therefore never imports project code. It READS the two source files as text, parses them with `ast`,
  keeps only the top-level definitions the requested functions actually need (found as a transitive closure of the
  names they use) and executes just those, unmodified, in a fresh namespace that holds only a few standard-library
  modules. Nothing else in the source file is executed: not load_dotenv(), not the imports, not the classes.

SAFETY RULES (a violation raises PureExtractionError instead of running anything)
  * a selected top-level statement must be a function definition or a single-name assignment;
  * an assignment may call only functions of the allowed standard-library modules (re.compile, logging.getLogger)
    or a few builtin constructors (dict, tuple, ...), so no file, network or environment access can sit in a constant;
  * every name a selected definition uses must be defined by the selection, be a builtin, or be an allowed module;
    any other name (for example `st`, `os`, `ChatGroq`) is refused.

Standard library only. Reads exactly two files: src/parser.py and frontend/display.py. Writes nothing.
"""
import ast
import builtins
import hashlib
import html
import json
import logging
import os
import re
import types
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARSER_RELATIVE = ("src", "parser.py")
DISPLAY_RELATIVE = ("frontend", "display.py")

# The only modules a selected definition may use, and the only names in the execution namespace besides builtins.
ALLOWED_MODULES = {"html": html, "json": json, "logging": logging, "re": re, "unicodedata": unicodedata}
_SAFE_BUILTIN_CALLS = frozenset({"dict", "tuple", "set", "frozenset", "list"})

PARSER_ENTRIES = (
    "build_prioritized_missing_skills", "PRIORITY_TIERS", "SKILL_STATUSES", "SKILL_CATEGORIES", "_skill_key",
)
DISPLAY_ENTRIES = (
    "get_recommendation_options", "summarize_priority_gaps", "summarize_skill_gaps", "build_skill_gap_records",
    "get_prioritized_view", "unique_prioritized_entries", "PRIORITY_TIER_LABELS",
    "STATE_NO_GAP", "STATE_PARTIAL", "STATE_NOT_DEMONSTRATED", "STATE_UNVERIFIED",
)
PARSER_MODULE_NAME = "pipeline_stages.parser_slice"
DISPLAY_MODULE_NAME = "pipeline_stages.display_slice"


class PureExtractionError(Exception):
    """The requested definitions cannot be extracted and executed safely."""


# ---------------------------------------------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------------------------------------------

def _top_level_definitions(tree):
    """{name: node} for top-level functions and single-name assignments; later definitions replace earlier ones."""
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            found[node.targets[0].id] = node
    return found


def _bound_names(node):
    """Names a definition binds itself: parameters, assignment targets, nested definitions, handlers, loop targets."""
    bound = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            bound.add(sub.id)
        elif isinstance(sub, ast.arg):
            bound.add(sub.arg)
        elif isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
            bound.add(sub.name)
        elif isinstance(sub, ast.ExceptHandler) and sub.name:
            bound.add(sub.name)
    return bound


def _check_assignment_calls(name, node):
    """An assignment (a module constant) may call only allowed-module attributes and a few builtin constructors."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        func = sub.func
        allowed_attribute = (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                             and func.value.id in ALLOWED_MODULES)
        allowed_builtin = isinstance(func, ast.Name) and func.id in _SAFE_BUILTIN_CALLS
        if not (allowed_attribute or allowed_builtin):
            raise PureExtractionError(f"assignment to {name!r} calls something that is not allowed: "
                                      f"{ast.dump(func)[:80]}")


def extract_pure(source_text, entry_names, module_name, filename="<source>"):
    """Execute only the definitions that `entry_names` need, from `source_text`, and return a SimpleNamespace.

    The namespace holds the selected definitions; `.pure_info` records the selected names and the SHA-256 of the
    source text (CRLF normalised), so a result can state exactly which source it exercised."""
    tree = ast.parse(source_text, filename=filename)
    definitions = _top_level_definitions(tree)
    unknown = [name for name in entry_names if name not in definitions]
    if unknown:
        raise PureExtractionError(f"{filename}: no top-level definition named {', '.join(unknown)}")

    selected, pending, unresolved = set(), list(entry_names), {}
    builtin_names = set(dir(builtins)) | {"__name__"}
    while pending:
        name = pending.pop()
        if name in selected:
            continue
        selected.add(name)
        node = definitions[name]
        if isinstance(node, ast.Assign):
            _check_assignment_calls(name, node)
        bound = _bound_names(node)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                used = sub.id
                if used in definitions and used not in bound:
                    pending.append(used)
                elif used not in bound and used not in ALLOWED_MODULES and used not in builtin_names:
                    unresolved.setdefault(name, set()).add(used)
    if unresolved:
        details = "; ".join(f"{owner} uses {', '.join(sorted(names))}" for owner, names in sorted(unresolved.items()))
        raise PureExtractionError(f"{filename}: names that are not defined, builtin or allowed: {details}")

    ordered = sorted((definitions[name] for name in selected), key=lambda node: node.lineno)
    module = ast.Module(body=ordered, type_ignores=[])
    namespace = {"__name__": module_name, "__builtins__": builtins}
    namespace.update(ALLOWED_MODULES)
    exec(compile(module, filename, "exec"), namespace)
    exported = {key: value for key, value in namespace.items()
                if not key.startswith("__") and key not in ALLOWED_MODULES}
    result = types.SimpleNamespace(**exported)
    digest = hashlib.sha256(source_text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    result.pure_info = {"source": filename, "source_sha256": digest, "selected": sorted(selected)}
    return result


def _read_text(root, relative):
    with open(os.path.join(root, *relative), "r", encoding="utf-8") as handle:
        return handle.read()


def load_parser(root=None):
    """The production E2 normaliser (src/parser.py), extracted without importing the module."""
    root = ROOT if root is None else root
    return extract_pure(_read_text(root, PARSER_RELATIVE), PARSER_ENTRIES, PARSER_MODULE_NAME, "src/parser.py")


def load_display(root=None):
    """The production course-eligibility logic (frontend/display.py), extracted without importing the module."""
    root = ROOT if root is None else root
    return extract_pure(_read_text(root, DISPLAY_RELATIVE), DISPLAY_ENTRIES, DISPLAY_MODULE_NAME, "frontend/display.py")
