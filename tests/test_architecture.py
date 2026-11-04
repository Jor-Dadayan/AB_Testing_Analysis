"""Автоматическая проверка разделения ответственности, дублей и исключений."""

import ast
import re
from pathlib import Path

import pytest

from experimentlab.config import PROJECT_ROOT

SRC = PROJECT_ROOT / "src" / "experimentlab"
# (пакет, запрещённые импорты, файлы-исключения)
RULES = [
    ("analysis", {"streamlit", "sqlalchemy", "experimentlab.db", "experimentlab.app"}, {"eda.py"}),
    (
        "db",
        {"scipy", "statsmodels", "streamlit", "experimentlab.analysis", "experimentlab.app"},
        set(),
    ),
    ("data", {"sqlalchemy", "streamlit", "experimentlab.db", "experimentlab.app"}, set()),
    ("app", {"scipy", "statsmodels", "sqlalchemy"}, {"services.py"}),
]
EXCEPT_EXCEPTION_ALLOWED = {"services.py", "components.py", "loader.py"}  # границы системы, с логом


def _modules(package: str | None = None):
    base = SRC / package if package else SRC
    return sorted(p for p in base.rglob("*.py") if p.name != "__init__.py")


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("package,forbidden,exempt", RULES, ids=[r[0] for r in RULES])
def test_layers_do_not_import_forbidden_modules(package, forbidden, exempt):
    violations = []
    for path in _modules(package):
        if path.name in exempt:
            continue
        for name in _imports(path):
            if any(name == f or name.startswith(f + ".") for f in forbidden):
                violations.append(f"{path.relative_to(SRC)} импортирует {name}")
    assert violations == []


def test_no_sql_text_in_analysis_or_app():
    pattern = re.compile(r"\bSELECT\b[\s\S]*\bFROM\b", re.IGNORECASE)
    hits = []
    for package in ("analysis", "app"):
        for path in _modules(package):
            for node in ast.walk(_tree(path)):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and pattern.search(node.value)
                ):
                    hits.append(str(path.relative_to(SRC)))
    assert hits == []


def test_no_print_in_library_code():
    hits = [
        str(p.relative_to(SRC))
        for p in _modules()
        if p.name != "check_report.py"
        for n in ast.walk(_tree(p))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
    ]
    assert hits == []


def test_no_bare_except_and_broad_except_only_at_boundaries():
    bare, broad = [], []
    for path in _modules():
        for n in ast.walk(_tree(path)):
            if isinstance(n, ast.ExceptHandler):
                if n.type is None:
                    bare.append(str(path.relative_to(SRC)))
                elif (
                    isinstance(n.type, ast.Name)
                    and n.type.id == "Exception"
                    and path.name not in EXCEPT_EXCEPTION_ALLOWED
                ):
                    broad.append(str(path.relative_to(SRC)))
    assert bare == [] and broad == []


def test_no_duplicated_function_bodies():
    seen: dict[str, str] = {}
    duplicates = []
    for path in _modules():
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = node.body[1:] if ast.get_docstring(node) else node.body
            shape = ast.dump(ast.Module(body=body, type_ignores=[])) + ast.dump(node.args)
            if len(shape) < 150:
                continue
            where = f"{path.relative_to(SRC)}:{node.name}"
            if shape in seen:
                duplicates.append((seen[shape], where))
            seen[shape] = where
    assert duplicates == []


def test_check_dataclass_has_single_definition():
    from experimentlab.data import quality
    from experimentlab.db import checks

    assert quality.Check is checks.Check
