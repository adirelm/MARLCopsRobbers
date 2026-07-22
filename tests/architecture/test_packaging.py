"""V3 §14 packaging gate — the project really is an installable, declared package.

§14 asks for four things a grader can check mechanically: every source directory is a
real package (``__init__.py``), every package declares its public surface (``__all__``),
that surface is HONEST (a listed name must actually resolve — a stale export is worse
than none), and the distribution metadata (name/version/authors) agrees with
``src.__version__``. These run against the REPO ITSELF, so drift fails CI.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import src

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"


def _package_dirs() -> list[Path]:
    """Return every source directory that must be a package (``__pycache__`` aside)."""
    dirs = [_SRC] + [p for p in _SRC.rglob("*") if p.is_dir() and p.name != "__pycache__"]
    # Only directories that actually carry Python sources are packages.
    return sorted(d for d in dirs if any(f.suffix == ".py" for f in d.iterdir() if f.is_file()))


def _module_name(path: Path) -> str:
    """Return the dotted import path of a package directory (``src.marl.env``)."""
    return ".".join(path.relative_to(_ROOT).parts)


def _pyproject() -> dict:
    """Return the parsed ``pyproject.toml``."""
    return tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("package", _package_dirs(), ids=_module_name)
def test_every_source_directory_is_a_package(package: Path) -> None:
    """Every source directory carries an ``__init__.py`` (§14)."""
    assert (package / "__init__.py").is_file(), f"{_module_name(package)} has no __init__.py"


@pytest.mark.parametrize("package", _package_dirs(), ids=_module_name)
def test_every_package_declares_all(package: Path) -> None:
    """Every package declares a non-empty ``__all__`` and a module docstring (§14/§16)."""
    module = importlib.import_module(_module_name(package))
    assert module.__doc__, f"{module.__name__} has no module docstring"
    exported = getattr(module, "__all__", None)
    assert exported, f"{module.__name__} declares no __all__"
    assert all(isinstance(name, str) for name in exported)


@pytest.mark.parametrize("package", _package_dirs(), ids=_module_name)
def test_every_exported_name_resolves(package: Path) -> None:
    """Each ``__all__`` entry resolves — as an attribute or as an importable submodule.

    This is the test that catches a STALE export: renaming or deleting a module without
    updating ``__all__`` leaves a name that ``from pkg import *`` would fail on.
    """
    name = _module_name(package)
    module = importlib.import_module(name)
    unresolved: list[str] = []
    for exported in module.__all__:
        if hasattr(module, exported):
            continue
        try:
            importlib.import_module(f"{name}.{exported}")
        except ImportError:
            unresolved.append(exported)
    assert not unresolved, f"{name}.__all__ lists unresolvable names: {unresolved}"


@pytest.mark.parametrize("package", _package_dirs(), ids=_module_name)
def test_no_private_names_exported(package: Path) -> None:
    """No ``__all__`` leaks a private helper (a leading-underscore, non-dunder name)."""
    module = importlib.import_module(_module_name(package))
    private = [n for n in module.__all__ if n.startswith("_") and not n.startswith("__")]
    assert not private, f"{module.__name__} exports private names: {private}"


def test_pyproject_declares_name_version_authors() -> None:
    """The distribution metadata §14 requires is present and non-placeholder."""
    project = _pyproject()["project"]
    assert project["name"] == "marlcopsrobbers"
    assert project["version"]
    authors = project["authors"]
    assert authors, "pyproject declares no authors"
    for author in authors:
        assert author.get("name"), "an author entry has no name"
        assert "@" in author.get("email", ""), "an author entry has no email"


def test_version_matches_pyproject() -> None:
    """``src.__version__`` is the single version of record and matches pyproject (§8.1)."""
    assert src.__version__ == _pyproject()["project"]["version"]


def test_root_package_exports_version() -> None:
    """``src.__version__`` is part of the declared root surface."""
    assert "__version__" in src.__all__
