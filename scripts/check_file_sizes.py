"""Fail if any .py under src/, tests/ or scripts/ exceeds 150 LOC (CLAUDE.md §1).

LOC excludes blank lines and pure-comment lines, so docstrings count but
`# divider` separators do not. Exit 1 if any file is over the limit.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 150
SCAN = ("src", "tests", "scripts")
EXCLUDED_DIRS = {
    ".venv",
    ".git",
    "build",
    "dist",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "vendor",
}


def count_loc(path: Path) -> int:
    """Count non-blank, non-comment lines in a Python file."""
    loc = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        loc += 1
    return loc


def scan_dirs(root: Path, dirs: tuple[str, ...] = SCAN) -> list[tuple[Path, int]]:
    """Return [(path, loc)] for every .py over LIMIT under the given top-level dirs."""
    over: list[tuple[Path, int]] = []
    for top in dirs:
        base = root / top
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            loc = count_loc(path)
            if loc > LIMIT:
                over.append((path.relative_to(root), loc))
    return over


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    over = scan_dirs(root)
    if over:
        print(f"FAIL: {len(over)} file(s) exceed {LIMIT} LOC:")
        for path, loc in over:
            print(f"  {path}: {loc} LOC")
        return 1
    print(f"OK: all .py files under {SCAN} are <= {LIMIT} LOC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
