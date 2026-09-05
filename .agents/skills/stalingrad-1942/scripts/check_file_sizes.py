"""Enforce the 700-line-per-file budget for the Stalingrad 1942 workspace.

Usage:
    python check_file_sizes.py [workspace-root] [max-lines]

Defaults: workspace root = this script's repo root (the directory five
levels up), max-lines = 700. Prints a line-count report and exits non-zero
if any tracked file exceeds the budget. Stdlib only — no third-party imports.
"""

import sys
from pathlib import Path

TRACKED_EXTENSIONS = {".py", ".html", ".js", ".css", ".md", ".bat"}
SKIP_DIR_NAMES = {"__pycache__", ".git", "node_modules", "dist", "build"}


def count_lines(path):
    """Physical line count (blank lines included); None on read failure."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return len(fh.read().splitlines())
    except OSError:
        return None


def discover_files(root):
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TRACKED_EXTENSIONS:
            continue
        parts = path.relative_to(root).parts[:-1]
        if any(part in SKIP_DIR_NAMES for part in parts):
            continue
        found.append(path)
    return found


def main(argv):
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parents[4]
    max_lines = int(argv[2]) if len(argv) > 2 else 700
    if not root.is_dir():
        print("ERROR: not a directory: %s" % root)
        return 2

    rows = []
    failures = []
    for path in discover_files(root):
        count = count_lines(path)
        rel = path.relative_to(root).as_posix()
        if count is None:
            failures.append("%s (unreadable)" % rel)
            continue
        rows.append((count, rel))
        if count > max_lines:
            failures.append("%s (%d > %d)" % (rel, count, max_lines))

    rows.sort(reverse=True)
    width = max(len(str(count)) for count, _ in rows) if rows else 1
    for count, rel in rows:
        flag = "  <-- OVER BUDGET" if count > max_lines else ""
        print("%*d lines  %s%s" % (width, count, rel, flag))

    print("")
    print("%d files checked against a %d-line budget; %d over."
          % (len(rows), max_lines, len(failures)))
    for failure in failures:
        print("OVER: %s" % failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))