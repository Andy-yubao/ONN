#!/usr/bin/env python
"""Check relative links inside repository markdown/text documents.

- Checks every inline markdown link `[text](target)` and image link `![alt](target)`
  whose target is a repository-relative path.
- Ignores absolute URLs (http://, https://, ftp://, mailto:), anchors (#...), emails,
  and template placeholders containing `{`.
- Correctly handles URL-encoded targets and Chinese/Unicode file names.
- Does NOT test external web availability.
- Exits nonzero if any broken relative link is found.

Usage:
    python tools/check_markdown_links.py

No third-party dependencies. Works on Windows and Git Bash.
"""

import os
import pathlib
import re
import subprocess
import sys
import urllib.parse

REPO = pathlib.Path(__file__).resolve().parent.parent

LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+[\"'][^\"')]*[\"'])?\)")
FENCE_RE = re.compile(r"(?m)^```.*?^```", re.DOTALL)
CODE_SPAN_RE = re.compile(r"`[^`]*`")
IGNORE_SCHEMES = ("http://", "https://", "ftp://", "mailto:", "tel:", "data:")


def tracked_docs():
    """Return repo-relative paths of tracked .md/.txt files (via git ls-files)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.md", "*.txt"],
            cwd=str(REPO),
            capture_output=True,
            check=True,
            text=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        # Fallback: walk the tree, skipping VCS and ignored-heavy directories.
        files = []
        skip_dirs = {".git", "__pycache__", ".pytest_cache"}
        for root, dirs, names in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            rel = pathlib.Path(root).relative_to(REPO)
            if rel.parts and rel.parts[0] in ("model", "fpga", "essay"):
                if rel.parts[1:] and rel.parts[1] in ("runs", "data", "db", "incremental_db"):
                    dirs[:] = []
            for name in names:
                if name.endswith((".md", ".txt")):
                    files.append((rel / name).as_posix())
        return sorted(files)
    return sorted(out.stdout.splitlines())


def resolve_target(file_rel, raw_target):
    """Resolve a raw link target to a repo-relative path (or None to skip)."""
    if raw_target.startswith(IGNORE_SCHEMES) or raw_target.startswith("#"):
        return None
    if "{" in raw_target or "}" in raw_target:
        return None  # template placeholder, e.g. <run-id> variants or {table}
    # Strip any fragment (URL fragment after #).
    path_part = raw_target.split("#", 1)[0]
    if not path_part:
        return None
    # URL-decode percent escapes.
    path_part = urllib.parse.unquote(path_part)
    if path_part.startswith("/"):
        resolved = REPO / path_part.lstrip("/")
    else:
        resolved = (REPO / file_rel).parent / path_part
    return resolved


def link_targets(file_rel, text):
    """Yield (lineno, raw_target) for every inline link outside code spans/blocks."""
    stripped = FENCE_RE.sub("", text)  # strip fenced code blocks first
    stripped = CODE_SPAN_RE.sub("", stripped)  # then inline code spans
    for match in LINK_RE.finditer(stripped):
        raw = match.group(1).strip()
        lineno = stripped.count("\n", 0, match.start()) + 1
        yield lineno, raw


def main():
    broken = []
    for file_rel in tracked_docs():
        path = REPO / file_rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, raw in link_targets(file_rel, text):
            resolved = resolve_target(file_rel, raw)
            if resolved is None:
                continue
            if not resolved.exists():
                broken.append((file_rel, lineno, raw, str(resolved)))

    if broken:
        for file_rel, lineno, raw, resolved in broken:
            print(f"BROKEN: {file_rel}:{lineno}  ->  {raw}  (expected at {resolved})")
        print(f"FAIL: {len(broken)} broken relative link(s) found", file=sys.stderr)
        return 1

    print("OK: no broken relative links found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
