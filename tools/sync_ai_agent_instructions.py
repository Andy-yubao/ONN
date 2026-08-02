#!/usr/bin/env python
"""Sync AGENTS.md and CLAUDE.md from the single source of truth.

Source file : docs/ai_agent_instructions.md
Generated   : AGENTS.md, CLAUDE.md  (byte-identical, UTF-8, LF newlines)

Usage:
    python tools/sync_ai_agent_instructions.py          # regenerate both root files
    python tools/sync_ai_agent_instructions.py --check  # verify consistency (no writes, nonzero exit on drift)

No third-party dependencies. Works on Windows and Git Bash.
"""

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SOURCE = REPO / "docs" / "ai_agent_instructions.md"
TARGETS = [REPO / "AGENTS.md", REPO / "CLAUDE.md"]


def read_source():
    """Return the source text, normalized to UTF-8 text with LF newlines."""
    if not SOURCE.is_file():
        print(f"ERROR: source file not found: {SOURCE}", file=sys.stderr)
        sys.exit(1)
    data = SOURCE.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        print(f"ERROR: source file is not valid UTF-8: {exc}", file=sys.stderr)
        sys.exit(1)
    # Normalize CRLF / lone CR to LF.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def write_targets(text):
    encoded = text.encode("utf-8")
    for target in TARGETS:
        target.write_bytes(encoded)
        print(f"wrote {target.relative_to(REPO)}")


def check(text):
    """Verify every target is byte-identical to the normalized source."""
    encoded = text.encode("utf-8")
    ok = True
    for target in TARGETS:
        rel = target.relative_to(REPO)
        if not target.is_file():
            print(f"DRIFT: {rel} is missing", file=sys.stderr)
            ok = False
            continue
        if target.read_bytes() != encoded:
            print(f"DRIFT: {rel} differs from {SOURCE.relative_to(REPO)}", file=sys.stderr)
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate AGENTS.md and CLAUDE.md from docs/ai_agent_instructions.md, "
            "or check they are in sync."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify consistency without modifying files (nonzero exit on drift)",
    )
    args = parser.parse_args()

    text = read_source()

    if args.check:
        ok = check(text)
        if ok:
            print("OK: AGENTS.md and CLAUDE.md are in sync with docs/ai_agent_instructions.md")
            return 0
        print("FAIL: drift detected (see above)", file=sys.stderr)
        return 1

    write_targets(text)
    print("OK: regenerated AGENTS.md and CLAUDE.md from docs/ai_agent_instructions.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
