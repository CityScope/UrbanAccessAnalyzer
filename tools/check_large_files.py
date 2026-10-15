#!/usr/bin/env python3
"""Block commits that add large files (GitHub warns at 50 MB and rejects pushes at 100 MB).

Usage:
    python tools/check_large_files.py            # check staged files (used by the pre-commit hook)
    python tools/check_large_files.py --all      # check every non-ignored file (tracked + untracked)
    python tools/check_large_files.py --untracked  # check untracked, not-ignored files (what `git add .` would add)
    python tools/check_large_files.py --install  # install this as the repo's git pre-commit hook

Limit defaults to 50 MB; override with LARGE_FILE_MB=NN. Bypass once with `git commit --no-verify`.
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

LIMIT = float(os.environ.get("LARGE_FILE_MB", "50")) * 1024 * 1024


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def sizes(check_all: bool, untracked: bool = False):
    if untracked or check_all:
        args = ["ls-files", "-z"] + (["--others", "--exclude-standard"] if untracked else ["--cached", "--others", "--exclude-standard"])
        for path in git(*args).split("\0"):
            if path and Path(path).is_file():
                yield path, Path(path).stat().st_size
    else:
        for path in git("diff", "--cached", "--name-only", "--diff-filter=AM", "-z").split("\0"):
            if path:
                yield path, int(git("cat-file", "-s", f":{path}"))


def install() -> None:
    hook = Path(git("rev-parse", "--git-path", "hooks/pre-commit").strip())
    script = Path(__file__).resolve().relative_to(Path(git("rev-parse", "--show-toplevel").strip()))
    if hook.exists() and "check_large_files" not in hook.read_text():
        sys.exit(f"{hook} already exists (another tool?) -- add `python {script}` to it manually.")
    hook.write_text(f'#!/bin/sh\nexec python3 "$(git rev-parse --show-toplevel)/{script}"\n')
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {hook}")


def main() -> int:
    if "--install" in sys.argv:
        install()
        return 0
    big = [(p, s) for p, s in sizes("--all" in sys.argv, "--untracked" in sys.argv) if s > LIMIT]
    for path, size in sorted(big, key=lambda x: -x[1]):
        print(f"LARGE FILE: {path} ({size / 1e6:.1f} MB > {LIMIT / 1e6:.0f} MB)", file=sys.stderr)
    if big:
        print("Add it to .gitignore / `git reset <file>`, or bypass with --no-verify.", file=sys.stderr)
    return 1 if big else 0


if __name__ == "__main__":
    sys.exit(main())
