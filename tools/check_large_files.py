#!/usr/bin/env python3
"""Block commits of large files.

Rules (the same in every repository of the project):

* up to 10 MB: allowed;
* over 10 MB and up to 50 MB: the first commit is refused with a warning; running the same commit again (the same file
  contents staged) goes through -- the refusal is remembered in ``.git/large_files_warned``;
* over 50 MB: never allowed.

Usage:
    python tools/check_large_files.py              # check the staged files (what the pre-commit hook runs)
    python tools/check_large_files.py --all        # check every tracked and untracked, not-ignored file
    python tools/check_large_files.py --untracked  # check untracked, not-ignored files (what `git add .` would add)
    python tools/check_large_files.py --install    # install this script as the repository's git pre-commit hook

Exit code 1 when a file breaks the rules. Note: ``git commit --no-verify`` skips every hook; do not use it to commit large files.
"""
import stat
import subprocess
import sys
from pathlib import Path

SOFT_LIMIT_MB = 10   # above this a file must be listed in .large_files_allowed
HARD_LIMIT_MB = 50   # above this a file is never committed
WARNED_FILE = "large_files_warned"   # inside .git: blobs already refused once (path + blob id)


def git(*args: str) -> str:
    """Run a git command in the current repository and return its standard output."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def sizes(check_all: bool, untracked: bool = False):
    """Yield ``(path, size in bytes, blob id or None)`` of the staged files (default), every file (``check_all``) or the untracked ones."""
    if untracked or check_all:
        args = ["ls-files", "-z"] + (["--others", "--exclude-standard"] if untracked else ["--cached", "--others", "--exclude-standard"])
        for path in git(*args).split("\0"):
            if path and Path(path).is_file():
                yield path, Path(path).stat().st_size, None
    else:
        for path in git("diff", "--cached", "--name-only", "--diff-filter=AM", "-z").split("\0"):
            if path:
                yield path, int(git("cat-file", "-s", f":{path}")), git("rev-parse", f":{path}").strip()


def warned_file() -> Path:
    """``.git/large_files_warned``: one ``<blob id> <path>`` line per large file refused once."""
    return Path(git("rev-parse", "--git-path", WARNED_FILE).strip())


def warned() -> set:
    """``(blob id, path)`` pairs already refused once."""
    f = warned_file()
    if not f.is_file():
        return set()
    return {tuple(line.split(" ", 1)) for line in f.read_text(encoding="utf-8").splitlines() if " " in line}


def install() -> None:
    """Write ``.git/hooks/pre-commit`` so that every commit runs this check."""
    hook = Path(git("rev-parse", "--git-path", "hooks/pre-commit").strip())
    script = Path(__file__).resolve().relative_to(Path(git("rev-parse", "--show-toplevel").strip()))
    if hook.exists() and "check_large_files" not in hook.read_text():
        sys.exit(f"{hook} already exists (another tool?) -- add `python3 {script}` to it by hand.")
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f'#!/bin/sh\nexec python3 "$(git rev-parse --show-toplevel)/{script}"\n')
    hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"installed {hook}")


def main() -> int:
    """Check the files and report every one that breaks the rules."""
    if "--install" in sys.argv:
        install()
        return 0
    staged = not ("--all" in sys.argv or "--untracked" in sys.argv)
    seen = warned() if staged else set()
    soft, hard = SOFT_LIMIT_MB * 1_000_000, HARD_LIMIT_MB * 1_000_000
    bad, new_warn = [], []
    for path, size, blob in sizes("--all" in sys.argv, "--untracked" in sys.argv):
        if size > hard:
            bad.append((path, size, f"over {HARD_LIMIT_MB} MB: never committed"))
        elif size > soft and (blob, path) not in seen:
            bad.append((path, size, f"over {SOFT_LIMIT_MB} MB: run the same commit again if you really want it"))
            if blob:
                new_warn.append(f"{blob} {path}")
    for path, size, why in sorted(bad, key=lambda x: -x[1]):
        print(f"LARGE FILE: {path} ({size / 1e6:.1f} MB) -- {why}", file=sys.stderr)
    if new_warn:
        with open(warned_file(), "a", encoding="utf-8") as fh:
            fh.write("\n".join(new_warn) + "\n")
    if bad:
        print("Otherwise remove it from the commit (`git reset <file>`) and add it to .gitignore.", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
