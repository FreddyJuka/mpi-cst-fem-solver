#!/usr/bin/env python3
"""Checks that every relative link and image path in README.md, CHANGELOG.md and docs/*.md exists.

Markdown links `[text](target)` and images `![alt](target)`; absolute URLs (http, https, mailto)
are skipped. The file part of `target` is resolved relative to the Markdown file; `#anchor`
parts are checked against the headings of the target file (GitHub-style slugs). Fenced code
blocks are ignored. Exits non-zero if anything is broken.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def strip_code(text):
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def slug(heading):
    s = re.sub(r"[`*_]", "", heading.strip().lower())
    s = re.sub(r"[^\w\s-]", "", s)
    return re.sub(r"\s", "-", s)


def anchors(path):
    text = path.read_text()
    fenced, result = False, set()
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and line.startswith("#"):
            result.add(slug(line.lstrip("#")))
    return result


def main():
    files = [ROOT / "README.md", ROOT / "CHANGELOG.md", *sorted((ROOT / "docs").glob("*.md"))]
    bad = checked = 0
    for f in files:
        if not f.exists():
            continue
        for target in LINK.findall(strip_code(f.read_text())):
            if re.match(r"^(https?:|mailto:)", target):
                continue
            checked += 1
            name, _, frag = target.partition("#")
            dest = f if not name else (f.parent / name).resolve()
            if not dest.exists():
                print(f"BROKEN {f.relative_to(ROOT)}: {target} (no such file)")
                bad += 1
            elif frag and dest.suffix == ".md" and frag not in anchors(dest):
                print(f"BROKEN {f.relative_to(ROOT)}: {target} (no such heading)")
                bad += 1
    print(f"checked {checked} relative links in {len(files)} files, {bad} broken")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
