"""Bundle the whole source tree into one reviewable text file.

    python export_codebase.py

Skips data and binaries (dataset, model weights, node_modules, lockfiles) and
redacts anything that looks like a credential, so the result is safe to share.
"""

import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "MindScanAI_codebase.txt"

# Ordered so a reader meets the project the way it is described in the README.
SECTIONS = [
    ("PROJECT ROOT", ["README.md", ".gitignore", ".env.example", "docker-compose.yml",
                      "requirements.txt", "requirements-ml.txt"]),
    ("BACKEND", ["backend/**/*.py", "backend/**/*.txt", "backend/**/*.md", "backend/Dockerfile"]),
    ("TRAINING", ["training/**/*.py"]),
    ("FRONTEND — CONFIG", ["frontend/package.json", "frontend/vite.config.js",
                           "frontend/tailwind.config.js", "frontend/postcss.config.js",
                           "frontend/index.html", "frontend/Dockerfile"]),
    ("FRONTEND — SOURCE", ["frontend/src/**/*.jsx", "frontend/src/**/*.js", "frontend/src/**/*.css"]),
    ("DOCS", ["docs/**/*.md"]),
]

# Data, binaries and generated trees carry no review value.
EXCLUDE_PARTS = {
    "node_modules", ".git", "dataset", "__pycache__", ".venv", "venv",
    "mediapipe-wasm", "models", "dist", ".vite",
}
EXCLUDE_SUFFIX = {".pt", ".onnx", ".task", ".wasm", ".mp4", ".png", ".jpg", ".jpeg",
                  ".ico", ".svg", ".wav", ".csv", ".db", ".lock", ".zip"}
EXCLUDE_NAMES = {"package-lock.json", "requirements-lock.txt", ".env",
                 "MindScanAI_codebase.txt", "export_codebase.py"}

# Belt and braces: the export must never carry a live credential even if one
# is ever hard-coded by mistake.
REDACTIONS = [
    (re.compile(r"\bgsk_[A-Za-z0-9]{20,}"), "gsk_<REDACTED_GROQ_KEY>"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "sk-<REDACTED_KEY>"),
    # [ \t]* rather than \s*: on a bare "KEY=" with no value, \s* would cross
    # the newline and swallow the following line as the "value", deleting a
    # legitimate line from the export.
    (re.compile(r"(SECRET_KEY[ \t]*[:=][ \t]*)(\S+)"), r"\1<REDACTED>"),
    (re.compile(r"(API_KEY[ \t]*=[ \t]*)(\S+)"), r"\1<REDACTED>"),
]


def excluded(p: Path) -> bool:
    if p.name in EXCLUDE_NAMES or p.suffix.lower() in EXCLUDE_SUFFIX:
        return True
    return any(part in EXCLUDE_PARTS for part in p.relative_to(ROOT).parts)


def redact(text: str) -> tuple[str, int]:
    n = 0
    for pattern, repl in REDACTIONS:
        text, hits = pattern.subn(repl, text)
        n += hits
    return text, n


def main():
    seen: set[Path] = set()
    chunks: list[str] = []
    counts: dict[str, int] = {}
    total_lines = redactions = 0

    for title, patterns in SECTIONS:
        files: list[Path] = []
        for pat in patterns:
            for p in sorted(ROOT.glob(pat)):
                if p.is_file() and p not in seen and not excluded(p):
                    seen.add(p)
                    files.append(p)
        if not files:
            continue

        counts[title] = len(files)
        chunks.append("\n" + "=" * 78)
        chunks.append(f"  {title}")
        chunks.append("=" * 78 + "\n")

        for p in files:
            rel = p.relative_to(ROOT).as_posix()
            try:
                body = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            body, hits = redact(body)
            redactions += hits
            total_lines += body.count("\n") + 1
            chunks.append("-" * 78)
            chunks.append(f"FILE: {rel}")
            chunks.append("-" * 78)
            chunks.append(body.rstrip() + "\n")

    header = [
        "=" * 78,
        "  MindScan AI — full source export",
        "=" * 78,
        f"  Generated : {datetime.now().strftime('%d %b %Y, %I:%M %p IST')}",
        f"  Files     : {len(seen)}",
        f"  Lines     : {total_lines:,}",
        "",
        "  Excluded  : dataset, model weights (.pt/.onnx/.task), node_modules,",
        "              lockfiles, binary assets, and .env",
        f"  Redacted  : {redactions} credential-shaped string(s)",
        "",
        "  Contents:",
    ]
    header += [f"    {t:<24} {n:>3} files" for t, n in counts.items()]
    header.append("=" * 78)

    OUT.write_text("\n".join(header) + "\n" + "\n".join(chunks), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"{OUT.name}: {len(seen)} files, {total_lines:,} lines, {kb:.0f} KB")
    for t, n in counts.items():
        print(f"  {t:<24} {n:>3}")
    print(f"redacted {redactions} credential-shaped strings")


if __name__ == "__main__":
    main()
