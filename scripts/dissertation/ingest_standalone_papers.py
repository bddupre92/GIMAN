#!/usr/bin/env python3
"""Ingest standalone paper8a, paper8b, paper9 LaTeX into dissertation chapters.

Strips the standalone preamble and `\\end{document}`, keeps the paper's section
structure (Introduction, Related Work, Methods, Results, Discussion, Supp),
converts natbib `\\citet{}` / `\\citep{}` to dissertation `\\cite{}`, prefixes
figure filenames with the paper tag, and wraps the whole thing in a `\\chapter`
with a dissertation-appropriate header. Also extracts the paper's own
bibliography entries and reports which cite keys are new vs already in
dissertation/bibliography.tex.

Output:
- outputs/dissertation/chapters/ch10_paper8a.tex (overwrites concise version)
- outputs/dissertation/chapters/ch11_paper8b.tex (overwrites)
- outputs/dissertation/chapters/ch12_paper9.tex (overwrites)
- Figures copied: outputs/dissertation/figures/p8a_*, p8b_*, p9_*
- Report printed: new cite keys per paper
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPERS = [
    {
        "tag": "p8a",
        "src": PROJECT_ROOT / "outputs/mechanistic_twin/paper8a_identifiability",
        "dst_tex": PROJECT_ROOT / "outputs/dissertation/chapters/ch10_paper8a.tex",
        "chapter_num": 10,
        "paper_num": "8a",
        "title": "Paper 8a: Practical Identifiability Limits of Spatial Propagation Parameters in Mechanistic PD Digital Twins",
        "label": "ch:paper8a",
    },
    {
        "tag": "p8b",
        "src": PROJECT_ROOT / "outputs/mechanistic_twin/paper8b_regional_rates",
        "dst_tex": PROJECT_ROOT / "outputs/dissertation/chapters/ch11_paper8b.tex",
        "chapter_num": 11,
        "paper_num": "8b",
        "title": "Paper 8b: Regional Dopamine Transporter Decline Rates from Serial DaT-SPECT",
        "label": "ch:paper8b",
    },
    {
        "tag": "p9",
        "src": PROJECT_ROOT / "outputs/mechanistic_twin/paper9_three_pathway_pkpd",
        "dst_tex": PROJECT_ROOT / "outputs/dissertation/chapters/ch12_paper9.tex",
        "chapter_num": 12,
        "paper_num": "9",
        "title": "Paper 9: Three-Pathway PK-PD Analysis --- DaT-SPECT-Calibrated Neurodegeneration Modulates Levodopa Benefit",
        "label": "ch:paper9",
    },
]

DISS_FIGS = PROJECT_ROOT / "outputs/dissertation/figures"
DISS_BIB = PROJECT_ROOT / "outputs/dissertation/bibliography.tex"


def copy_figures(src_dir: Path, tag: str) -> dict[str, str]:
    """Copy PNGs/PDFs with a per-paper prefix. Returns old→new mapping."""
    fig_src = src_dir / "figures"
    mapping: dict[str, str] = {}
    if not fig_src.exists():
        return mapping
    for src in fig_src.iterdir():
        if src.suffix.lower() not in {".pdf", ".png"}:
            continue
        new_name = f"{tag}_{src.name}"
        dst = DISS_FIGS / new_name
        shutil.copy2(src, dst)
        mapping[src.stem] = f"{tag}_{src.stem}"
        print(f"  copied {src.name} → {new_name}")
    return mapping


def extract_body(tex_path: Path) -> str:
    """Return the body between \\maketitle (inclusive of what follows) and \\begin{thebibliography}."""
    text = tex_path.read_text()
    # Find \maketitle
    m1 = re.search(r"\\maketitle\s*\n", text)
    if not m1:
        # Fallback: take everything after \\begin{document}
        m1 = re.search(r"\\begin\{document\}\s*\n", text)
        if not m1:
            raise ValueError(f"No \\maketitle or \\begin{{document}} in {tex_path}")
    body_start = m1.end()
    m2 = re.search(r"\\begin\{thebibliography\}", text)
    if not m2:
        # Fallback: take everything before \\end{document}
        m2 = re.search(r"\\end\{document\}", text)
    body_end = m2.start()
    return text[body_start:body_end]


def extract_bibitems(tex_path: Path) -> list[str]:
    """Return the list of \\bibitem{} entries (as raw lines) from the paper's own bibliography."""
    text = tex_path.read_text()
    m1 = re.search(r"\\begin\{thebibliography\}\{\d+\}", text)
    m2 = re.search(r"\\end\{thebibliography\}", text)
    if not (m1 and m2):
        return []
    bib_text = text[m1.end():m2.start()]
    # Split on \bibitem
    entries = re.split(r"(?=\\bibitem\{)", bib_text)
    return [e.strip() for e in entries if e.strip().startswith("\\bibitem")]


def rewrite_body(body: str, fig_map: dict[str, str]) -> str:
    """Convert natbib to \\cite and remap figure filenames."""
    # natbib → \cite{}
    body = re.sub(r"\\citep\{([^}]+)\}", r"\\cite{\1}", body)
    body = re.sub(r"\\citet\{([^}]+)\}", r"\\cite{\1}", body)
    body = re.sub(r"\\citeauthor\{([^}]+)\}", r"\\cite{\1}", body)
    # Remap includegraphics filenames (with or without extension)
    def _remap(m: re.Match) -> str:
        opts = m.group(1) or ""
        name = m.group(2)
        stem = Path(name).stem
        ext = Path(name).suffix
        new_stem = fig_map.get(stem, stem)
        return f"\\includegraphics{opts}{{{new_stem}{ext}}}"

    body = re.sub(
        r"\\includegraphics(\[[^\]]*\])?\{([^}]+)\}",
        _remap,
        body,
    )
    return body


def build_chapter(paper: dict, body: str) -> str:
    """Wrap body with chapter header + label."""
    header = f"""% =============================================================================
% Chapter {paper['chapter_num']} --- Paper {paper['paper_num']}
% Full manuscript content imported from outputs/mechanistic_twin/{paper['src'].name}/main.tex
% Preamble, title page, and standalone bibliography stripped; figure filenames
% remapped with prefix "{paper['tag']}_"; natbib \\citep/\\citet converted to \\cite.
% =============================================================================

\\chapter{{{paper['title']}}}
\\label{{{paper['label']}}}

"""
    return header + body.lstrip()


def diff_bibitems(paper_bibitems: list[str], diss_bib_text: str) -> tuple[list[str], list[str]]:
    """Return (already_present, new) bibitems. Match by cite key."""
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", diss_bib_text))
    present, new = [], []
    for entry in paper_bibitems:
        m = re.match(r"\\bibitem\{([^}]+)\}", entry)
        if not m:
            continue
        if m.group(1) in defined:
            present.append(entry)
        else:
            new.append(entry)
    return present, new


def main() -> None:
    DISS_FIGS.mkdir(parents=True, exist_ok=True)
    diss_bib_text = DISS_BIB.read_text()
    new_bibitems_all: list[str] = []

    for paper in PAPERS:
        print(f"\n=== {paper['tag']} ({paper['title'][:60]}...) ===")
        src_tex = paper["src"] / "main.tex"
        if not src_tex.exists():
            print(f"  MISSING: {src_tex}")
            continue

        fig_map = copy_figures(paper["src"], paper["tag"])

        body = extract_body(src_tex)
        body = rewrite_body(body, fig_map)
        chapter_text = build_chapter(paper, body)
        paper["dst_tex"].write_text(chapter_text)
        print(f"  wrote {paper['dst_tex'].relative_to(PROJECT_ROOT)} ({len(chapter_text)} chars)")

        bibitems = extract_bibitems(src_tex)
        present, new = diff_bibitems(bibitems, diss_bib_text)
        print(f"  bibitems: {len(bibitems)} total, {len(present)} already in dissertation, {len(new)} new")
        new_bibitems_all.extend(new)

    # Dedup new bibitems across papers (same cite key may appear in multiple papers)
    seen_keys: set[str] = set()
    deduped_new: list[str] = []
    for entry in new_bibitems_all:
        m = re.match(r"\\bibitem\{([^}]+)\}", entry)
        if m and m.group(1) not in seen_keys:
            seen_keys.add(m.group(1))
            deduped_new.append(entry)

    if deduped_new:
        append_block = (
            "\n\n% =============================================================================\n"
            "% Paper 8a/8b/9 standalone bibliography entries (merged 2026-04-13)\n"
            "% =============================================================================\n"
            + "\n\n".join(deduped_new)
            + "\n"
        )
        DISS_BIB.write_text(diss_bib_text + append_block)
        print(f"\nAppended {len(deduped_new)} new \\bibitem entries to {DISS_BIB.name}")
    else:
        print("\nNo new bibitems to append.")


if __name__ == "__main__":
    main()
