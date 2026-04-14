#!/usr/bin/env python3
"""Merge Phase 5 BibTeX citations into dissertation bibliography.tex as \\bibitem entries.

The dissertation uses a hand-curated \\bibitem-based bibliography (NOT BibTeX).
This script parses outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib,
finds cite keys missing from outputs/dissertation/bibliography.tex, and appends
IEEE-style \\bibitem entries for them. Preserves existing entries intact.

Also appends a minimal placeholder for registry-only citations that don't live in
the BibTeX file (bourdenx2020, brooks1990, etc. — registry-verified but not in
the Phase 5 bibliography).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DISS_BIB = PROJECT_ROOT / "outputs/dissertation/bibliography.tex"
P5_BIB = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib"

# Registry-only fallbacks (cited in our new chapters but not in phase5 .bib)
FALLBACKS = {
    "bourdenx2020": r"\bibitem{bourdenx2020} M. Bourdenx, A. Nioche, S. Dovero, et al., ``Identification of distinct pathological signatures induced by patient-derived $\alpha$-synuclein structures in nonhuman primates,'' Science Advances, vol. 6, no. 20, eaaz9165, 2020. doi: 10.1126/sciadv.aaz9165",
    "brooks1990": r"\bibitem{brooks1990} D. J. Brooks, V. Ibanez, G. V. Sawle, et al., ``Differing patterns of striatal 18F-dopa uptake in Parkinson's disease, multiple system atrophy, and progressive supranuclear palsy,'' Annals of Neurology, vol. 28, no. 4, pp. 547--555, 1990. doi: 10.1002/ana.410280412",
    "contin1997": r"\bibitem{contin1997} M. Contin, R. Riva, P. Martinelli, et al., ``Longitudinal monitoring of the levodopa concentration-effect relationship in Parkinson's disease,'' Neurology, vol. 42, no. 11, pp. 2125--2130, 1992.",
    "datavalidation2026": r"\bibitem{datavalidation2026} B. Dupre, ``DATA\_LITERATURE\_REGISTRY for the mechanistic PD digital twin,'' Internal technical registry, 2026. Available at outputs/mechanistic\_twin/phase2/DATA\_LITERATURE\_REGISTRY.md",
    "drori2022": r"\bibitem{drori2022} E. Drori, Y. Berman, A. A. Mezer, ``Mapping microstructural gradients of the human striatum in normal aging and Parkinson's disease,'' Science Advances, vol. 8, no. 28, eabm1971, 2022. doi: 10.1126/sciadv.abm1971",
    "fearnleyLees1991": r"\bibitem{fearnleyLees1991} J. M. Fearnley and A. J. Lees, ``Ageing and Parkinson's disease: substantia nigra regional selectivity,'' Brain, vol. 114, pt 5, pp. 2283--2301, 1991. doi: 10.1093/brain/114.5.2283",
    "fu2022": r"\bibitem{fu2022} J. F. Fu, N. Klyuzhin, H. McKenzie, et al., ``Effects of anterior-posterior gradients of striatal dopamine loss on disease progression in Parkinson's disease,'' NeuroImage: Clinical, vol. 35, 103030, 2022. doi: 10.1016/j.nicl.2022.103030",
    "kerstens2023": r"\bibitem{kerstens2023} V. S. Kerstens, P. Svenningsson, A. Varrone, ``Annual decline in caudate and putaminal [123I]FP-CIT binding in Parkinson's disease,'' NeuroImage: Clinical, vol. 37, 103324, 2023. doi: 10.1016/j.nicl.2023.103324",
    "oh2012": r"\bibitem{oh2012} M. Oh, J. S. Kim, J. Y. Kim, et al., ``Subregional patterns of preferential striatal dopamine transporter loss differ in Parkinson disease, progressive supranuclear palsy, and multiple-system atrophy,'' Journal of Nuclear Medicine, vol. 53, no. 3, pp. 399--406, 2012. doi: 10.2967/jnumed.111.095224",
    "oliverassalva2013": r"\bibitem{oliverassalva2013} A. Oliveras-Salv\'a, A. Van der Perren, N. Casadei, et al., ``rAAV2/7 vector-mediated overexpression of alpha-synuclein in mouse substantia nigra induces protein aggregation and progressive dose-dependent neurodegeneration,'' Molecular Neurodegeneration, vol. 8, p. 44, 2013. doi: 10.1186/1750-1326-8-44",
    "simuni2018": r"\bibitem{simuni2018} T. Simuni, C. L. Siderowf, S. Lasch, et al., ``Longitudinal change of clinical and biological measures in early Parkinson's disease: PPMI cohort,'' Movement Disorders, vol. 33, no. 5, pp. 771--782, 2018. doi: 10.1002/mds.27361",
    "vogel2024": r"\bibitem{vogel2024} J. W. Vogel, J. L. Young, S. Sch\"oll, R. La Joie, et al., ``Four distinct trajectories of tau deposition identified in Alzheimer's disease,'' Nature Medicine, vol. 27, no. 5, pp. 871--881, 2021. doi: 10.1038/s41591-021-01309-6",
    "wakasugi2024combat": r"\bibitem{wakasugi2024combat} N. Wakasugi, H. Takano, M. Abe, et al., ``Harmonizing multisite data with the ComBat method for enhanced Parkinson's disease diagnosis via DAT-SPECT,'' Frontiers in Neurology, vol. 15, 1306546, 2024. doi: 10.3389/fneur.2024.1306546",
    "viceconti2021insilico": r"\bibitem{viceconti2021insilico} M. Viceconti, F. Pappalardo, B. Rodriguez, et al., ``In silico trials: Verification, validation and uncertainty quantification of predictive models used in the regulatory evaluation of biomedical products,'' Methods, vol. 185, pp. 120--127, 2021. doi: 10.1016/j.ymeth.2020.01.011",
    "viceconti2025credibilityml": r"\bibitem{viceconti2025credibilityml} M. Viceconti, F. Lanubile, A. Carbonaro, et al., ``Position paper: Extending credibility assessment of in silico medicine predictors to machine learning predictors,'' IEEE J. Biomed. Health Inform., vol. 29, no. 7, pp. 5284--5290, 2025. doi: 10.1109/JBHI.2025.3552320",
}


def parse_phase5_bib(path: Path) -> dict[str, dict]:
    """Very simple BibTeX parser — returns {citekey: {type, fields}}."""
    text = path.read_text()
    entries = {}
    # Match @type{citekey, ... fields ... }
    for match in re.finditer(r"@(\w+)\{([^,]+),\s*(.*?)\n\}", text, flags=re.DOTALL):
        etype, citekey, body = match.group(1), match.group(2).strip(), match.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        entries[citekey] = {"type": etype.lower(), "fields": fields}
    return entries


def bibtex_to_bibitem(citekey: str, entry: dict) -> str:
    """Convert a parsed BibTeX entry into an IEEE-style \\bibitem{} line."""
    f = entry["fields"]
    author = f.get("author", "Unknown")
    title = f.get("title", "[untitled]")
    # Cleanup: remove LaTeX-unfriendly characters in bibliography context
    title = title.replace("{", "").replace("}", "")
    author = author.replace("{", "").replace("}", "")

    journal = f.get("journal", "")
    volume = f.get("volume", "")
    number = f.get("number", "")
    pages = f.get("pages", "")
    year = f.get("year", "")
    doi = f.get("doi", "")
    howpublished = f.get("howpublished", "")
    eprint = f.get("eprint", "")

    parts = [author, f'``{title},\'\'']
    if journal:
        parts.append(journal + ",")
    elif howpublished:
        parts.append(howpublished + ",")
    if volume:
        vol_str = f"vol. {volume}"
        if number:
            vol_str += f", no. {number}"
        parts.append(vol_str + ",")
    if pages:
        parts.append(f"pp. {pages},")
    if year:
        parts.append(f"{year}.")
    if doi:
        parts.append(f"doi: {doi}")
    elif eprint:
        parts.append(f"arXiv:{eprint}")

    body = " ".join(parts)
    return f"\\bibitem{{{citekey}}} {body}"


def main() -> int:
    # Step 1: gather cite keys missing from dissertation
    chapters_dir = PROJECT_ROOT / "outputs/dissertation/chapters"
    new_chapters = [
        "ch10_paper8a",
        "ch11_paper8b",
        "ch12_paper9",
        "ch13_paper10",
        "ch14_discussion",
        "ch15_conclusion",
    ]
    cited = set()
    for ch in new_chapters:
        p = chapters_dir / f"{ch}.tex"
        if not p.exists():
            continue
        for m in re.finditer(r"\\cite\{([^}]+)\}", p.read_text()):
            for k in m.group(1).split(","):
                cited.add(k.strip())

    bib_text = DISS_BIB.read_text()
    defined = set(re.findall(r"\\bibitem\{([^}]+)\}", bib_text))
    missing = sorted(cited - defined)
    print(f"Missing cite keys: {len(missing)}")

    # Step 2: parse Phase 5 BibTeX
    p5 = parse_phase5_bib(P5_BIB)

    # Step 3: generate new \bibitem entries
    new_entries = []
    unresolved = []
    for key in missing:
        if key in p5:
            new_entries.append(bibtex_to_bibitem(key, p5[key]))
        elif key in FALLBACKS:
            new_entries.append(FALLBACKS[key])
        else:
            unresolved.append(key)

    print(f"Resolved from Phase 5 .bib: {sum(1 for k in missing if k in p5)}")
    print(f"Resolved from fallbacks: {sum(1 for k in missing if k in FALLBACKS)}")
    print(f"Unresolved: {len(unresolved)}: {unresolved}")

    # Step 4: append to dissertation bibliography under a clearly labeled section
    if new_entries:
        append_block = (
            "\n\n% =============================================================================\n"
            "% Phase 5 Task 5 + Task 6 + Task 7 citations (Papers 8a, 8b, 9, 10 chapters)\n"
            "% Merged 2026-04-13 from phase5_literature_bibliography.bib\n"
            "% =============================================================================\n"
            + "\n\n".join(new_entries)
            + "\n"
        )
        DISS_BIB.write_text(bib_text + append_block)
        print(f"Appended {len(new_entries)} \\bibitem entries to {DISS_BIB.name}")

    # Step 5: return non-zero if anything unresolved
    return 1 if unresolved else 0


if __name__ == "__main__":
    sys.exit(main())
