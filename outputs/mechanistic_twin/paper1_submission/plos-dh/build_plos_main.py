"""Build a PLOS-formatted main.tex from the IEEE-formatted archive chapter_content.tex.

Per PLOS guidelines:
- Single .tex file (no \\input)
- \\section*{} (unnumbered) for major sections
- \\subsection*{} (unnumbered) for sub-sections
- "Fig~\\ref{}" not "Figure~\\ref{}"
- Standard sections: Abstract / Author summary / Introduction / Materials and methods /
  Results / Discussion / Conclusion / Supporting information / Acknowledgments / References
- No IEEE-specific commands (\\IEEEkeywords, \\markboth, etc.)
- No graphics inside \\begin{figure} (figures uploaded separately)

Strategy:
1. Read archive's chapter_content.tex as the body source (preserves all content)
2. Strip IEEE-specific commands
3. Convert section levels to \\section*{} / \\subsection*{}
4. Replace "Figure~\\ref" → "Fig~\\ref" and "Figure~" → "Fig~"
5. Strip \\includegraphics (PLOS wants figure files separate)
6. Read archive's bibliography_extracted.tex and embed as thebibliography{}
7. Wrap with PLOS preamble + title block + abstract + author summary
"""

from __future__ import annotations

import re
from pathlib import Path

ARCHIVE = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/outputs/mechanistic_twin/paper1_submission/_archive_2026-04-25_full_27page")
PLOS_DIR = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/outputs/mechanistic_twin/paper1_submission/plos-dh")

PLOS_PREAMBLE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[top=0.85in,left=2.75in,footskip=0.75in]{geometry}

\usepackage{amsmath,amssymb}
\usepackage{changepage}
\usepackage{textcomp,marvosym}
\usepackage{cite}
\usepackage{nameref,hyperref}
\hypersetup{hidelinks=true}
\usepackage[right]{lineno}
\usepackage[nopatch=eqnum]{microtype}
\DisableLigatures[f]{encoding = *, family = * }
\usepackage[table]{xcolor}
\usepackage{array}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{enumitem}

\newcolumntype{+}{!{\vrule width 2pt}}
\newlength\savedwidth
\newcommand\thickcline[1]{%
  \noalign{\global\savedwidth\arrayrulewidth\global\arrayrulewidth 2pt}%
  \cline{#1}%
  \noalign{\vskip\arrayrulewidth}%
  \noalign{\global\arrayrulewidth\savedwidth}%
}
\newcommand\thickhline{\noalign{\global\savedwidth\arrayrulewidth\global\arrayrulewidth 2pt}%
\hline
\noalign{\global\arrayrulewidth\savedwidth}}

\raggedright
\setlength{\parindent}{0.5cm}
\textwidth 5.25in
\textheight 8.75in

\usepackage[aboveskip=1pt,labelfont=bf,labelsep=period,justification=raggedright,singlelinecheck=off]{caption}
\renewcommand{\figurename}{Fig}

\makeatletter
\renewcommand{\@biblabel}[1]{\quad#1.}
\makeatother

\usepackage{lastpage,fancyhdr,graphicx}
\usepackage{epstopdf}
\pagestyle{fancy}
\fancyhf{}
\rfoot{\thepage/\pageref{LastPage}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrule}{\hrule height 2pt \vspace{2mm}}
\fancyheadoffset[L]{2.25in}
\fancyfootoffset[L]{2.25in}
\lfoot{\today}

\graphicspath{{figures/}}

% Custom macros (preserved from IEEE submission)
\newcommand{\etal}{\textit{et al.}}
\newcommand{\eg}{e.g.}
\newcommand{\ie}{i.e.}

\begin{document}
\vspace*{0.2in}

\begin{flushleft}
{\Large
\textbf{Benchmarked machine learning for neuronal $\alpha$-synuclein disease integrated staging system (NSD-ISS) prediction with calibrated uncertainty in Parkinson's disease}
}
\newline
\\
Blair Dupre\textsuperscript{1*}
\\
\bigskip
\textbf{1} Department of Biomedical Engineering, University of North Dakota, Grand Forks, North Dakota, United States of America
\\
\bigskip

* blair.dupre@und.edu

\end{flushleft}

\section*{Abstract}
\textbf{Background.} The Neuronal $\alpha$-Synuclein Disease Integrated Staging System (NSD-ISS) redefines Parkinson's disease (PD) through two biological anchors (S: $\alpha$-synuclein seed amplification assay; D: dopamine-transporter SPECT), but no computational model predicts NSD-ISS stage from routinely collected clinical data with calibrated uncertainty. \textbf{Methods.} Using the Parkinson's Progression Markers Initiative (PPMI; n=2,201), we benchmark a strict-circularity specification excluding the four variables entering the NSD-ISS staging algorithm, retaining 21 non-staging features across seven clinical domains. Five state-of-the-art tabular methods (CatBoost default+HPO, LightGBM HPO, TabPFN v2, AutoGluon) are evaluated alongside graph-attention baselines, with cross-conformal CV+ uncertainty quantification at three confidence levels. External validation uses BioFIND (n=103) staged by Russo et al. 2025 and the externally-validated Venuto et al. 2025 SAA prediction model for construct-validity assessment of D-anchor-driven labels. \textbf{Results.} CatBoost (default) achieves binary NSD$+$ AUC 0.901 and NSD$+$ sub-staging AUC 0.908; the five SOTA methods are statistically equivalent at $\varepsilon{=}0.02$ (paired-bootstrap TOST: 34/40 pairs). Removing imaging and genetics features cuts binary AUC by $-$17.6 percentage points but leaves NSD$+$ sub-staging unchanged ($\Delta{=}+0.008$, p=0.21), validating a two-stage deployment. Cross-conformal CV+ achieves $>$90\% internal coverage; external coverage on BioFIND holds for binary (0.915) but collapses for multiclass without recalibration, restored to 0.944 by Mondrian per-class CP on n$\geq$50 labelled local patients. The Venuto-imputed S$+$ rate among 647 SAA-untested NSD-positive PD patients is 91.5\%, in the literature band of 88--93\%, supporting construct validity of D-anchor-driven labels. \textbf{Conclusions.} A strict-circularity NSD-ISS staging benchmark with calibrated uncertainty supports a deployment-ready two-stage clinical pipeline: imaging-anchored binary detection followed by clinical-only sub-staging, with on-site Mondrian conformal recalibration on small labelled local cohorts for multiclass external transport.

\section*{Author summary}
Parkinson's disease is now redefined biologically by two markers: $\alpha$-synuclein protein aggregates detected in cerebrospinal fluid, and dopamine transporter activity measured by brain imaging. Together, they define a six-stage staging system that maps biological progression---but predicting which stage a patient is in, from routine clinical data, has not been computationally benchmarked. We trained machine learning models on 2,201 PPMI patients to predict NSD-ISS stage and quantified uncertainty using conformal prediction. We found that gradient-boosted trees and modern tabular foundation models achieve statistically equivalent performance ($>$90\% AUC for binary detection), that biological-marker imaging is essential for the initial binary detection but clinical features alone suffice for finer sub-staging, and that distribution-free uncertainty guarantees transport to a new cohort only after on-site recalibration on $\sim$40--50 labelled local patients. The work closes the methodology gap for deploying NSD-ISS in clinical practice---particularly at community sites without imaging access---and provides the first construct-validity check using an externally-validated S-anchor predictor to verify that PPMI's labels are biologically concordant rather than rule-recapitulation artifacts.

\clearpage
\newgeometry{top=0.85in,left=1in,right=1in,footskip=0.75in}
\linenumbers

"""

PLOS_POSTAMBLE = r"""

\section*{Acknowledgments}
The author thanks the Parkinson's Progression Markers Initiative (PPMI) for cohort access; data used in preparation of this article were obtained from the PPMI database (\url{https://www.ppmi-info.org/access-data-specimens/download-data}). Computational resources were provided by the Department of Biomedical Engineering, University of North Dakota.

\nolinenumbers

"""


def main() -> None:
    body = (ARCHIVE / "chapter_content.tex").read_text()
    bib = (ARCHIVE / "bibliography_extracted.tex").read_text()

    # 1. Strip IEEE-specific commands
    body = re.sub(r"\\label\{p1:ch:paper1\}", "", body)
    body = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\\begin\{IEEEkeywords\}.*?\\end\{IEEEkeywords\}", "", body, flags=re.DOTALL)
    body = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w+)\}", r"\1\2", body)

    # 2. Strip the original "Supplementary Information" pointer paragraph block
    body = re.sub(
        r"\\section\*\{Supplementary Information\}.*?(?=\Z)",
        "",
        body,
        flags=re.DOTALL,
    )
    body = re.sub(r"\\section\*\{Conflict of Interest\}.*?(?=\\section)", "", body, flags=re.DOTALL)
    body = re.sub(r"\\section\*\{Funding\}.*?(?=\\section)", "", body, flags=re.DOTALL)
    body = re.sub(r"\\section\*\{Author Contributions\}.*?(?=\\section|\Z)", "", body, flags=re.DOTALL)
    body = re.sub(r"\\section\*\{Data Availability\}.*?(?=\\section|\Z)", "", body, flags=re.DOTALL)

    # 3. Convert numbered sections to unnumbered (PLOS style)
    body = re.sub(r"\\section\{(.+?)\}", r"\\section*{\1}", body)
    body = re.sub(r"\\subsection\{(.+?)\}", r"\\subsection*{\1}", body)
    body = re.sub(r"\\subsubsection\{(.+?)\}", r"\\subsubsection*{\1}", body)

    # 4. Section renaming: "Methods" → "Materials and methods" (PLOS standard)
    body = re.sub(r"\\section\*\{Methods\}", r"\\section*{Materials and methods}", body)

    # 5. PLOS uses "Fig" not "Figure" for citations (but figure ENVIRONMENTS still use "Figure")
    body = re.sub(r"Figure~\\ref", r"Fig~\\ref", body)
    body = re.sub(r"Figure\\ ?ref", r"Fig\\ref", body)
    body = re.sub(r"\\Figref", r"\\Figref", body)  # placeholder, no change

    # 5b. Convert \S\ref{p1:sec:*} cross-refs to plain text labels (PLOS doesn't use § for sections)
    section_label_map = {
        r"p1:sec:methods:circularity-audit": "Methods (Circularity Audit)",
        r"p1:sec:methods:reproducibility": "Methods (Reproducibility)",
        r"p1:sec:discussion:limitations": "Discussion (Limitations)",
        r"p1:sec:discussion:confounders": "Discussion (Confounder Sensitivity)",
        r"p1:sec:discussion:internal-vs-external": "Discussion (Internal vs External Calibration)",
        r"p1:sec:discussion": "Discussion",
        r"p1:sec:results:tabular": "Results (Tabular Models)",
        r"p1:sec:results:ablation": "Results (Feature Ablation)",
        r"p1:sec:results:pdonly": "Results (Training-Label Confound)",
        r"p1:sec:results": "Results",
        r"p1:sec:methods": "Methods",
        r"p1:sec:introduction": "Introduction",
        r"p1:sec:related": "Related Work",
        r"p1:sec:conclusion": "Conclusion",
        r"p1:sec:supp:hbs": "Supporting Information",
        r"p1:sec:supp:abstention": "Supporting Information",
        r"p1:sec:supp:stagingflow": "Supporting Information",
        r"p1:sec:supp:shap": "Supporting Information",
        r"p1:sec:supp:fullordinal-confusion": "Supporting Information",
        r"p1:sec:supp:confounders": "Supporting Information",
        r"p1:sec:supp:r3-q7": "Supporting Information",
        r"p1:sec:supp": "Supporting Information",
        r"p1:sec:data-availability": "REPRODUCIBILITY\\_PACKAGE.md",
    }
    for label, replacement in section_label_map.items():
        # Matches \S\ref{label}, \S~\ref{label}, ~\S\ref{label}, \ref{label}
        body = re.sub(r"\\S\s*\\ref\{" + re.escape(label) + r"\}", "the " + replacement, body)
        body = re.sub(r"\\S~\\ref\{" + re.escape(label) + r"\}", "the " + replacement, body)
        body = re.sub(r"~\\ref\{" + re.escape(label) + r"\}", " " + replacement, body)
        body = re.sub(r"\\ref\{" + re.escape(label) + r"\}", replacement, body)

    # 6. Strip ALL TikZ blocks (PLOS wants figures uploaded separately as image files).
    # This includes the architecture diagram, CONSORT flow, and any inline TikZ.
    body = re.sub(
        r"\\begin\{tikzpicture\}[\s\S]*?\\end\{tikzpicture\}",
        r"% [TikZ figure removed for PLOS submission — upload as separate image file]",
        body,
    )
    # Also strip \usetikzlibrary lines if any leaked in
    body = re.sub(r"\\usetikzlibrary\{[^}]*\}", "", body)
    # Strip empty figure environments left over (figure with only the comment)
    body = re.sub(
        r"\\begin\{figure\*?\}\[[^\]]*\]\s*(?:%[^\n]*\n)*\s*% \[TikZ figure removed[^\n]*\n\s*\\caption",
        r"\\begin{figure}[!h]\n\\caption",
        body,
    )

    # 7. Replace \includegraphics inside figure environments (PLOS wants no graphics)
    # Per PLOS: figures should be uploaded separately from manuscript file.
    body = re.sub(
        r"\\centerline\{\\includegraphics\[[^\]]*\]\{[^}]+\}\}",
        "",
        body,
    )
    body = re.sub(
        r"\\includegraphics\[[^\]]*\]\{[^}]+\}",
        "",
        body,
    )

    # 8. The CONSORT-style flow diagram block (lines 80-117 area in archive) is also
    # mostly TikZ. Strip the block but keep caption.
    # Already covered by TikZ regex above.

    # 9. Single-file requirement — embed the bibliography directly
    # Already in thebibliography format
    bib_content = re.sub(
        r".*?\\begin\{thebibliography\}", r"\\begin{thebibliography}", bib, flags=re.DOTALL, count=1
    )

    # 10. Build final document
    final = PLOS_PREAMBLE + body + PLOS_POSTAMBLE + bib_content + "\n\n\\end{document}\n"

    # Clean up multiple consecutive blank lines
    final = re.sub(r"\n{4,}", "\n\n\n", final)

    out = PLOS_DIR / "main.tex"
    out.write_text(final)
    print(f"Wrote {out}")
    print(f"  Total lines: {final.count(chr(10))}")
    print(f"  Total bytes: {len(final)}")


if __name__ == "__main__":
    main()
