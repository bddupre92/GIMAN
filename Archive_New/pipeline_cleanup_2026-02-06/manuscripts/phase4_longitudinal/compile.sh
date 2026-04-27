#!/bin/bash
# Linux/Mac compilation script for Phase 6 manuscript
# For Nature Machine Intelligence submission

echo "============================================"
echo " Phase 6 Explainability Manuscript"
echo " Compilation Script (Linux/Mac)"
echo "============================================"
echo

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found!"
    echo "Please install LaTeX distribution:"
    echo "  - Mac: brew install mactex"
    echo "  - Linux: sudo apt-get install texlive-full"
    exit 1
fi

echo "[1/5] First pdfLaTeX pass..."
pdflatex -interaction=nonstopmode main.tex
if [ $? -ne 0 ]; then
    echo "ERROR: First pdfLaTeX pass failed!"
    exit 1
fi

echo
echo "[2/5] BibTeX pass (generating references)..."
bibtex main
if [ $? -ne 0 ]; then
    echo "WARNING: BibTeX had issues, but continuing..."
fi

echo
echo "[3/5] Second pdfLaTeX pass (resolving references)..."
pdflatex -interaction=nonstopmode main.tex

echo
echo "[4/5] Third pdfLaTeX pass (final resolution)..."
pdflatex -interaction=nonstopmode main.tex

echo
echo "[5/5] Compiling supplementary materials..."
pdflatex -interaction=nonstopmode supplementary.tex
pdflatex -interaction=nonstopmode supplementary.tex

echo
echo "============================================"
echo " Compilation Complete!"
echo "============================================"
echo
echo "Output files:"
echo "  - main.pdf              (Main manuscript)"
echo "  - supplementary.pdf     (Supplementary materials)"
echo

echo "Cleaning up auxiliary files..."
rm -f *.aux *.log *.out *.bbl *.blg *.toc

echo
echo "Opening main.pdf..."
if command -v open &> /dev/null; then
    open main.pdf  # Mac
elif command -v xdg-open &> /dev/null; then
    xdg-open main.pdf  # Linux
else
    echo "PDF viewer not found. Please open main.pdf manually."
fi
