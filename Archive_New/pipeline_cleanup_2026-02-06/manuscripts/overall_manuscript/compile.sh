#!/bin/bash
# Unix/Mac compilation script for GIMAN comprehensive manuscript

echo "========================================"
echo "GIMAN Comprehensive Manuscript Compile"
echo "========================================"
echo ""

# First pass - generate aux files
echo "[1/4] First pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex

# BibTeX for references
echo "[2/4] Running BibTeX..."
bibtex main

# Second pass - resolve references
echo "[3/4] Second pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex

# Third pass - finalize
echo "[4/4] Third pdflatex pass..."
pdflatex -interaction=nonstopmode main.tex

# Cleanup auxiliary files
echo ""
echo "Cleaning up auxiliary files..."
rm -f main.aux main.log main.out main.bbl main.blg main.toc

# Check if PDF was created
if [ -f main.pdf ]; then
    echo ""
    echo "========================================"
    echo "SUCCESS! PDF generated: main.pdf"
    echo "========================================"
    echo ""
    
    # Try to open PDF (platform-dependent)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open main.pdf
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open main.pdf 2>/dev/null || echo "PDF generated. Please open main.pdf manually."
    fi
else
    echo ""
    echo "========================================"
    echo "ERROR: PDF generation failed!"
    echo "========================================"
    echo "Check the LaTeX log for errors."
    exit 1
fi
