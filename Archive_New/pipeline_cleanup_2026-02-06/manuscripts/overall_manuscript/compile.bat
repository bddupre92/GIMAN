@echo off
REM Windows compilation script for GIMAN comprehensive manuscript

echo ========================================
echo GIMAN Comprehensive Manuscript Compile
echo ========================================
echo.

REM First pass - generate aux files
echo [1/4] First pdflatex pass...
pdflatex -interaction=nonstopmode main.tex

REM BibTeX for references
echo [2/4] Running BibTeX...
bibtex main

REM Second pass - resolve references
echo [3/4] Second pdflatex pass...
pdflatex -interaction=nonstopmode main.tex

REM Third pass - finalize
echo [4/4] Third pdflatex pass...
pdflatex -interaction=nonstopmode main.tex

REM Cleanup auxiliary files
echo.
echo Cleaning up auxiliary files...
del main.aux main.log main.out main.bbl main.blg main.toc 2>nul

REM Check if PDF was created
if exist main.pdf (
    echo.
    echo ========================================
    echo SUCCESS! PDF generated: main.pdf
    echo ========================================
    echo.
    echo Opening PDF...
    start main.pdf
) else (
    echo.
    echo ========================================
    echo ERROR: PDF generation failed!
    echo ========================================
    echo Check the LaTeX log for errors.
)

pause
