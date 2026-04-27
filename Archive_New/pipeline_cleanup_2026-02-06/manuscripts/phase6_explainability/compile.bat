@echo off
REM Windows compilation script for Phase 6 manuscript
REM For Nature Machine Intelligence submission

echo ============================================
echo  Phase 6 Explainability Manuscript
echo  Compilation Script (Windows)
echo ============================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pdflatex not found!
    echo Please install MiKTeX from https://miktex.org/download
    pause
    exit /b 1
)

echo [1/5] First pdfLaTeX pass...
pdflatex -interaction=nonstopmode main.tex
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: First pdfLaTeX pass failed!
    pause
    exit /b 1
)

echo.
echo [2/5] BibTeX pass (generating references)...
bibtex main
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: BibTeX had issues, but continuing...
)

echo.
echo [3/5] Second pdfLaTeX pass (resolving references)...
pdflatex -interaction=nonstopmode main.tex

echo.
echo [4/5] Third pdfLaTeX pass (final resolution)...
pdflatex -interaction=nonstopmode main.tex

echo.
echo [5/5] Compiling supplementary materials...
pdflatex -interaction=nonstopmode supplementary.tex
pdflatex -interaction=nonstopmode supplementary.tex

echo.
echo ============================================
echo  Compilation Complete!
echo ============================================
echo.
echo Output files:
echo   - main.pdf              (Main manuscript)
echo   - supplementary.pdf     (Supplementary materials)
echo.
echo Cleaning up auxiliary files...
del /Q *.aux *.log *.out *.bbl *.blg *.toc 2>nul

echo.
echo Opening main.pdf...
start main.pdf

pause
