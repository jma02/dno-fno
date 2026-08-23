# Manuscript

Compile from this directory with:

    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex

The manuscript source is split across `sections/`.  The numerical results and
discussion distinguish matched ablations from bundled or post hoc evidence,
and distinguish the historical v9 dataset used by the reported checkpoints
from the completed, source-authenticated parameterized replacement dataset,
which has not yet been used to train a reported model.
