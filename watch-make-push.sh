watchexec -w main.tex -r -s SIGKILL -- bash -c \"pdflatex\ main.tex\;bibtex\ main\;git\ commit\ -amauto\;git\ push\"
