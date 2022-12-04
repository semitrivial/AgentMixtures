#watchexec -w main.tex -r -s SIGKILL -- bash -c \"pdflatex\ main.tex\;bibtex\ main\"
watchexec -w author_response.tex -r -s SIGKILL -- bash -c \"pdflatex\ author_response.tex\;bibtex\ main\"
