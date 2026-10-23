#!/bin/sh
# Renders the tikz diagrams of vignettes/articles/cuda-kernels.Rmd to SVG.
# Needs latex (with tikz and standalone) and dvisvgm; run via `make diagrams`.
set -e
cd "$(dirname "$0")"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for f in *.tex; do
  [ "$f" = preamble.tex ] && continue
  name=${f%.tex}
  latex -interaction=nonstopmode -halt-on-error -output-directory="$tmp" "$f" >/dev/null ||
    { tail -30 "$tmp/$name.log"; exit 1; }
  dvisvgm --no-fonts --exact-bbox --optimize -o "$name.svg" "$tmp/$name.dvi" 2>/dev/null
done
