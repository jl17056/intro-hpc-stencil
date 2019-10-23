stencil: stencil.c
	icc -fast -std=c99 -Wall -qopt-report=5  $^ -o $@

