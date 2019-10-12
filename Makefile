stencil: stencil.c
	icc -fast -std=c99 -Wall $^ -o $@

