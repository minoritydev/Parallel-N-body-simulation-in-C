CFLAGS	+= -w
CFLAGS	+= -O3
CFLAGS	+= -g2

nbody-seq: nbody-seq.c
	gcc $(CFLAGS) -o nbody-seq nbody-seq.c -lm
	mpicc $(CFLAGS) -o nbody-par nbody-par.c -lm
	
clean:
	rm -f *.o nbody-seq *~ *core
