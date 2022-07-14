#----------------------------------------

HIPCC    = hipcc
HIPFLAGS =

#----------------------------------------

matrix_addition: matrix_addition.o
	${HIPCC} ${HIPFLAGS} matrix_addition.o -o matrix_addition

matrix_addition.o: matrix_addition.cpp
	${HIPCC} ${HIPFLAGS} -c matrix_addition.cpp

.PHONY: clean

clean:
	rm -f matrix_addition *.o
