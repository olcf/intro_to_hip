#----------------------------------------

HIPCC    = hipcc
HIPFLAGS =

#----------------------------------------

matrix_transpose: matrix_transpose.o
	${HIPCC} ${HIPFLAGS} matrix_transpose.o -o matrix_transpose

matrix_transpose.o: matrix_transpose.cpp
	${HIPCC} ${HIPFLAGS} -c matrix_transpose.cpp

.PHONY: clean

clean:
	rm -f matrix_transpose *.o
