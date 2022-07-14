#----------------------------------------

HIPCC    = hipcc
HIPFLAGS =

#----------------------------------------

matrix_multiply: matrix_multiply.o
	${HIPCC} ${HIPFLAGS} matrix_multiply.o -o matrix_multiply

matrix_multiply.o: matrix_multiply.cpp
	${HIPCC} ${HIPFLAGS} -c matrix_multiply.cpp

.PHONY: clean

clean:
	rm -f matrix_multiply *.o
