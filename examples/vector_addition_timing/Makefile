#----------------------------------------

HIPCC    = hipcc
HIPFLAGS =

#----------------------------------------

vector_addition: vector_addition.o
	${HIPCC} ${HIPFLAGS} vector_addition.o -o vector_addition

vector_addition.o: vector_addition.cpp
	${HIPCC} ${HIPFLAGS} -c vector_addition.cpp

.PHONY: clean

clean:
	rm -f vector_addition *.o
