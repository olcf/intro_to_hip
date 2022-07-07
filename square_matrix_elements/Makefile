#----------------------------------------

HIPCC    = hipcc
HIPFLAGS =

#----------------------------------------

square_matrix_elements: square_matrix_elements.o
	${HIPCC} ${HIPFLAGS} square_matrix_elements.o -o square_matrix_elements

square_matrix_elements.o: square_matrix_elements.cpp
	${HIPCC} ${HIPFLAGS} -c square_matrix_elements.cpp

.PHONY: clean

clean:
	rm -f square_matrix_elements *.o
