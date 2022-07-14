#----------------------------------------

HIPCC    = hipcc
HIPFLAGS =

#----------------------------------------

square_array_elements: square_array_elements.o
	${HIPCC} ${HIPFLAGS} square_array_elements.o -o square_array_elements

square_array_elements.o: square_array_elements.cpp
	${HIPCC} ${HIPFLAGS} -c square_array_elements.cpp

.PHONY: clean

clean:
	rm -f square_array_elements *.o
