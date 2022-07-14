#----------------------------------------

HIPCC    = hipcc
HIPFLAGS = -Xcompiler -fopenmp

#----------------------------------------

# Target to compile all targets -------------------------------------------
all: pageable pinned overlap

# Target to compile the original code with pageable memory ----------------
pageable: average_array_elements.o
	${HIPCC} ${HIPFLAGS} average_array_elements.o -o average_array_elements

average_array_elements.o: average_array_elements.cpp
	${HIPCC} ${HIPFLAGS} -c average_array_elements.cpp

# Target to compile the version with pinned memory ------------------------
pinned: average_array_elements_pinned.o
	${HIPCC} ${HIPFLAGS} average_array_elements_pinned.o -o average_array_elements_pinned

average_array_elements_pinned.o: average_array_elements_pinned.cpp
	${HIPCC} ${HIPFLAGS} -c average_array_elements_pinned.cpp

# Target to compile the version with pinned memory and overlap ------------
overlap: average_array_elements_overlap.o
	${HIPCC} ${HIPFLAGS} average_array_elements_overlap.o -o average_array_elements_overlap

average_array_elements_overlap.o: average_array_elements_overlap.cpp
	${HIPCC} ${HIPFLAGS} -c average_array_elements_overlap.cpp

# Clean-up targets --------------------------------------------------------
.PHONY: clean-pageable clean-pinned clean-overlap clean-all

clean-pageable:
	rm -f average_array_elements *.o

clean-pinned:
	rm -f average_array_elements_pinned *.o

clean-overlap:
	rm -f average_array_elements_overlap *.o

clean-all:
	rm -f average_array_elements average_array_elements_pinned average_array_elements_overlap *.o
