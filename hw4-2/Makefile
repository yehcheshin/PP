NVFLAGS  := -std=c++11 -O3 -Xcompiler="-fopenmp" -arch=sm_61 
LDFLAGS  := -lm
EXES     := hw4-2

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw4-2: hw4-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
