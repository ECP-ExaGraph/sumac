CXX = g++
NV = nvcc
CXXFLAGS = -g -Wall -O0 -std=c++11 -fopenmp
LINKFLAGS = -g -Wall -O0 -std=c++11 -fopenmp
NVCCFLAGS = -g -G -arch=sm_70
TARGET = main
TOCOMPILE = main.o
TOCOMPILEC = graph.cpp util.cpp rand.cpp io.cpp main.o match.o


all: $(TOCOMPILE) $(TOCOMPILEC)
	g++ -L/usr/local/cuda-11.6/lib64/ -o $(TARGET) $(TOCOMPILEC) -lcudadevrt -lcudart -lstdc++ -fopenmp 

match.o:
	$(NV) $(NVCCFLAGS) match.cu -c -o match.o
main.o:
	$(NV) $(NVCCFLAGS) main.cu -c -o main.o
.cpp.o:
	$(CXX) $(CXXFLAGS) -c $*.cpp 

clean:
	rm -f *.o $(TARGET)