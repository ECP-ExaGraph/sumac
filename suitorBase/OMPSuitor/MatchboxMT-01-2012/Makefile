
CC  = gcc
CPP = g++
CFLAGS   = -g -Ofast -fopenmp -std=c99 
CPPFLAGS = -g -Ofast -fopenmp -std=c++11

LDFLAGS =  $(CPPFLAGS)

SNAP_HOME = ./snap-0.3
LIBS = $(SNAP_HOME)/lib -lsnap -lm

INCLUDES = $(SNAP_HOME)/include/
#IOFOLDER = ./InputsOutput
#COFOLDER = ./BasicCommunitiesDetection
#UTFOLDER = ./Utility
#LIBS     = -lm

#TARGET = driverForColoringHaosInput
#TARGET = driverForColoring
#TARGET = driverForColoringGeneralGraphs
#TARGET = driverForMatching
#TARGET = driverForMatchingNewD
TARGET = driverMatchingApproxAlgorithmsNew
#TARGET = driverForMatchingGeneralGraphs
#TARGET = driverForChordal
#TARGET = driverForBFS
#TARGET = driverForMatchingSuitorRmat

#CFLAGS = -fast -xW -openmp -std=c99                        #Intel Opt
###CFLAGS = -g -debug inline_debug_info -O3 -openmp -std=c99 #Intel Debug


OBJECTS = timer.o RngStream.o generateRMAT.o \
          utilityFunctions.o parseInputFiles.o \
          coloringDistanceOneParallelIterative.o coloringDistanceOneSerial.o \
          coloringDistanceOneParallelIterativeNewD.o \
          matchingHalfApproxDominating.o matchingHalfApproxDominatingSerial.o \
          matchingHalfApproxDominatingNewD.o \
          matchingHalfApproxDominatingSuitorNewD.o matchingHalfApproxDominatingSuitorSerialNewD.o \
	  maximalChordalSubgraph.o writeGraphDimacsFormat.o \
	  breadthFirstSearch.o netAlignExactMatch.o matchingWeightAugmentingPathsNewD.o\
          $(TARGET).o

all: $(TARGET) message

$(TARGET): $(OBJECTS)
	$(CPP) $(LDFLAGS) -o $@ $(OBJECTS) -L$(LIBS)

.c.o:
	$(CC) $(CFLAGS) -c $< -I$(INCLUDES) -o $@

.cpp.o:
	$(CPP) $(CPPFLAGS) -c $< -I$(INCLUDES) -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

message:
	echo "Executable: $(TARGET) has been created"
