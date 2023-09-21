CC  = gcc-9
CPP = g++-9

TARGET = driverForChordal

CFLAGS = -O3 -fopenmp -std=c99 
CPPFLAGS = -O3 -fopenmp 

LDFLAGS =  $(CPPFLAGS)
SNAP_HOME = $(HOME)/snap-0.3
INCLUDES = $(SNAP_HOME)/include
LIBS = $(SNAP_HOME)/lib -lsnap -lm

OBJECTS = timer.o RngStream.o generateRMAT.o \
          utilityFunctions.o parseInputFiles.o \
	  maximalChordalSubgraph.o writeGraphDimacsFormat.o \
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
