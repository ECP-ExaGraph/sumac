NVCC:=nvcc
CC:=g++

SM=70
<<<<<<< HEAD
NGPU=8
=======
NGPU=1
>>>>>>> 4cd5a937fb1ec78dc2df42861ad5a11b74132dbb
graph_ft_load=0
check=0
debug=0
#NCCL_PATH=/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.18.3-1_cuda-12.0_gcc-9.4.0
NCCL_PATH=/lus/theta-fs0/software/thetagpu/nccl/nccl-v2.17.1-1_gcc-9.4.0-1ubuntu1-20.04/lib
#NCCL_PATH=/usr/lib/x86_64-linux-gnu/
ifeq ($(debug), 1)
	CUFLAGS = -g -G -Ddebug --std=c++14 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
                  -Xcompiler -fopenmp -DMULTIPHASE -DNGPU=${NGPU} -DPRINT
	CFLAGS = -g -Ddebug -std=c++14 -Wextra -Wall -DMULTIPHASE -DNGPU=1 -DPRINT
else 
	CUFLAGS = -O3 -Xptxas -O3 --std=c++14 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
                  -Xcompiler -O3 -Xcompiler -fopenmp -DMULTIPHASE -DPRINT -DNGPU=${NGPU}
	CFLAGS = -O3 -std=c++14 -Xcompiler -fopenmp -DMULTIPHASE -DPRINT -DNGPU=${NGPU}
	LDFLAGS = -L/usr/lib/cuda -L${NCCL_PATH} -lnccl -lcudart -lcuda
endif

ifeq ($(check),1)
	CUFLAGS += -DCHECK
	CFLAGS  += -DCHECK
endif

ifeq ($(graph_ft_load),1)
	CUFLAGS += -DGRAPH_FT_LOAD=4
	CFLAGS  += -DGRAPH_FT_LOAD=4
else
	CUFLAGS += -DGRAPH_FT_LOAD=2
	CFLAGS  += -DGRAPH_FT_LOAD=2
endif

ifeq ($(bit),32)
	CUFLAGS += -DUSE_32BIT
	CFLAGS  += -DUSE_32BIT
else ifeq ($(bit),64)
	CUFLAGS += -DUSE_64BIT
	CFLAGS  += -DUSE_64BIT
else
	CUFLAGS += -DUSE_64BIT
	CFLAGS  += -DUSE_64BIT
endif

EXE:= redPCMatch${NGPU}

all: ${EXE}

graph.o: graph.cpp graph.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

graph_gpu.o: graph_gpu.cpp graph_gpu.hpp cuda_wrapper.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

main.o: main.cpp graph.hpp graph_gpu.hpp cuda_wrapper.hpp types.hpp
	${NVCC} ${CFLAGS} -I/usr/local/cuda/include -c $< -o $@

graph_cuda.o: graph_cuda.cu graph_cuda.hpp graph.cpp graph.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

heap.o : heap.cpp heap.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

clustering.o: clustering.cpp clustering.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

ifeq ($(check),1)
graph_cpu.o: graph_cpu.cpp graph_cpu.hpp
	${NVCC} ${CFLAGS} -c $< -o $@
endif

ifeq ($(check),1)
${EXE}: graph_gpu.o graph_cuda.o graph.o main.o heap.o clustering.o graph_cpu.o
	${NVCC} ${CFLAGS} $^ -o $@
else
${EXE}: graph_gpu.o graph_cuda.o graph.o main.o heap.o clustering.o
	${NVCC} ${CFLAGS} $^ -o $@ ${LDFLAGS}
endif

clean:
	rm -f *.o ${EXE}