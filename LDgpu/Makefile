NVCC:=nvcc
CC:=g++


SM=70
graph_ft_load=0
check=0
debug=0


NCCL_PATH=
CUDA_PATH= #Add to Linker
NGPU=1


ifeq ($(debug), 1)
	CUFLAGS = -g -G -Ddebug --std=c++14 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
                  -Xcompiler -fopenmp -DMULTIPHASE -DNGPU=${NGPU} -DPRINT
	CFLAGS = -g -Ddebug -std=c++14 -Wextra -Wall -DMULTIPHASE -DNGPU=1 -DPRINT
else 
	CUFLAGS = -O3 -Xptxas -O3 --std=c++14 --gpu-architecture=compute_${SM} --gpu-code=sm_${SM},compute_${SM} \
                  -Xcompiler -O3 -Xcompiler -fopenmp -DMULTIPHASE -DPRINT -DNGPU=${NGPU}
	CFLAGS = -O3 -std=c++14 -Xcompiler -fopenmp -DMULTIPHASE -DPRINT -DNGPU=${NGPU}
	LDFLAGS = -I${NCCL_INC_PATH} -I${CUDA_INC_PATH} -L${NCCL_PATH} -L${CUDA_PATH} -lnccl -lcudart -lcuda
#LDFLAGS = -L${NCCL_PATH} -L${CUDA_PATH} -lnccl -lcudart -lcuda
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

EXE:= ldgpu${NGPU}

all: ${EXE}

graph.o: graph.cpp graph.hpp types.hpp
	${NVCC} ${CFLAGS} -c $< -o $@

graph_gpu.o: graph_gpu.cpp graph_gpu.hpp cuda_wrapper.hpp types.hpp
	${NVCC} ${CFLAGS} -c -I${CUDA_INC_PATH} -I${NCCL_INC_PATH} $< -o $@
#	${NVCC} ${CFLAGS} -c $< -o $@

main.o: main.cpp graph.hpp graph_gpu.hpp cuda_wrapper.hpp types.hpp
#	${NVCC} ${CFLAGS} -I${CUDA_INC_PATH} -c $< -o $@
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