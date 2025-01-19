---Platform---
Currently Loaded Modulefiles:
  1) gcc/12.2.0               2) openmpi/4.1.4            3) cuda/12.1                4) cmake/3.28.1             5) python/miniconda24.4.0

--Build-- 
/// mmio files are for reference only, may need to rebuild mmio prior to linking

nvcc -arch=sm_70 --expt-relaxed-constexpr --expt-extended-lambda
-DSPDLOG_FMT_EXTERNAL -DFMT_HEADER_ONLY=1
-DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -DTHRUST_DISABLE_ABI_NAMESPACE
-DTHRUST_IGNORE_ABI_NAMESPACE_ERROR
-I/share/apps/openmpi/4.1.4/gcc/12.2.0/include
-I/people/ghos167/.conda/envs/cugraph-ldgpu2/include/rapids
-I/people/ghos167/.conda/envs/cugraph-ldgpu2/include/rapids/libcudacxx -isystem
/people/ghos167/.conda/envs/cugraph-ldgpu2/include -isystem
/people/ghos167/.conda/envs/cugraph-ldgpu2/targets/x86_64-linux/include
-isystem /share/apps/cuda/12.1/include -I. -std=c++17 -o mg_matching_test
mg_matching_cugraph.cu mmio.o -L/share/apps/cuda/12.1/lib
-L/people/ghos167/.conda/envs/cugraph-ldgpu2/lib
-L/share/apps/openmpi/4.1.4/gcc/12.2.0/lib -lmpi -ldl -lcudart -lcugraph -lnccl
