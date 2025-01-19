# SuMac - Locally-Dominant Multi-GPU Implementation
This is an implementation of the locally-dominant or pointer chasing approximate maximum weight matching algorithm on single host, multi-GPU NVIDIA platforms for general graphs, titled Locally-Dominant-GPU (LD-GPU). If including any results or content of this work in any capacity, please cite the original publication [here](https://dl.acm.org/doi/10.1109/SC41406.2024.00024). If you have any concerns using the code within this project, please contact mandum@rpi.edu.

# Input Data 
LD-GPU primarily uses graphs in the Matrix Market format that have been converted to a binary edge list. We recommend retrieving test instances from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/). To obtain a binary edge list, we recommend the usage of the conversion steps outlined in the [Vite Louvain Method Implementation](https://github.com/ECP-ExaGraph/vite).

# Parameters 
Relative to the input size, the original work details our batching method to ensure that graph data fits on the given device setup. This parameter is primarily a trial-and-error point, as different hardware and input sets will require differing amounts of device memory. Runtime results may also be better or worse depending on batch counts and the overall balance of data distributed across a given hardware setup. Either way, if there are issues with running a given input set, we recommend first altering batch counts to ensure that any issues are not from device memory limits. 

# Compilation

On first run, make sure to specify in the `Makefile` within the `LDgpu` folder the variables `NCCL_Path` and `CUDA_PATH` for the relevant environment variables. Similarly, change the variable `NGPU` to modify the number of GPUs recognized for a given compilation.

For our purposes, this code was used for experimentation on up to 8 NVIDIA A100 GPUs on the following platform specifications:

| Platform  | Version |
|--|--|
|CUDA|10.1.243|
|NCCL|2.8.3.1|
|OpenMP|11.2.0|

Other versions are largely untested but may show results -- if you are interested in testing an alternative platform version and run into specific issues, please reach out.

After editing the Makefile and having the appropriate platform versions, simply run `make` to generate the executable for the specified GPU count.