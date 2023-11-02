#include <omp.h>
#include <algorithm>
#include <unordered_map>
#include "graph_gpu.hpp"
#include <nccl.h>
#include "graph_cuda.hpp"
#include <cuda_runtime.h>
//include host side definition of graph
#include "graph.hpp"
#include "cuda_wrapper.hpp"

#include <numeric>
#ifdef USE_PAR_EXEC_POLICY
#include <execution>
#endif

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__,__LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess)
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

GraphGPU::GraphGPU(Graph* graph, const int& nbatches, const int& part_on_device, const int& part_on_batch, const int& edgebal) : 
graph_(graph), nbatches_(nbatches), part_on_device_(part_on_device), part_on_batch_(part_on_batch), 
NV_(0), NE_(0), maxOrder_(0), mass_(0)
{
    //printf("Starting GraphGPU\n");
    NV_ = graph_->get_num_vertices();
    NE_ = graph_->get_num_edges();

    unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight);       
    indicesHost_     = graph_->get_index_ranges();
    edgeWeightsHost_ = graph_->get_edge_weights();
    edgesHost_       = graph_->get_edges();

    
    if(edgebal){
        degree_based_edge_device_partition();
    }  
    else{
        logical_partition_devices();
        //determine_edge_device_partition();
        //for(int i=0;i<NGPU;i++){
        //    printf("GPU %d: %ld - %ld\n",i,vertex_per_device_host_[i],vertex_per_device_host_[i+1]);
        //}
    }
    //degree_based_edge_device_partition();


    
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaSetDevice(id);

        for(unsigned i = 0; i < 6; ++i)
            CudaCall(cudaStreamCreate(&cuStreams[id][i]));

        e0_[id] = 0; e1_[id] = 0;
        w0_[id] = 0; w1_[id] = 0;

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];

        

        vertex_per_batch_[id] = new GraphElem [nbatches+1];
        vertex_per_batch_partition_[id].resize(nbatches);

        CudaDeviceSynchronize();
    }



    if(edgebal==1){
        //printf("Balancing Edges in Batches\n");
        degree_based_edge_batch_partition();
    }
    else{
        //printf("Logical Batching\n");
        logical_partition_batches();
        //partition_graph_edge_batch();
        /*
        for(int g=0;g<NGPU;g++){
        printf("GPU VERT RANGES\n");
            for(int i=0;i<nbatches_;i++){
                printf("%ld - %ld : %ld - %ld\n",vertex_per_batch_[g][i],vertex_per_batch_[g][i+1],indicesHost_[vertex_per_batch_[g][i]],indicesHost_[vertex_per_batch_[g][i+1]]);
            }
        }*/
    }
    CudaDeviceSynchronize();
     #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaSetDevice(id);

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];
        printf("Device %d with NV: %ld NE: %ld\n",id,nv,ne);

        GraphElem maxnv = -1;
        GraphElem maxne = -1;
        for(unsigned i = 1; i < nbatches+1; i++){
            GraphElem newnv = (vertex_per_batch_[id][i] - vertex_per_batch_[id][i-1])+1;
            GraphElem newne = indicesHost_[vertex_per_batch_[id][i]] - indicesHost_[(vertex_per_batch_[id][i-1])];
            //printf("id: %d newnv: %ld newne: %ld\n",id,newnv,newne);
            if(newnv > maxnv)
                maxnv = newnv;
            if(newne > maxne)
                maxne = newne;
    }
    maxne++;
    maxnv++;

    printf("id: %d MAXNV: %ld MAXNE: %ld\n",id,maxnv,maxne);
    CudaMallocHost(indices_[id],       sizeof(GraphElem)   * (maxnv+1));
    long long numWarps = MAX_GRIDDIM*(BLOCKDIM02/WARPSIZE);
    //warpWorkHost = new GraphElem[numWarps];
    //printf("Alloc Managed\n");
    cudaMallocManaged(&warpWork[id],sizeof(GraphElem)*numWarps);
    //printf("Done Alloc\n");
    CudaMemset(warpWork[id],0,sizeof(GraphElem)*numWarps);
    //CudaMalloc(vertexWeights_[id], sizeof(GraphWeight) * maxnv);

    cudaMallocManaged(&mate_[id], sizeof(GraphElem) * NV_);
    //CudaMalloc(matePtr_[id], sizeof(GraphElem*) * NGPU);
    CudaMalloc(partners_[id], sizeof(GraphElem) * NV_);
    //CudaMalloc(partnersPtr_[id], sizeof(GraphElem*) * NGPU);
    //CudaMalloc(ws_[id], sizeof(GraphWeight) * maxnv);


    CudaMemset(mate_[id],-1,sizeof(GraphElem) * NV_);
    CudaMemset(partners_[id],-1,sizeof(GraphElem) * NV_);
    //CudaMemset(ws_[id],-1.0,sizeof(GraphWeight) * maxnv);


    CudaMallocHost(vertex_per_device_[id], sizeof(GraphElem)*(NGPU+1));
    CudaMallocHost(finishFlag[id], sizeof(char));

    CudaMallocHost(vertex_per_batch_device_[id], sizeof(GraphElem) * (nbatches+1));
    CudaMemcpyUVA(vertex_per_batch_device_[id], vertex_per_batch_[id], sizeof(GraphElem) * (nbatches+1));
    //CudaMemcpyAsyncHtoD(indices_[id], indicesHost_+v_base_[id], sizeof(GraphElem)*(maxnv+1), 0);
    CudaMemcpyUVA(vertex_per_device_[id], vertex_per_device_host_, sizeof(GraphElem)*(NGPU+1));


    //ne_per_partition_[id] = determine_optimal_edges_per_partition(nv, ne, unit_size);
    //GraphElem ne_per_partition = ne_per_partition_[id];
    //determine_optimal_vertex_partition(indicesHost_, nv, ne, ne_per_partition, vertex_partition_[id], v_base_[id]);

    CudaMallocHost(edges_[id],          sizeof(GraphElem)*(maxne+1));
    CudaMallocHost(edgeWeights_[id],    sizeof(GraphWeight)*(maxne+1));


    CudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


    CudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    }
    /*
    for(int i = 0; i < NGPU; ++i)
    {
        CudaSetDevice(i);
        for(int j = 0; j < NGPU; ++j)
        {
           if(i != j)
                CudaCall(cudaDeviceEnablePeerAccess(j, 0));
        }
        CudaMemcpyUVA(matePtr_[i], mate_, sizeof(GraphElem*)*NGPU);
        CudaMemcpyUVA(partnersPtr_[i], partners_, sizeof(GraphElem*)*NGPU);

    }*/
}

GraphGPU::~GraphGPU()
{
    for(int g = 0; g < NGPU; ++g)
    {
        CudaSetDevice(g);
        
        cudaDeviceDisablePeerAccess(g);

        for(unsigned i = 0; i < 4; ++i)
            CudaCall(cudaStreamDestroy(cuStreams[g][i]));
        free(mate_host_[g]);
        CudaFree(edges_[g]);
        CudaFree(edgeWeights_[g]);
        //CudaFree(commIdKeys_[g]);
        CudaFree(indices_[g]);
        CudaFree(mate_[g]);
        //CudaFree(matePtr_[g]);
        CudaFree(partners_[g]);
        //CudaFree(partnersPtr_[g]);
        //CudaFree(vertexWeights_[g]);
        //CudaFree(commIds_[g]);
        //CudaFree(commIdsPtr_[g]);
        //CudaFree(commWeights_[g]);
        //CudaFree(commWeightsPtr_[g]);
        //CudaFree(newCommIds_[g]);
        //CudaFree(localCommNums_[g]);
        //CudaFree(orderedWeights_[g]);
        CudaFree(vertex_per_device_[g]);
        //CudaFree(reducedWeights_[g]);
        //CudaFree(reducedCommIdKeys_[g]);
        //CudaFree(reducedWeights_[g]);
        delete [] vertex_per_batch_[g];
    }
    #ifdef MULTIPHASE
    free(buffer_);
    buffer_ = nullptr;
    delete [] commIdsHost_;
    delete [] vertexIdsHost_;
    delete [] vertexIdsOffsetHost_;
    free(sortedVertexIdsHost_);
    #endif    
}

void GraphGPU::determine_edge_device_partition()
{
    std::vector<GraphElem> vertex_parts;
    vertex_parts.push_back(0);

    if(!part_on_device_)
    {
        GraphElem ave_nv = NV_/NGPU;
        for(int i = 1; i < NGPU; ++i)
            vertex_parts.push_back(i*ave_nv);
        vertex_parts.push_back(NV_);
    }
    else
    {
        GraphElem start = 0;
        GraphElem ave_ne = NE_/NGPU;
        for(GraphElem i = 1; i <= NV_; ++i)
        { 
            if(indicesHost_[i]-indicesHost_[start] > ave_ne)
            {
                vertex_parts.push_back(i);
                start = i;
            }
            else if (i == NV_)
                vertex_parts.push_back(NV_);
        }

        if(vertex_parts.size() > NGPU+1)
        {
            GraphElem remain = NV_ - vertex_parts[NGPU];
            GraphElem nv = remain/NGPU;
            for(GraphElem i = 1; i < NGPU; ++i)
                vertex_parts[i] += nv;
            vertex_parts[NGPU] = NV_;
        }
        else if(vertex_parts.size() < NGPU+1)
        {
            for(int i = vertex_parts.size(); i < NGPU+1; ++i)
                vertex_parts.push_back(NV_);
        }
    }
    for(int i = 0; i <= NGPU; ++i)
        vertex_per_device_host_[i] = vertex_parts[i];
    
    for(GraphElem id = 0; id < NGPU; ++id)
    {
        v_base_[id] = vertex_parts[id+0];
        v_end_ [id] = vertex_parts[id+1];

        e_base_[id] = indicesHost_[v_base_[id]];
        e_end_[id]  = indicesHost_[v_end_[id]];

        nv_[id] = v_end_[id]-v_base_[id];
        ne_[id] = e_end_[id]-e_base_[id];
    }
}


GraphElem GraphGPU::binarySearchIdx(GraphElem arr[], GraphElem l, GraphElem r, GraphElem val) {
    GraphElem low = l, high = r - 1;
    GraphElem closestIdx = -1;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        GraphElem local_low = low + thread_id * (high - low + 1) / num_threads;
        GraphElem local_high = low + (thread_id + 1) * (high - low + 1) / num_threads - 1;

        while (local_low <= local_high) {
            GraphElem mid = local_low + (local_high - local_low) / 2;

            if (arr[mid] == val) {
                #pragma omp critical
                closestIdx = mid; // Return the index of the exact match
                break;
            } else if (arr[mid] < val) {
                local_low = mid + 1;
            } else {
                local_high = mid - 1;
            }

            // Update the closest index
            #pragma omp critical
            if (closestIdx == -1 || abs(arr[mid] - val) < abs(arr[closestIdx] - val)) {
                closestIdx = mid;
            }
        }
    }

    return closestIdx;
}

void GraphGPU::degree_based_edge_device_partition(){


    vertex_per_device_host_[0] = 0;
    vertex_per_device_host_[NGPU] = NV_;
    GraphElem v0 = 0;
    GraphElem v1 = NV_;
    for(int i=1;i<NGPU;i++){
        GraphElem valForCut = (i*((indicesHost_[v1]-indicesHost_[v0])/NGPU))+indicesHost_[v0];
            //printf("Bin Search for %ld between v0:%ld and v1:%ld\n",valForCut,indicesHost_[v0],indicesHost_[v1]);
        GraphElem cut = binarySearchIdx(indicesHost_,v0,v1,valForCut);
            //printf("GPU: %d, v0: %ld v1: %ld - cut at %ld\n",g,v0,v1,cut);
        vertex_per_device_host_[i] = cut;
    }
    for(int i=0;i<NGPU;i++){
        nv_[i] = vertex_per_device_host_[i+1]-vertex_per_device_host_[i];
        ne_[i] = indicesHost_[vertex_per_device_host_[i+1]] - indicesHost_[vertex_per_device_host_[i]];
    }
    for(int i=0;i<NGPU;i++){
        printf("GPU %d: %ld - %ld\n",i,vertex_per_device_host_[i],vertex_per_device_host_[i+1]);
    }

}


void GraphGPU::degree_based_edge_batch_partition(){
    //Filled out: vertex_per_device_host_[]
    //Need to fill out: vertex_per_batch_[g][i]
    
    for(int g=0;g<NGPU;g++){
        
        GraphElem v0 = vertex_per_device_host_[g];
        GraphElem v1 = vertex_per_device_host_[g+1];
        //printf("Starting GPU %d: v0: %ld v1: %ld\n",g,v0,v1);
        vertex_per_batch_[g][0] = v0;
        vertex_per_batch_[g][nbatches_] = v1;
        for(int b=1;b<nbatches_;b++){
            GraphElem valForCut = (b*((indicesHost_[v1]-indicesHost_[v0])/nbatches_))+indicesHost_[v0];
            //printf("Bin Search for %ld between v0:%ld and v1:%ld\n",valForCut,indicesHost_[v0],indicesHost_[v1]);
            GraphElem cut = binarySearchIdx(indicesHost_,v0,v1,valForCut);
            //printf("GPU: %d, v0: %ld v1: %ld - cut at %ld\n",g,v0,v1,cut);
            vertex_per_batch_[g][b] = cut;
        }
        /*
        printf("GPU VERT RANGES\n");
        for(int i=0;i<nbatches_;i++){
            printf("%ld - %ld : %ld - %ld\n",vertex_per_batch_[g][i],vertex_per_batch_[g][i+1],indicesHost_[vertex_per_batch_[g][i]],indicesHost_[vertex_per_batch_[g][i+1]]);
        }*/
    }


}

void GraphGPU::logical_partition_devices(){
    vertex_per_device_host_[0] = 0;
    vertex_per_device_host_[NGPU] = NV_;
    for(int i=1;i<NGPU;i++){
        vertex_per_device_host_[i] = i*(NV_/NGPU);
    }
    for(int i=0;i<NGPU;i++){
        nv_[i] = vertex_per_device_host_[i+1]-vertex_per_device_host_[i];
        ne_[i] = indicesHost_[vertex_per_device_host_[i+1]] - indicesHost_[vertex_per_device_host_[i]];
    }
}
void GraphGPU::logical_partition_batches(){
    for(int g=0;g<NGPU;g++){ 
        GraphElem v0 = vertex_per_device_host_[g];
        GraphElem v1 = vertex_per_device_host_[g+1];
        printf("Starting GPU %d: v0: %ld v1: %ld\n",g,v0,v1);
        vertex_per_batch_[g][0] = v0;
        vertex_per_batch_[g][nbatches_] = v1;
        for(int b=1;b<nbatches_;b++){
            vertex_per_batch_[g][b] = (b*((v1-v0)/nbatches_))+v0;
        }
        /*
        printf("GPU VERT RANGES\n");
        for(int i=0;i<nbatches_;i++){
            printf("%ld - %ld : %ld - %ld\n",vertex_per_batch_[g][i],vertex_per_batch_[g][i+1],indicesHost_[vertex_per_batch_[g][i]],indicesHost_[vertex_per_batch_[g][i+1]]);
        }*/
    }
}



void GraphGPU::partition_graph_edge_batch()
{ 
    for(int g= 0; g < NGPU; ++g)
    {
        GraphElem nv = nv_[g]; 
        GraphElem ne = ne_[g];
        std::vector<GraphElem> vertex_parts;
        GraphElem V0 = v_base_[g];

        if(!part_on_batch_)
        {
            vertex_parts.resize(nbatches_+1);
            GraphElem ave_nv = nv / nbatches_;
            for(int i = 0; i < nbatches_; ++i)
            {
                vertex_parts[i] = (GraphElem)i*ave_nv+V0;
                if(vertex_parts[i] > nv)
                    vertex_parts[i] = nv+V0;
            }
            vertex_parts[nbatches_] = nv+V0; 
        }
        else
        {
            GraphElem ave_ne = ne / nbatches_;
            /*if(ave_ne == 0)
            {
                std::cout << "Too many batches\n";
                exit(-1); 
            }*/

            vertex_parts.push_back(V0);

            GraphElem start = V0;
            for(GraphElem i = 1; i <= nv; ++i)
            {
                if(indicesHost_[i+V0]-indicesHost_[start] > ave_ne)
                {
                    vertex_parts.push_back(i+V0);
                    start = i+V0;
                }
                else if (i == nv)
                    vertex_parts.push_back(v_end_[g]);
            }

            if(vertex_parts.size() > nbatches_+1)
            {
                GraphElem remain = v_end_[g] - vertex_parts[nbatches_];
                GraphElem nnv = remain/nbatches_;
                for(int i = 1; i < nbatches_; ++i)
                    vertex_parts[i] += nnv;
                vertex_parts[nbatches_] = v_end_[g];
            }
            else if(vertex_parts.size() < nbatches_+1) 
            {
                for(int i = vertex_parts.size(); i < nbatches_+1; ++i)
                    vertex_parts.push_back(v_end_[g]);
            }
        }
        for(int i = 0; i < nbatches_+1; ++i)
            vertex_per_batch_[g][i] = vertex_parts[i];
  
        for(int b = 0; b < nbatches_; ++b)
        {
            GraphElem v0 = vertex_per_batch_[g][b+0];
            GraphElem v1 = vertex_per_batch_[g][b+1];
            GraphElem start = v0; 
            vertex_per_batch_partition_[g][b].push_back(v0);
            for(GraphElem i = v0+1; i <= v1; ++i)
            {
                if(indicesHost_[i]-indicesHost_[start] > ne_per_partition_[g])
                {
                    vertex_per_batch_partition_[g][b].push_back(i-1);
                    start = i-1;
                    i--;
                }  
            }
            vertex_per_batch_partition_[g][b].push_back(v1);
        }    
    }
}

//TODO: in the future, this could be moved to a new class
GraphElem GraphGPU::determine_optimal_edges_per_partition
(
    const GraphElem& nv,
    const GraphElem& ne,
    const unsigned& unit_size
)
{
    if(nv > 0)
    {
        float free_m;//,total_m,used_m;
        size_t free_t,total_t;

        CudaCall(cudaMemGetInfo(&free_t,&total_t));

        float occ_m = (uint64_t)((4*sizeof(GraphElem)+2*sizeof(GraphWeight))*nv)/1048576.0;
        free_m =(uint64_t)free_t/1048576.0 - occ_m;

        GraphElem ne_per_partition = (GraphElem)(free_m / unit_size / 8.25 * 1048576.0); //5 is the minimum, i chose 8
        //std::cout << ne_per_partition << " " << ne << std::endl;
        if(ne_per_partition < ne)
            std::cout << "!!! Graph too large !!!\n";
        #ifdef debug
        return ((ne_per_partition > ne) ? ne : ne_per_partition);
        #else 
        return ((ne_per_partition > ne) ? ne : ne_per_partition);
        //return ne/4;
        #endif
    }
    return 0;
}

void GraphGPU::determine_optimal_vertex_partition
(
    GraphElem* indices,
    const GraphElem& nv,
    const GraphElem& ne,
    const GraphElem& ne_per_partition,
    std::vector<GraphElem>& vertex_partition,
    const GraphElem& V0
)
{
    vertex_partition.push_back(V0);
    GraphElem start = indices[V0];
    GraphElem end = 0;
    for(GraphElem idx = 1; idx <= nv; ++idx)
    {
        end = indices[idx+V0];
        if(end - start > ne_per_partition)
        {
            vertex_partition.push_back(V0+idx-1);
            start = indices[V0+idx-1];
            idx--;
        }
    }
    vertex_partition.push_back(V0+nv);
}


void GraphGPU::singleton_partition()
{
    omp_set_num_threads(NGPU);
    //std::cout << NGPU << std::endl;
    #pragma omp parallel
    //for(int i = 0; i < NGPU; ++i)
    {
        //std::cout << omp_get_num_threads() << std::endl;
        int i = omp_get_thread_num() % NGPU;
        //std::cout << i << std::endl;
        GraphElem V0 = v_base_[i];
        CudaSetDevice(i);
        if(nv_[i] > 0)
            singleton_partition_cuda(commIds_[i], newCommIds_[i], commWeights_[i], vertexWeights_[i], nv_[i], V0);

        CudaDeviceSynchronize();
    }
}
/*
void GraphGPU::sum_vertex_weights(const int& host_id)
{
    GraphElem V0 = v_base_[host_id];
    for(GraphElem b = 0; b < vertex_partition_[host_id].size()-1; ++b)
    {
        GraphElem v0 = vertex_partition_[host_id][b+0];
        GraphElem v1 = vertex_partition_[host_id][b+1];


        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];

        move_weights_to_device(e0, e1, host_id);

        if(v1 > v0)
            sum_vertex_weights_cuda(vertexWeights_[host_id], edgeWeights_[host_id], indices_[host_id], v0, v1, e0, e1, V0);
    }
}

void GraphGPU::move_edges_to_device
(
    const GraphElem& e0,
    const GraphElem& e1,
    const int& host_id, 
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        if(e0 < e0_[host_id] or e1 > e1_[host_id])
        {
            e0_[host_id] = e0; e1_[host_id] = e0+ne_per_partition_[host_id];
            if(e1_[host_id] > e_end_[host_id])
                e1_[host_id] = e_end_[host_id];
            if(e1_[host_id] < e1)
                std::cout << "range error\n";
            GraphElem ne = e1_[host_id]-e0_[host_id];
            CudaMemcpyAsyncHtoD(edges_[host_id], edgesHost_+e0, sizeof(GraphElem)*ne, stream);
        }
    }
}

void GraphGPU::move_edges_to_host
(
    const GraphElem& e0,  
    const GraphElem& e1,
    const int& host_id,
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        GraphElem ne = e1-e0;
        CudaMemcpyAsyncDtoH(edgesHost_+e0, edges_[host_id], sizeof(GraphElem)*ne, stream);
    }
}

void GraphGPU::move_weights_to_device
(
    const GraphElem& e0, 
    const GraphElem& e1, 
    const int& host_id,
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        if(e0 < w0_[host_id] or e1 > w1_[host_id])
        {
            w0_[host_id] = e0; w1_[host_id] = e0+ne_per_partition_[host_id];
            if(w1_[host_id] > e_end_[host_id])
                w1_[host_id] = e_end_[host_id];
            if(w1_[host_id] < e1)
                std::cout << "range error\n";

            GraphElem ne = w1_[host_id] - w0_[host_id];
            CudaMemcpyAsyncHtoD(edgeWeights_[host_id], edgeWeightsHost_+e0, sizeof(GraphWeight)*ne, stream);
        }
    }
}

void GraphGPU::move_weights_to_host
(
    const GraphElem& e0, 
    const GraphElem& e1, 
    const int& host_id,
    cudaStream_t stream
)
{
    if(e1 > e0)
    {
        GraphElem ne = e1-e0;
        CudaMemcpyAsyncDtoH(edgeWeightsHost_+e0, edgeWeights_[host_id], sizeof(GraphWeight)*ne, stream);
    }
}

void checkNcclError(ncclResult_t result, const char* operation) {
    if (result != ncclSuccess) {
        fprintf(stderr, "NCCL Error during %s: %s\n", operation, ncclGetErrorString(result));
        exit(EXIT_FAILURE);  // You can choose how to handle the error here
    }
}
*/
//Matching Functions


double GraphGPU::single_batch_p1(int id, int threadCount){
    run_pointer_chase_p1(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
    vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,0,threadCount,cuStreams[id]);
    return 0.0;
}

double GraphGPU::multi_batch_p1_inc(int id, int threadCount){
    run_pointer_chase_p1(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
        vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,0,threadCount,cuStreams[id]);
    for(int batch_id=1;batch_id<nbatches_;batch_id++){
        this->move_batch_to_GPU(batch_id,id);
        run_pointer_chase_p1(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
            vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,batch_id,threadCount,cuStreams[id]);
    }
    return 0.0;
}

double GraphGPU::multi_batch_p1_dec(int id, int threadCount){
    run_pointer_chase_p1(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
        vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,nbatches_-1,threadCount,cuStreams[id]);
    for(int batch_id=nbatches_-2;batch_id>=0;batch_id--){
        this->move_batch_to_GPU(batch_id,id);
        run_pointer_chase_p1(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
            vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,batch_id,threadCount,cuStreams[id]);
    }
    return 0.0;
}



double GraphGPU::count_num_verts_matched(int id,int iter){
    if(iter == 0){
        printf("Num Verts Matched: %ld on iter: %d, Fraction: %f\n",0,0,0.0);
    }
    GraphElem total = 0;
    #pragma omp parallel for num_threads(omp_get_max_threads())
    for (GraphElem i = 0; i < NV_; i++) {
        if (mate_[id][i] >= 0) {
            #pragma omp atomic
            total++;
        }
    }
    printf("Num Verts Matched: %ld on iter: %d, Fraction: %f\n",total,iter+1,((float)total)/NV_);
    return total;
}



double GraphGPU::run_pointer_chase()
{
    
    omp_set_num_threads(NGPU);

    ncclUniqueId nccl_id;
    ncclComm_t comm[NGPU];
    ncclGetUniqueId(&nccl_id);

    double sp1;
    double ep1;
    double sp2;
    double ep2;
    double batchS;
    double batchE;
    double batchTotal = 0;
    double p1total = 0;
    double p2total = 0;

    double bc1S;
    double bc2S;
    double bc1E;
    double bc2E;
    double bc1T = 0;
    double bc2T = 0;

    int batchCount = 1;
    double totalTime = 0;
    double start;
    double end;
    long long numWarps = MAX_GRIDDIM*(BLOCKDIM02/WARPSIZE);
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU; 


        CudaSetDevice(id);


        ncclResult_t nccl_result;

        ncclCommInitRank(&comm[id], NGPU, nccl_id, id);
        int rank, nranks;
        ncclCommUserRank(comm[id], &rank);
        ncclCommCount(comm[id], &nranks);
        //printf("Device ID: %d Rank: %d\n",id,rank);
        int iter = 0;
        int threadCount = BLOCKDIM02;
        char* flaghost_ = new char[1];
        flaghost_[0] = '1';
        
        




        this->move_batch_to_GPU(0,id);
        start = omp_get_wtime();
        while(batchCount != 0 && iter < 10){
            
            if(nbatches_==1){
                //this->single_batch_p1(id,threadCount);
                if(id == 0){
                    sp1 = omp_get_wtime();
                }
                 run_pointer_chase_p1_work(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                    vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,0,threadCount,cuStreams[id],warpWork[id]);
                if(id == 0){
                    p1total += omp_get_wtime() - sp1;
                }
            }
            else{
                if(iter%2==0){
                    //this->multi_batch_p1_inc(id,threadCount);
                    if(id == 0){
                        sp1 = omp_get_wtime();
                    }
                    run_pointer_chase_p1_work(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                        vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,0,threadCount,cuStreams[id],warpWork[id]);
                    if(id == 0){
                        p1total += omp_get_wtime() - sp1;
                    }
                    for(int batch_id=1;batch_id<nbatches_;batch_id++){
                        
                        if(id == 0){
                            batchS = omp_get_wtime();
                        }
                        this->move_batch_to_GPU(batch_id,id);

                        if(id == 0){
                            batchTotal += omp_get_wtime() - batchS;
                            sp1 = omp_get_wtime();
                        }
                        run_pointer_chase_p1_work(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                            vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,batch_id,threadCount,cuStreams[id],warpWork[id]);
                        if(id == 0){
                            p1total += omp_get_wtime() - sp1;
                        }
                    }
                }
                else{
                    //this->multi_batch_p1_dec(id,threadCount);

                    if(id == 0){
                        sp1 = omp_get_wtime();
                    }

                    run_pointer_chase_p1_work(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                        vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,nbatches_-1,threadCount,cuStreams[id],warpWork[id]);

                    if(id == 0){
                        p1total += omp_get_wtime() - sp1;
                    }
                    
                    for(int batch_id=nbatches_-2;batch_id>=0;batch_id--){

                        if(id == 0){
                            batchS = omp_get_wtime();
                        }
                        this->move_batch_to_GPU(batch_id,id);
                        if(id == 0){
                            batchTotal += omp_get_wtime() - batchS;
                            sp1 = omp_get_wtime();
                        }

                        run_pointer_chase_p1_work(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                            vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,batch_id,threadCount,cuStreams[id],warpWork[id]);

                        if(id == 0){
                            p1total += omp_get_wtime() - sp1;
                        }
                        
                    }
                }
            }
            if(id == 0){
                p1total += omp_get_wtime() - sp1;
                bc1S = omp_get_wtime();
            }
            if(NGPU!=1){
                for(int i=0;i<NGPU;i++){
                    GraphElem offset = vertex_per_device_host_[i];
                    GraphElem count = vertex_per_device_host_[i+1] - vertex_per_device_host_[i];
                    ncclBroadcast(partners_[id]+offset,partners_[id]+offset,count,ncclInt64,i,comm[id],0);

                }
            }
            if(id == 0){
                bc1T += omp_get_wtime() - bc1S;
                sp2 = omp_get_wtime();
            }
            flaghost_[0] = '0';
            CudaMemcpyAsyncHtoD(finishFlag[id],flaghost_,sizeof(char),cuStreams[id][0]);
            batchCount = 0;
            run_pointer_chase_p2(mate_[id],partners_[id],vertex_per_device_[id],vertex_per_device_host_,finishFlag[id],id,threadCount,cuStreams[id]);
            CudaMemcpyAsyncDtoH(flaghost_,finishFlag[id],sizeof(char),cuStreams[id][0]);
            if(flaghost_[0] == '1'){
                #pragma omp atomic
                batchCount += 1;
            }
            if(id == 0){
                p2total += omp_get_wtime() - sp2;
                bc2S = omp_get_wtime();
            }
            if(NGPU!=1){
                for(int i=0;i<NGPU;i++){
                    GraphElem offset = vertex_per_device_host_[i];
                    GraphElem count = vertex_per_device_host_[i+1] - vertex_per_device_host_[i];
                    ncclBroadcast(mate_[id]+offset,mate_[id]+offset,count,ncclInt64,i,comm[id],0);
                }
            }
            if(id == 0){
                bc2T += omp_get_wtime() - bc2S;
                count_num_verts_matched(id,iter);
            }
            #pragma omp barrier
            iter++;
            
            //printf("Iter %d\n",iter);
            
        }
        if(id == 0){
            end = omp_get_wtime();
            //printf("Number of Iterations: %d\n",iter);
            
        }
        CudaDeviceSynchronize();
        
        
    }
    
    printf("WarpWork\n");
        for(int i=0;i<NGPU;i++){
            std::cout << "GPU " << i << std::endl;
            //printf("GPU %d\n",i);
            for(GraphElem j=0;j<numWarps;j++){
                std::cout << j << "," << warpWork[i][j] << std::endl;
                //printf("%ld,%ld\n",j,warpWork[i][j]);
        }
    }
    printf("P1Time: %f\n",p1total);
    printf("P2Time: %f\n",p2total);
    printf("BC1Time: %f\n",bc1T);
    printf("BC2Time: %f\n",bc2T);
    printf("Batch Transfer: %f\n",batchTotal);
    CudaDeviceSynchronize();
    totalTime += end - start;
    return totalTime;

}

/*
void GraphGPU::move_edges_to_device_UVA()
{
    #pragma omp parallel
    {
        int host_id = omp_get_num_threads() % NGPU;
        GraphElem v0 = vertex_partition_[host_id][0];
        GraphElem v1 = vertex_partition_[host_id][1];
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        if(e1 > e0)
        {
            if(e0 < e0_[host_id] or e1 > e1_[host_id])
            {
                e0_[host_id] = e0; e1_[host_id] = e0+ne_per_partition_[host_id];
                if(e1_[host_id] > e_end_[host_id])
                    e1_[host_id] = e_end_[host_id];
                if(e1_[host_id] < e1)
                    std::cout << "range error\n";
                GraphElem ne = e1_[host_id]-e0_[host_id];
                CudaMemcpyUVA(edges_[host_id], edgesHost_+e0, sizeof(GraphElem)*ne);
                CudaMemcpyUVA(edgeWeights_[host_id], edgeWeightsHost_+e0, sizeof(GraphWeight)*ne);
            }
        }
    }
}
*/
void GraphGPU::move_batch_to_GPU(int batch_id,int device_id){
        //if(device_id == 1)
        //printf("Loading batch %d into device: %d\n",batch_id,device_id);
        CudaSetDevice(device_id);

        GraphElem v0 = vertex_per_batch_[device_id][batch_id];
        GraphElem v1 = vertex_per_batch_[device_id][batch_id+1];
        GraphElem nv = v1 - v0 + 1;
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;
        //printf("V0 %ld V1 %ld nv %ld ne %ld e0 %ld e1 %ld\n",v0,v1,nv,ne,e0,e1);
        //gpuErrchk( cudaPeekAtLastError() );
        //CudaDeviceSynchronize();
        CudaMemcpyAsyncHtoD(indices_[device_id], indicesHost_ + v0, sizeof(GraphElem) * nv,cuStreams[device_id][0]);
        CudaMemcpyAsyncHtoD(edges_[device_id], edgesHost_ + e0, sizeof(GraphElem) * ne,cuStreams[device_id][1]);
        CudaMemcpyAsyncHtoD(edgeWeights_[device_id], edgeWeightsHost_ + e0, sizeof(GraphWeight) * ne,cuStreams[device_id][2]);
        //CudaMemcpyHtoD(indices_[device_id], indicesHost_ + v0, sizeof(GraphElem) * nv);
        //gpuErrchk( cudaPeekAtLastError() );
        //CudaMemcpyHtoD(edges_[device_id], edgesHost_ + e0, sizeof(GraphElem) * ne);
        //gpuErrchk( cudaPeekAtLastError() );
        //CudaMemcpyHtoD(edgeWeights_[device_id], edgeWeightsHost_ + e0, sizeof(GraphWeight) * ne);
        //gpuErrchk( cudaPeekAtLastError() );
        //CudaDeviceSynchronize();

}
/*
void GraphGPU::move_batch_from_GPU(int batch_id){
    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaSetDevice(id);
        GraphElem v0 = vertex_per_batch_[id][batch_id];
        GraphElem v1 = vertex_per_batch_[id][batch_id+1];
        GraphElem nv = v1 - v0;
        GraphElem e1 = indicesHost_[v1-1];
        GraphElem e0 = indicesHost_[v0];
        GraphElem ne = e1-e0;

        CudaMemcpyUVA(indicesHost_ + v0, indices_[id],sizeof(GraphElem) * nv);
        CudaMemcpyUVA(mate_host_ + v0, mate_[id],sizeof(GraphElem) * nv);
        CudaMemcpyUVA(partners_host_ + v0, partners_[id], sizeof(GraphElem) * nv);
        CudaMemcpyUVA(edgesHost_ + e0, edges_[id], sizeof(GraphElem) * ne);
        CudaMemcpyUVA(edgeWeightsHost_ + e0, edgeWeights_[id], sizeof(GraphElem) * ne);

    }

}
*/
