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

GraphGPU::GraphGPU(Graph* graph, const int& nbatches, const int& part_on_device, const int& part_on_batch) : 
graph_(graph), nbatches_(nbatches), part_on_device_(part_on_device), part_on_batch_(part_on_batch), 
NV_(0), NE_(0), maxOrder_(0), mass_(0)
{
    NV_ = graph_->get_num_vertices();
    NE_ = graph_->get_num_edges();

    unsigned unit_size = (sizeof(GraphElem) > sizeof(GraphWeight)) ? sizeof(GraphElem) : sizeof(GraphWeight);       
    indicesHost_     = graph_->get_index_ranges();
    edgeWeightsHost_ = graph_->get_edge_weights();
    edgesHost_       = graph_->get_edges();


    determine_edge_device_partition();

    omp_set_num_threads(NGPU);
    #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaSetDevice(id);

        for(unsigned i = 0; i < 4; ++i)
            CudaCall(cudaStreamCreate(&cuStreams[id][i]));

        e0_[id] = 0; e1_[id] = 0;
        w0_[id] = 0; w1_[id] = 0;

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];

        

        vertex_per_batch_[id] = new GraphElem [nbatches+1];
        vertex_per_batch_partition_[id].resize(nbatches);

        //CudaDeviceSynchronize();
    }


    #ifdef MULTIPHASE
    buffer_ = malloc(unit_size*NE_);
    commIdsHost_ = new GraphElem [NV_];
    vertexIdsHost_ = new GraphElem [NV_];
    vertexIdsOffsetHost_ = new GraphElem [NV_];
    sortedVertexIdsHost_ = (GraphElem2*)malloc(sizeof(GraphElem2)*(NV_+1));

    clusters_ = new Clustering(NV_);
    #endif


    partition_graph_edge_batch();
     #pragma omp parallel
    {
        int id =  omp_get_thread_num() % NGPU;   
        CudaSetDevice(id);

        GraphElem nv = nv_[id];
        GraphElem ne = ne_[id];

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

    //printf("id: %d MAXNV: %ld MAXNE: %ld\n",id,maxnv,maxne);
    CudaMalloc(indices_[id],       sizeof(GraphElem)   * maxnv+1);
    //CudaMalloc(vertexWeights_[id], sizeof(GraphWeight) * maxnv);

    CudaMalloc(mate_[id], sizeof(GraphElem) * NV_);
    //CudaMalloc(matePtr_[id], sizeof(GraphElem*) * NGPU);
    CudaMalloc(partners_[id], sizeof(GraphElem) * NV_);
    //CudaMalloc(partnersPtr_[id], sizeof(GraphElem*) * NGPU);
    //CudaMalloc(ws_[id], sizeof(GraphWeight) * maxnv);


    CudaMemset(mate_[id],-1,sizeof(GraphElem) * NV_);
    CudaMemset(partners_[id],-1,sizeof(GraphElem) * NV_);
    //CudaMemset(ws_[id],-1.0,sizeof(GraphWeight) * maxnv);


    CudaMalloc(vertex_per_device_[id], sizeof(GraphElem)*(NGPU+1));
    CudaMalloc(finishFlag[id], sizeof(char));

    CudaMalloc(vertex_per_batch_device_[id], sizeof(GraphElem) * (nbatches+1));
    CudaMemcpyUVA(vertex_per_batch_device_[id], vertex_per_batch_[id], sizeof(GraphElem) * (nbatches+1));
    //CudaMemcpyAsyncHtoD(indices_[id], indicesHost_+v_base_[id], sizeof(GraphElem)*(maxnv+1), 0);
    CudaMemcpyUVA(vertex_per_device_[id], vertex_per_device_host_, sizeof(GraphElem)*(NGPU+1));


    ne_per_partition_[id] = determine_optimal_edges_per_partition(nv, ne, unit_size);
    GraphElem ne_per_partition = ne_per_partition_[id];
    determine_optimal_vertex_partition(indicesHost_, nv, ne, ne_per_partition, vertex_partition_[id], v_base_[id]);

    CudaMalloc(edges_[id],          unit_size*maxne);
    CudaMalloc(edgeWeights_[id],    unit_size*maxne);


    CudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


    CudaDeviceSynchronize();
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

//Matching Functions

void GraphGPU::run_pointer_chase()
{
    


    ncclUniqueId nccl_id;

    ncclGetUniqueId(&nccl_id);

    omp_set_num_threads(NGPU);

    int batchCount = 1;
    #pragma omp parallel
    {
        int id = omp_get_thread_num() % NGPU; 

        ncclComm_t comm;
        ncclCommInitRank(&comm, NGPU, nccl_id, id);
        int rank, nranks;
        ncclCommUserRank(comm, &rank);
        ncclCommCount(comm, &nranks);
        printf("Device ID: %d Rank: %d\n",id,rank);
        
        int iter = 0;
         
        int threadCount = BLOCKDIM02;
        char* flaghost_ = new char[1];
        
        flaghost_[0] = '1';
        while(batchCount != 0){

            for(int batch_id=0;batch_id<nbatches_;batch_id++){
                //printf("%d - Moving Batch %d to GPU\n",id,batch_id);
                this->move_batch_to_GPU(batch_id,id);
                //printf("%d - Moved Batch %d to GPU\n",id,batch_id);
                run_pointer_chase_p1(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                vertex_per_batch_device_[id], vertex_per_batch_[id], vertex_per_device_[id],id,batch_id,threadCount);
                printf("%d - Finished P1 Batch %d\n",id,batch_id);
                #pragma omp barrier
            }
            for(int i=0;i<NGPU;i++){
                GraphElem offset = vertex_per_device_host_[i];
                GraphElem count = vertex_per_device_host_[i+1] - vertex_per_device_host_[i];
                ncclBroadcast((void*)(partners_+offset),(void*)(partners_+offset),count,ncclInt64,i,comm,0);
            }



            flaghost_[0] = '0';
            CudaMemcpyUVA(finishFlag[id],flaghost_,sizeof(char));
            batchCount = 0;
            for(int batch_id=0;batch_id<nbatches_;batch_id++){
                //printf("%d - Moving Batch %d to GPU\n",id,batch_id);
                //this->move_batch_to_GPU(batch_id,id);
                //printf("%d - Moved Batch %d to GPU\n",id,batch_id);
                run_pointer_chase_p2(indices_[id],edgeWeights_[id],edges_[id],mate_[id],partners_[id],
                vertex_per_batch_device_[id],vertex_per_batch_[id],vertex_per_device_[id],finishFlag[id],id,batch_id,threadCount);
                printf("%d - Finished P2 Batch %d\n",id,batch_id);
                #pragma omp barrier
                CudaMemcpyUVA(flaghost_,finishFlag[id],sizeof(char));
                if(flaghost_[0] == '1'){
                    #pragma omp atomic
                    batchCount += 1;
                }
                for(int i=0;i<NGPU;i++){
                    GraphElem offset = vertex_per_device_host_[i];
                    GraphElem count = vertex_per_device_host_[i+1] - vertex_per_device_host_[i];
                    ncclBroadcast((void*)(mate_+offset),(void*)(mate_+offset),count,ncclInt64,i,comm,0);
                }
                //printf("flagHost: %c\n",flaghost_[0]);
            }
            iter++;
            
            if(id == 0)
                printf("Finished Iteration %d\n",iter);
            
        }
        //printf("%d - Done\n",id);
        //CudaMemcpyUVA(mate_host_[id],mate_[id],sizeof(GraphElem)*nv_[id]);
        if(id == 0 ){
            printf("Finished in %d Iterations\n",iter);
        }
        ncclCommDestroy(comm);
    }
    /*
    int totVerts = 0;
    for(int i=0;i<NGPU;i++){
        for(int j=0;j<nv_[i];j++){
            printf("Mate[%d]=%ld\n",totVerts+j,mate_host_[i][j]);
        }
        totVerts += nv_[i];
    }*/
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


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

void GraphGPU::move_batch_to_GPU(int batch_id,int device_id){
    
    /*
    for(int i=0;i<NGPU;i++){
        for(int j=0;j<DEFAULT_BATCHES+1;j++){
            printf("%d - %d : %ld\n",i,j,vertex_per_batch_[i][j]);
        }
    }*/
        //set_P2P_Ptrs();
        CudaSetDevice(device_id);
        GraphElem v0 = vertex_per_batch_[device_id][batch_id];
        GraphElem v1 = vertex_per_batch_[device_id][batch_id+1];
        GraphElem nv = v1 - v0 + 1;
        GraphElem e0 = indicesHost_[v0];
        GraphElem e1 = indicesHost_[v1];
        GraphElem ne = e1-e0;
        //printf("%d - Verts: %ld-%ld Moved: %ld-%ld\n",device_id,v0,v1,indicesHost_[v0],indicesHost_[v1-1]);
        //printf("%d - Copying nv: %ld  ne: %ld\n",device_id,nv,ne);
        CudaMemcpyUVA(indices_[device_id], indicesHost_ + v0, sizeof(GraphElem) * nv);
        //CudaMemcpyUVA(mate_[id], mate_host_ + v0, sizeof(GraphElem) * nv);
        //CudaMemcpyUVA(partners_[id], partners_host_ + v0, sizeof(GraphElem) * nv);
        CudaMemcpyUVA(edges_[device_id], edgesHost_ + e0, sizeof(GraphElem) * ne);
        CudaMemcpyUVA(edgeWeights_[device_id], edgeWeightsHost_ + e0, sizeof(GraphElem) * ne);
        //CudaMemcpyUVA(matePtr_[device_id], mate_, sizeof(GraphElem*)*NGPU);
        //CudaMemcpyUVA(partnersPtr_[device_id], partners_, sizeof(GraphElem*)*NGPU);

}

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



void GraphGPU::dump_partition(const std::string& filename, GraphElem* new_orders=nullptr)
{
    clusters_->dump_partition(filename, new_orders);
}
