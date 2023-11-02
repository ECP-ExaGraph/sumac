#ifndef GRAPH_GPU_HPP_
#define GRAPH_GPU_HPP_
#include <cstring>
#include <vector>
#include <thrust/device_ptr.h>
#include "types.hpp"
#include "graph.hpp"
#include "cuda_wrapper.hpp"
#ifdef MULTIPHASE
#include "clustering.hpp"
#endif

//all indices are stored in global index
class GraphGPU
{
  private:

    Graph* graph_;

    GraphElem NV_, NE_;
    int nbatches_;
    int part_on_device_, part_on_batch_; //use edge-wise partition 1: yes 0: no

    GraphElem nv_[NGPU], ne_[NGPU], ne_per_partition_[NGPU]; //maximum number of edges to fit on gpu

    GraphElem vertex_per_device_host_[NGPU+1]; //subgraph vertex range on each device, allocated on host 
    GraphElem *vertex_per_device_[NGPU]; //subgraph vertex range on each device, allocated on devices
    GraphElem *vertex_per_batch_[NGPU];  //divide full vertex range into batches (batched update)
    GraphElem *vertex_per_batch_device_[NGPU]; 


    std::vector< std::vector<GraphElem> > vertex_per_batch_partition_[NGPU]; //divide batched vertex ranges (batched update) into maximum-edge partition

    GraphElem*   indices_[NGPU];
    GraphElem   *edges_[NGPU];
    GraphWeight *edgeWeights_[NGPU];

    GraphElem* warpWork[NGPU];
    GraphElem* warpWorkHost;


    GraphElem2 *commIdKeys_[NGPU];
    //GraphElem* indexOrders_[NGPU]; 

    
    GraphWeight* vertexWeights_[NGPU]; 



    GraphElem*  commIds_[NGPU];
    GraphElem** commIdsPtr_[NGPU];

    GraphWeight*  commWeights_[NGPU];
    GraphWeight** commWeightsPtr_[NGPU];

    GraphElem* newCommIds_[NGPU];

    GraphElem* mate_host_[NGPU];
    GraphElem* mate_[NGPU];
    //GraphElem** matePtr_[NGPU];

    GraphElem* partners_host_;
    GraphElem* partners_[NGPU];
    //GraphElem** partnersPtr_[NGPU];

    char* finishFlag[NGPU];


    GraphElem*   localCommNums_[NGPU];
    GraphWeight* orderedWeights_[NGPU];
    GraphWeight* reducedWeights_[NGPU];
    GraphElem2*  reducedCommIdKeys_[NGPU];

    GraphElem maxOrder_;
    GraphWeight mass_;

    GraphElem*   indicesHost_;
    GraphElem*   edgesHost_;
    GraphWeight* edgeWeightsHost_;

    std::vector<GraphElem> vertex_partition_[NGPU];

    cudaStream_t cuStreams[NGPU][6];

    //related to sorting
    thrust::device_ptr<GraphWeight> ordered_weights_ptr[NGPU];
    //thrust::device_ptr<GraphWeight> reduced_weights_ptr[NGPU]; //= thrust::device_pointer_cast(indexOrders_);
    thrust::device_ptr<GraphElem2>  keys_ptr[NGPU]; // = thrust::device_pointer_cast(commIdKeys_);
    thrust::device_ptr<GraphElem>   local_comm_nums_ptr[NGPU];
    thrust::device_ptr<GraphWeight> reduced_weights_ptr[NGPU];
    thrust::device_ptr<GraphElem2>  reduced_keys_ptr[NGPU];

    less_int2 comp;
    equal_int2 is_equal_int2;

    GraphElem e0_[NGPU], e1_[NGPU];  //memory position with respect to edgesHost_
    GraphElem w0_[NGPU], w1_[NGPU];  //memory position with respect to edgeWeightsHost_

    GraphElem v_base_[NGPU], v_end_[NGPU]; //first and last global indices of the vertices in a given gpu 
    GraphElem e_base_[NGPU], e_end_[NGPU]; //firt and last global indices of the edges in a give gpu

    //GraphElem maxPartitions_;

    //GraphElem ne_per_partition_cap_;
    #ifdef MULTIPHASE
    void*       buffer_;
    GraphElem*  commIdsHost_;
    GraphElem*  vertexIdsHost_;
    GraphElem*  vertexIdsOffsetHost_;
    GraphElem*  numEdgesHost_;
    GraphElem*  sortedIndicesHost_;
    GraphElem2* sortedVertexIdsHost_;
    #endif

    GraphElem determine_optimal_edges_per_partition 
    (
        const GraphElem&, 
        const GraphElem&, 
        const unsigned& size
    );

    void determine_optimal_vertex_partition
    (
        GraphElem*, 
        const GraphElem&, 
        const GraphElem&, 
        const GraphElem&, 
        std::vector<GraphElem>& partition,
        const GraphElem&
    );

    void determine_edge_device_partition();
    void partition_graph_edge_batch();


    void degree_based_edge_device_partition();
    void degree_based_edge_batch_partition();
    

    GraphElem max_order();

    void sum_vertex_weights(const int&);
    void compute_mass();

    #ifdef MULTIPHASE

    GraphElem sort_vertex_by_community_ids();
    void shuffle_edge_list();
    void compress_all_edges();
    void compress_edges();

    Clustering* clusters_;

    #endif

    void sort_edges_by_community_ids
    (
        const GraphElem& v0,   //starting global vertex index 
        const GraphElem& v1,   //ending global vertex index
        const GraphElem& e0,   //starting global edge index
        const GraphElem& e1,   //ending global edge index
        const int& host_id,
        const int& exclude_self_loops
    );

    void louvain_update
    (
        const GraphElem& v0,
        const GraphElem& v1,
        const GraphElem& e0,
        const GraphElem& e1,
        const int& host_id
    );

    void update_community_weights
    (
        const GraphElem& v0,
        const GraphElem& v1,
        const GraphElem& e0,
        const GraphElem& e1,
        const int& host_id
    );

    void update_community_ids
    (
        const GraphElem& v0,
        const GraphElem& v1,
        const GraphElem& u0,
        const GraphElem& u1,
        const int& host_id
    );

  public:
    GraphGPU (Graph* graph, const int& nbatches, const int& part_on_deivce, const int& part_on_batch, const int& edgebal);
    ~GraphGPU();

    void set_community_ids(GraphElem* commIds);
    void singleton_partition();

    void compute_community_weights(const int&);

    GraphWeight compute_modularity();

    void sort_edges_by_community_ids
    (
        const int& batch,
        const int& host_id
    );

    void louvain_update
    (
        const int& batch,
        const int& host_id
    );
    
    void move_edges_to_device
    (
        const GraphElem& e0, 
        const GraphElem& e1, 
        const int& host_id, 
        cudaStream_t stream = 0
    );

    void move_edges_to_host
    (
        const GraphElem& e0,  
        const GraphElem& e1, 
        const int& host_id, 
        cudaStream_t stream = 0
    );

    void move_weights_to_device
    (
        const GraphElem& e0, 
        const GraphElem& e1, 
        const int& host_id, 
        cudaStream_t stream = 0
    );

    void move_weights_to_host
    (
        const GraphElem& e0, 
        const GraphElem& e1, 
        const int& host_id, 
        cudaStream_t stream = 0
    );

    
    void update_community_weights
    (
        const int& batch,
        const int& host_id
    );

    void update_community_ids
    (
        const int& batch,
        const int& host_id
    );

    void restore_community();

    #ifdef MULTIPHASE
    bool aggregation();
    void dump_partition(const std::string&, GraphElem*);
    #endif

    #ifdef CHECK
    void louvain_update_host(const int&);
    void set_community_ids();
    void compute_modularity_host();
    #endif

    //Matching Functions

    GraphElem** get_indices_device();
    GraphElem* get_nv_device();
    void move_edges_to_device_UVA();
    double run_pointer_chase();
    void move_batch_to_GPU(int batch_id,int device_id);
    void move_batch_from_GPU(int batch_id);
    void set_P2P_Ptrs();
    double single_batch_p1(int id, int threadCount);
    double multi_batch_p1_inc(int id, int threadCount);
    double multi_batch_p1_dec(int id, int threadCount);
    GraphElem binarySearchIdx(GraphElem arr[], GraphElem l, GraphElem r, GraphElem val);
    void logical_partition_devices();
    void logical_partition_batches();
    double count_num_verts_matched(int id,int iter);

};

#endif