/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <cugraph/algorithms.hpp>
#include <cugraph/graph_functions.hpp>

#include <thrust/binary_search.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>

#include "mg_partition_utils.cuh"

#include <iostream>
#include <string>

extern "C" {
#include "mmio.h"
}

void initialize_mpi_and_set_device(int argc, char** argv)
{
  RAFT_MPI_TRY(MPI_Init(&argc, &argv));

  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  int num_gpus_per_node{};
  RAFT_CUDA_TRY(cudaGetDeviceCount(&num_gpus_per_node));
  RAFT_CUDA_TRY(cudaSetDevice(comm_rank % num_gpus_per_node));
}

std::unique_ptr<raft::handle_t> initialize_mg_handle()
{
  int comm_rank{};
  RAFT_MPI_TRY(MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank));

  std::shared_ptr<rmm::mr::device_memory_resource> resource =
    std::make_shared<rmm::mr::cuda_memory_resource>();
  rmm::mr::set_current_device_resource(resource.get());

  std::unique_ptr<raft::handle_t> handle =
    std::make_unique<raft::handle_t>(rmm::cuda_stream_per_thread, resource);

  raft::comms::initialize_mpi_comms(handle.get(), MPI_COMM_WORLD);
  auto& comm           = handle->get_comms();
  auto const comm_size = comm.get_size();

  auto gpu_row_comm_size = static_cast<int>(sqrt(static_cast<double>(comm_size)));
  while (comm_size % gpu_row_comm_size != 0) {
    --gpu_row_comm_size;
  }

  cugraph::partition_manager::init_subcomm(*handle, gpu_row_comm_size);

  return std::move(handle);
}

/**
 * @brief Create a graph from edge sources, destination, and optional weights.
 */
template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<cugraph::edge_property_t<
             cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
             weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
create_graph(raft::handle_t const& handle,
             std::vector<vertex_t>&& edge_srcs,
             std::vector<vertex_t>&& edge_dsts,
             std::optional<std::vector<weight_t>>&& edge_wgts,
             bool renumber,
             bool is_symmetric)
{
  size_t num_edges = edge_srcs.size();
  assert(edge_dsts.size() == num_edges);
  if (edge_wgts.has_value()) { assert((*edge_wgts).size() == num_edges); }

  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  //
  // Assign part of the edge list to each GPU. If there are N edges and P GPUs, each GPU except the
  // one with rank P-1 reads N/P edges and the GPU with rank P -1 reads (N/P + N%P) edges into GPU
  // memory.
  //

  auto start = comm_rank * (num_edges / comm_size);
  auto end   = (comm_rank + 1) * (num_edges / comm_size);
  if (comm_rank == comm_size - 1) { end = num_edges; }
  auto work_size = end - start;

  rmm::device_uvector<vertex_t> d_edge_srcs(work_size, handle.get_stream());
  rmm::device_uvector<vertex_t> d_edge_dsts(work_size, handle.get_stream());

  auto d_edge_wgts = edge_wgts.has_value() ? std::make_optional<rmm::device_uvector<weight_t>>(
                                               work_size, handle.get_stream())
                                           : std::nullopt;

  raft::update_device(d_edge_srcs.data(), edge_srcs.data() + start, work_size, handle.get_stream());
  raft::update_device(d_edge_dsts.data(), edge_dsts.data() + start, work_size, handle.get_stream());
  if (d_edge_wgts.has_value()) {
    raft::update_device(
      (*d_edge_wgts).data(), (*edge_wgts).data() + start, work_size, handle.get_stream());
  }

  //
  // In cugraph, each vertex and edge is assigned to a specific GPU using hash functions. Before
  // creating a graph from edges, we need to ensure that all edges are already assigned to the
  // proper GPU.
  //

  if (multi_gpu) {
    std::tie(d_edge_srcs, d_edge_dsts, d_edge_wgts, std::ignore, std::ignore) =
      cugraph::shuffle_external_edges<vertex_t, vertex_t, weight_t, int32_t>(handle,
                                                                             std::move(d_edge_srcs),
                                                                             std::move(d_edge_dsts),
                                                                             std::move(d_edge_wgts),
                                                                             std::nullopt,
                                                                             std::nullopt);
  }

  //
  // Create graph
  //

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);

  std::optional<cugraph::edge_property_t<decltype(graph.view()), weight_t>> edge_weights{
    std::nullopt};

  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};

  std::tie(graph, edge_weights, std::ignore, std::ignore, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_t,
                                        int32_t,
                                        store_transposed,
                                        multi_gpu>(handle,
                                                   std::nullopt,
                                                   std::move(d_edge_srcs),
                                                   std::move(d_edge_dsts),
                                                   std::move(d_edge_wgts),
                                                   std::nullopt,
                                                   std::nullopt,
                                                   cugraph::graph_properties_t{is_symmetric, false},
                                                   renumber,
                                                   true);

  auto graph_view       = graph.view();
  auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

template <typename IndexType_>
int mm_properties(FILE* f, int tg, MM_typecode* t, IndexType_* m, IndexType_* n, IndexType_* nnz)
{
  // Read matrix properties from file
  int mint, nint, nnzint;
  if (fseek(f, 0, SEEK_SET)) {
    fprintf(stderr, "Error: could not set position in file\n");
    return -1;
  }
  if (mm_read_banner(f, t)) {
    fprintf(stderr, "Error: could not read Matrix Market file banner\n");
    return -1;
  }
  if (!mm_is_matrix(*t) || !mm_is_coordinate(*t)) {
    fprintf(stderr, "Error: file does not contain matrix in coordinate format\n");
    return -1;
  }
  if (mm_read_mtx_crd_size(f, &mint, &nint, &nnzint)) {
    fprintf(stderr, "Error: could not read matrix dimensions\n");
    return -1;
  }
  if (!mm_is_pattern(*t) && !mm_is_real(*t) && !mm_is_integer(*t) && !mm_is_complex(*t)) {
    fprintf(stderr, "Error: matrix entries are not valid type\n");
    return -1;
  }
  *m   = mint;
  *n   = nint;
  *nnz = nnzint;

  // Find total number of non-zero entries
  if (tg && !mm_is_general(*t)) {
    // Non-diagonal entries should be counted twice
    *nnz *= 2;

    // Diagonal entries should not be double-counted
    int st;
    for (int i = 0; i < nnzint; ++i) {
      // Read matrix entry
      // MTX only supports int for row and col idx
      int row, col;
      double rval, ival;
      if (mm_is_pattern(*t))
        st = fscanf(f, "%d %d\n", &row, &col);
      else if (mm_is_real(*t) || mm_is_integer(*t))
        st = fscanf(f, "%d %d %lg\n", &row, &col, &rval);
      else  // Complex matrix
        st = fscanf(f, "%d %d %lg %lg\n", &row, &col, &rval, &ival);
      if (ferror(f) || (st == EOF)) {
        fprintf(stderr, "Error: error %d reading Matrix Market file (entry %d)\n", st, i + 1);
        return -1;
      }

      // Check if entry is diagonal
      if (row == col) --(*nnz);
    }
  }

  return 0;
}

/// Read Matrix Market file and convert to COO format matrix
/** Matrix Market file is assumed to be a sparse matrix in coordinate
 *  format.
 *
 *  @param f File stream for Matrix Market file.
 *  @param tg Boolean indicating whether to convert matrix to general
 *  format (from symmetric, Hermitian, or skew symmetric format).
 *  @param nnz Number of non-zero matrix entries.
 *  @param cooRowInd (Output) Row indices for COO matrix. Should have
 *  at least nnz entries.
 *  @param cooColInd (Output) Column indices for COO matrix. Should
 *  have at least nnz entries.
 *  @param cooRVal (Output) Real component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @param cooIVal (Output) Imaginary component of COO matrix
 *  entries. Should have at least nnz entries. Ignored if null
 *  pointer.
 *  @return Zero if matrix was read successfully. Otherwise non-zero.
 */
template <typename IndexType_, typename ValueType_>
int mm_to_coo(FILE* f,
              int tg,
              IndexType_ nnz,
              IndexType_* cooRowInd,
              IndexType_* cooColInd,
              ValueType_* cooRVal,
              ValueType_* cooIVal)
{
  // Read matrix properties from file
  MM_typecode t;
  int m, n, nnzOld;
  if (fseek(f, 0, SEEK_SET)) {
    fprintf(stderr, "Error: could not set position in file\n");
    return -1;
  }
  if (mm_read_banner(f, &t)) {
    fprintf(stderr, "Error: could not read Matrix Market file banner\n");
    return -1;
  }
  if (!mm_is_matrix(t) || !mm_is_coordinate(t)) {
    fprintf(stderr, "Error: file does not contain matrix in coordinate format\n");
    return -1;
  }
  if (mm_read_mtx_crd_size(f, &m, &n, &nnzOld)) {
    fprintf(stderr, "Error: could not read matrix dimensions\n");
    return -1;
  }
  if (!mm_is_pattern(t) && !mm_is_real(t) && !mm_is_integer(t) && !mm_is_complex(t)) {
    fprintf(stderr, "Error: matrix entries are not valid type\n");
    return -1;
  }

  // Add each matrix entry in file to COO format matrix
  int i;      // Entry index in Matrix Market file; can only be int in the MTX format
  int j = 0;  // Entry index in COO format matrix; can only be int in the MTX format
  for (i = 0; i < nnzOld; ++i) {
    // Read entry from file
    int row, col;
    double rval, ival;
    int st;
    if (mm_is_pattern(t)) {
      st   = fscanf(f, "%d %d\n", &row, &col);
      rval = 1.0;
      ival = 0.0;
    } else if (mm_is_real(t) || mm_is_integer(t)) {
      st   = fscanf(f, "%d %d %lg\n", &row, &col, &rval);
      ival = 0.0;
    } else  // Complex matrix
      st = fscanf(f, "%d %d %lg %lg\n", &row, &col, &rval, &ival);
    if (ferror(f) || (st == EOF)) {
      fprintf(stderr, "Error: error %d reading Matrix Market file (entry %d)\n", st, i + 1);
      return -1;
    }

    // Switch to 0-based indexing
    --row;
    --col;

    // Record entry
    cooRowInd[j] = row;
    cooColInd[j] = col;
    if (cooRVal != NULL) cooRVal[j] = rval;
    if (cooIVal != NULL) cooIVal[j] = ival;
    ++j;

    // Add symmetric complement of non-diagonal entries
    if (tg && !mm_is_general(t) && (row != col)) {
      // Modify entry value if matrix is skew symmetric or Hermitian
      if (mm_is_skew(t)) {
        rval = -rval;
        ival = -ival;
      } else if (mm_is_hermitian(t)) {
        ival = -ival;
      }

      // Record entry
      cooRowInd[j] = col;
      cooColInd[j] = row;
      if (cooRVal != NULL) cooRVal[j] = rval;
      if (cooIVal != NULL) cooIVal[j] = ival;
      ++j;
    }
  }
  return 0;
}

template <typename vertex_t, typename weight_t>
std::tuple<rmm::device_uvector<vertex_t>,
           rmm::device_uvector<vertex_t>,
           std::optional<rmm::device_uvector<weight_t>>,
           rmm::device_uvector<vertex_t>,
           bool>
read_edgelist_from_matrix_market_file(raft::handle_t const& handle,
                                      std::string const& graph_file_full_path,
                                      bool test_weighted,
                                      bool store_transposed,
                                      bool multi_gpu)
{
  MM_typecode mc{};
  vertex_t m{};
  size_t nnz{};

  FILE* file = fopen(graph_file_full_path.c_str(), "r");
  CUGRAPH_EXPECTS(file != nullptr, "fopen (%s) failure.", graph_file_full_path.c_str());

  size_t tmp_m{};
  size_t tmp_k{};
  auto mm_ret = mm_properties<size_t>(file, 1, &mc, &tmp_m, &tmp_k, &nnz);
  CUGRAPH_EXPECTS(mm_ret == 0, "could not read Matrix Market file properties.");
  m = static_cast<vertex_t>(tmp_m);
  CUGRAPH_EXPECTS(mm_is_matrix(mc) && mm_is_coordinate(mc) && !mm_is_complex(mc) && !mm_is_skew(mc),
                  "invalid Matrix Market file properties.");

  vertex_t number_of_vertices = m;
  bool is_symmetric           = mm_is_symmetric(mc);

  std::vector<vertex_t> h_rows(nnz);
  std::vector<vertex_t> h_cols(nnz);
  std::vector<weight_t> h_weights(nnz);

  mm_ret = mm_to_coo<vertex_t, weight_t>(
    file, 1, nnz, h_rows.data(), h_cols.data(), h_weights.data(), static_cast<weight_t*>(nullptr));
  CUGRAPH_EXPECTS(mm_ret == 0, "could not read matrix data");

  auto file_ret = fclose(file);
  CUGRAPH_EXPECTS(file_ret == 0, "fclose failure.");

  rmm::device_uvector<vertex_t> d_edgelist_srcs(h_rows.size(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_edgelist_dsts(h_cols.size(), handle.get_stream());
  auto d_edgelist_weights = test_weighted ? std::make_optional<rmm::device_uvector<weight_t>>(
                                              h_weights.size(), handle.get_stream())
                                          : std::nullopt;

  rmm::device_uvector<vertex_t> d_vertices(number_of_vertices, handle.get_stream());

  raft::update_device(d_edgelist_srcs.data(), h_rows.data(), h_rows.size(), handle.get_stream());
  raft::update_device(d_edgelist_dsts.data(), h_cols.data(), h_cols.size(), handle.get_stream());
  if (d_edgelist_weights) {
    raft::update_device(
      (*d_edgelist_weights).data(), h_weights.data(), h_weights.size(), handle.get_stream());
  }

  thrust::sequence(handle.get_thrust_policy(), d_vertices.begin(), d_vertices.end(), vertex_t{0});

  if (multi_gpu) {
    auto& comm                 = handle.get_comms();
    auto const comm_size       = comm.get_size();
    auto const comm_rank       = comm.get_rank();
    auto& major_comm           = handle.get_subcomm(cugraph::partition_manager::major_comm_name());
    auto const major_comm_size = major_comm.get_size();
    auto& minor_comm           = handle.get_subcomm(cugraph::partition_manager::minor_comm_name());
    auto const minor_comm_size = minor_comm.get_size();

    //auto vertex_key_func = cugraph::detail::compute_gpu_id_from_ext_vertex_t<vertex_t>{
    auto vertex_key_func = compute_gpu_id_from_ext_vertex_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size};
    d_vertices.resize(
      thrust::distance(d_vertices.begin(),
                       thrust::remove_if(handle.get_thrust_policy(),
                                         d_vertices.begin(),
                                         d_vertices.end(),
                                         [comm_rank, key_func = vertex_key_func] __device__(
                                           auto val) { return key_func(val) != comm_rank; })),
      handle.get_stream());
    d_vertices.shrink_to_fit(handle.get_stream());

    //auto edge_key_func = cugraph::detail::compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
    auto edge_key_func = compute_gpu_id_from_ext_edge_endpoints_t<vertex_t>{
      comm_size, major_comm_size, minor_comm_size};
    size_t number_of_local_edges{};
    if (d_edgelist_weights) {
      auto edge_first       = thrust::make_zip_iterator(thrust::make_tuple(
        d_edgelist_srcs.begin(), d_edgelist_dsts.begin(), (*d_edgelist_weights).begin()));
      number_of_local_edges = thrust::distance(
        edge_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + d_edgelist_srcs.size(),
          [store_transposed, comm_rank, key_func = edge_key_func] __device__(auto e) {
            auto major = thrust::get<0>(e);
            auto minor = thrust::get<1>(e);
            return store_transposed ? key_func(minor, major) != comm_rank
                                    : key_func(major, minor) != comm_rank;
          }));
    } else {
      auto edge_first = thrust::make_zip_iterator(
        thrust::make_tuple(d_edgelist_srcs.begin(), d_edgelist_dsts.begin()));
      number_of_local_edges = thrust::distance(
        edge_first,
        thrust::remove_if(
          handle.get_thrust_policy(),
          edge_first,
          edge_first + d_edgelist_srcs.size(),
          [store_transposed, comm_rank, key_func = edge_key_func] __device__(auto e) {
            auto major = thrust::get<0>(e);
            auto minor = thrust::get<1>(e);
            return store_transposed ? key_func(minor, major) != comm_rank
                                    : key_func(major, minor) != comm_rank;
          }));
    }

    d_edgelist_srcs.resize(number_of_local_edges, handle.get_stream());
    d_edgelist_srcs.shrink_to_fit(handle.get_stream());
    d_edgelist_dsts.resize(number_of_local_edges, handle.get_stream());
    d_edgelist_dsts.shrink_to_fit(handle.get_stream());
    if (d_edgelist_weights) {
      (*d_edgelist_weights).resize(number_of_local_edges, handle.get_stream());
      (*d_edgelist_weights).shrink_to_fit(handle.get_stream());
    }
  }

  return std::make_tuple(std::move(d_edgelist_srcs),
                         std::move(d_edgelist_dsts),
                         std::move(d_edgelist_weights),
                         std::move(d_vertices),
                         is_symmetric);
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
std::tuple<cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>,
           std::optional<
             cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>,
                                      weight_t>>,
           std::optional<rmm::device_uvector<vertex_t>>>
read_graph_from_matrix_market_file(raft::handle_t const& handle,
                                   std::string const& graph_file_full_path,
                                   bool test_weighted,
                                   bool renumber)
{
  auto [d_edgelist_srcs, d_edgelist_dsts, d_edgelist_weights, d_vertices, is_symmetric] =
    read_edgelist_from_matrix_market_file<vertex_t, weight_t>(
      handle, graph_file_full_path, test_weighted, store_transposed, multi_gpu);

  cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu> graph(handle);
  std::optional<
    cugraph::edge_property_t<cugraph::graph_view_t<vertex_t, edge_t, store_transposed, multi_gpu>, weight_t>>
    edge_weights{std::nullopt};
  std::optional<rmm::device_uvector<vertex_t>> renumber_map{std::nullopt};
  std::tie(graph, edge_weights, std::ignore, std::ignore, renumber_map) =
    cugraph::create_graph_from_edgelist<vertex_t,
                                        edge_t,
                                        weight_t,
                                        edge_t,
                                        int32_t,
                                        store_transposed,
                                        multi_gpu>(handle,
                                                   std::move(d_vertices),
                                                   std::move(d_edgelist_srcs),
                                                   std::move(d_edgelist_dsts),
                                                   std::move(d_edgelist_weights),
                                                   std::nullopt,
                                                   std::nullopt,
                                                   cugraph::graph_properties_t{is_symmetric, false},
                                                   renumber);

  return std::make_tuple(std::move(graph), std::move(edge_weights), std::move(renumber_map));
}

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool multi_gpu>
void run_graph_matching(
  raft::handle_t const& handle,
  cugraph::graph_view_t<vertex_t, edge_t, false, multi_gpu> const& graph_view,
  cugraph::edge_property_view_t<edge_t, weight_t const*> edge_weight_view)
{
  auto const comm_rank = handle.get_comms().get_rank();
  auto const comm_size = handle.get_comms().get_size();

  // 
  // Matching
  //
  rmm::device_uvector<vertex_t> mg_partners(0, handle.get_stream());
  weight_t mg_matching_weights;

  std::forward_as_tuple(mg_partners, mg_matching_weights) =
    cugraph::approximate_weighted_matching(handle, graph_view, edge_weight_view);

  if (comm_rank == 0)
    std::cout << "matching weight: " << mg_matching_weights << std::endl;
  
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

int main(int argc, char** argv)
{
  initialize_mpi_and_set_device(argc, argv);
  std::unique_ptr<raft::handle_t> handle = initialize_mg_handle();

  auto const comm_rank = handle->get_comms().get_rank();
  auto const comm_size = handle->get_comms().get_size();
  
  //
  // Create graph from edge source, destination and weight list
  //

  using vertex_t    = int32_t;
  using edge_t      = int32_t;
  using weight_t    = float;
 
  std::string graph_file_full_path;
  constexpr bool multi_gpu        = true;
  constexpr bool store_transposed = false;
  bool renumber                   = true;  // must be true for multi-GPU applications
  bool test_weighted              = true;

  if (argc == 1)
  {
    std::cout << "./a.out <path-to-matrix-market-file>\n";
    RAFT_MPI_TRY(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
  }
  else
    graph_file_full_path = argv[1];

  /*
  bool is_symmetric = true;

  std::vector<vertex_t> edge_srcs = {0, 1, 1, 2, 2, 2, 3, 4, 1, 3, 4, 0, 1, 3, 5, 5};
  std::vector<vertex_t> edge_dsts = {1, 3, 4, 0, 1, 3, 5, 5, 0, 1, 1, 2, 2, 2, 3, 4};
  std::vector<weight_t> edge_wgts = {
    0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f, 0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};

  auto [graph, edge_weights, renumber_map] =
    create_graph<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
      *handle,
      std::move(edge_srcs),
      std::move(edge_dsts),
      std::move(std::make_optional(edge_wgts)),
      renumber,
      is_symmetric);
  */
  if (comm_rank == 0)
    std::cout << "Reading Matrix Market file: " << graph_file_full_path << std::endl;

  double rt0 = MPI_Wtime();
  
  auto [graph, edge_weights, renumber_map] =
    read_graph_from_matrix_market_file<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(*handle, 
        graph_file_full_path,
        test_weighted,
        renumber);

  double rt1 = MPI_Wtime();
  double rtt = (rt1 - rt0), tot_rtt = 0.0;

  RAFT_MPI_TRY(MPI_Reduce(&rtt, &tot_rtt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));
  
  if (comm_rank == 0)
    std::cout << "Average time (in s) for reading the graph: " << graph_file_full_path << " : " << (double)(tot_rtt/(double)comm_size) << std::endl; 

  // Non-owning view of the graph object
  auto graph_view = graph.view();

  // Non-owning of the edge_weights object
  //auto edge_weight_view = edge_weights ? std::make_optional((*edge_weights).view()) : std::nullopt;
  auto edge_weight_view = (*edge_weights).view();
  
  if (comm_rank == 0)
    std::cout << "Running Matrix Market file: " << graph_file_full_path << std::endl;

  //
  // Run example graph algorithms
  //

  double t0 = MPI_Wtime();

  run_graph_matching<vertex_t, edge_t, weight_t, multi_gpu>(
    *handle, graph_view, edge_weight_view);

  double t1 = MPI_Wtime();
  double tt = (t1 - t0), tot_tt = 0.0;

  RAFT_MPI_TRY(MPI_Reduce(&tt, &tot_tt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD));

  if (comm_rank == 0)
    std::cout << "Average execution time (in s) on " << comm_size << " processes/GPUs: " << (double)(tot_tt/(double)comm_size) << std::endl; 

  handle.reset();

  RAFT_MPI_TRY(MPI_Finalize());
}
