/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <filesystem>
#include <memory>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <string>

#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/hnsw.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "common.cuh"

#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib> // for exit

int cagra_build_search_ace(raft::device_resources const& dev_resources)
{
  using namespace cuvs::neighbors;

  int64_t topk      = 10;

  // HNSW index parameters
  hnsw::index_params index_params;
  index_params.m = 24;
  index_params.ef_construction = 120;
  index_params.hierarchy = hnsw::HnswHierarchy::GPU;

  // ACE index parameters
  auto ace_params = hnsw::graph_build_params::ace_params();
//   // Set the number of partitions. Small values might improve recall but potentially degrade
//   // performance and increase memory usage. Partitions should not be too small to prevent issues in
//   // KNN graph construction. 100k - 5M vectors per partition is recommended depending on the
//   // available host and GPU memory. The partition size is on average 2 * (n_rows / npartitions) *
//   // dim * sizeof(T). 2 is because of the core and augmented vectors. Please account for imbalance
//   // in the partition sizes (up to 3x in our tests).
  ace_params.npartitions = 10;
//   // Set the index quality for the ACE build. Bigger values increase the index quality. At some
//   // point, increasing this will no longer improve the quality.
//   ace_params.ef_construction = 120;
//   // Set the directory to store the ACE build artifacts. This should be the fastest disk in the
//   // system and hold enough space for twice the dataset, final graph, and label mapping.
ace_params.build_dir = "/tmp/ace_build";
//   // Set whether to use disk-based storage for ACE build. When true, enables disk-based operations
//   // for memory-efficient graph construction. If not set, the index will be built in memory if the
//   // graph fits in host and GPU memory, and on disk otherwise.
  ace_params.use_disk  = true;
 index_params.graph_build_params = ace_params;

  // Open dataset in big-ann-benchmarks binary format.
  int fd = open("openai_5M/base.5M.fbin", O_RDONLY);
   if (fd == -1) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }
  uint32_t shape[2];
  ssize_t bytesRead = read(fd, shape, 8);
  if (bytesRead != 8) {
        perror("Error reading shape");
        close(fd);
        return EXIT_FAILURE;
    }
  size_t data_size = shape[0] * static_cast<size_t>(shape[1]);
  std::cout<< "Dataset size " << data_size << std::endl;
  size_t header_size = sizeof(shape);
  size_t file_size = data_size * sizeof(float) + header_size;
  float *dataset_ptr = (float*) mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  std::cout << "shape [" << shape[0] <<", " << shape[1]<<"]"<<std::endl;
  if (dataset_ptr == MAP_FAILED) {
        perror("Error mmapping the file");
        close(fd);
        return EXIT_FAILURE;
   } 
  uint32_t n_rows = shape[0];
  n_rows = 1000000;
  auto dataset_host_view = raft::make_host_matrix_view<const float, int64_t, raft::row_major>(dataset_ptr + header_size, n_rows, shape[1]);

  std::cout << "Building CAGRA index (search graph)" << std::endl;
  auto index = hnsw::build(dev_resources, index_params, dataset_host_view);

  hnsw::serialize(dev_resources, "hnsw_index.bin", *index);

  // For disk-based indices, the HNSW index file path can be obtained via file_path()
  std::string hnsw_index_path = index->file_path();
  std::cout << "HNSW index file location: " << hnsw_index_path << std::endl;


hnsw::index<float>* hnsw_index_raw = nullptr;
  hnsw::deserialize(
    dev_resources, index_params, hnsw_index_path, index->dim(), index->metric(), &hnsw_index_raw);
  std::unique_ptr<hnsw::index<float>> hnsw_index_deserialized(hnsw_index_raw);


    // HNSW search requires host matrices
  size_t n_queries = 10;
  auto queries_host = raft::make_host_matrix<float, int64_t>(n_queries, dataset_host_view.extent(1));

  raft::copy(queries_host.data_handle(),
             dataset_host_view.data_handle(),
             queries_host.size(),
             raft::resource::get_cuda_stream(dev_resources));
  raft::resource::sync_stream(dev_resources);

  // HNSW search outputs uint64_t indices
  auto indices_hnsw_host   = raft::make_host_matrix<uint64_t, int64_t>(n_queries, topk);
  auto distances_hnsw_host = raft::make_host_matrix<float, int64_t>(n_queries, topk);

  hnsw::search_params hnsw_search_params;
  hnsw_search_params.ef          = std::max(200, static_cast<int>(topk) * 2);
  hnsw_search_params.num_threads = 1;

  std::cout << "Searching HNSW index." << std::endl;
  hnsw::search(dev_resources,
               hnsw_search_params,
               *hnsw_index_deserialized,
               queries_host.view(),
               indices_hnsw_host.view(),
               distances_hnsw_host.view());

  for (int query_id = 0; query_id < std::min<int>(n_queries, 10); query_id++) {
    std::cout << "Query " << query_id << " neighbor indices: ";
    raft::print_host_vector("", &indices_hnsw_host(query_id, 0), topk, std::cout);
    std::cout << "Query " << query_id << " neighbor distances: ";
    raft::print_host_vector("", &distances_hnsw_host(query_id, 0), topk, std::cout);
  }
  
  munmap(dataset_ptr, file_size);
  close(fd);
  return 0;
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  // ACE build and search example.
  cagra_build_search_ace(dev_resources);
}
