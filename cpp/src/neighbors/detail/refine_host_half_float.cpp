/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "refine_host-inl.hpp"

#include <cuda_fp16.h>

#define instantiate_cuvs_neighbors_refine(IdxT, DataT, DistanceT, ExtentsT)             \
  template void cuvs::neighbors::detail::refine_host<IdxT, DataT, DistanceT, ExtentsT>( \
    raft::host_matrix_view<const DataT, ExtentsT, raft::row_major> dataset,             \
    raft::host_matrix_view<const DataT, ExtentsT, raft::row_major> queries,             \
    raft::host_matrix_view<const IdxT, ExtentsT, raft::row_major> neighbor_candidates,  \
    raft::host_matrix_view<IdxT, ExtentsT, raft::row_major> indices,                    \
    raft::host_matrix_view<DistanceT, ExtentsT, raft::row_major> distances,             \
    cuvs::distance::DistanceType metric);

instantiate_cuvs_neighbors_refine(int64_t, half, float, int64_t);

#undef instantiate_cuvs_neighbors_refine
