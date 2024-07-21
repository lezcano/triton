#include <cub/cub.cuh>

extern "C" __device__ float aggregateSum(float *shmem,
                                         float (&thread_data)[4]) {
  using BlockReduceT = cub::BlockReduce<float, 32>;
  auto &temp_storage = *reinterpret_cast<BlockReduceT::TempStorage *>(shmem);
  return BlockReduceT(temp_storage).Sum(thread_data);
}
