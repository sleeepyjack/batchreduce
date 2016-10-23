#include "include/cudahelpers/cuda_helpers.cuh"
#include <iostream>
#include <algorithm>
#include "include/batch_reduce.cuh"
#include "include/reduce_functors.cuh"

int main()
{
  typedef uint64_t index_t;
  typedef uint32_t value_t;
  typedef sum_op_t<value_t> op_t;

  typedef BatchReduce<index_t, value_t, op_t> reduce_t;
  reduce_t reduce = reduce_t();

  constexpr index_t n = (1UL << 20); //number of batches
  constexpr index_t b = (1UL << 10); //batch size

  value_t * in = new value_t[n*b];

  std::iota(in, in+(n*b), 0);

  value_t * out = new value_t[n];

  value_t * in_d; cudaMalloc(&in_d, sizeof(value_t)*b*n); CUERR
  cudaMemcpy(in_d, in, sizeof(value_t)*b*n, H2D); CUERR
  value_t * out_d; cudaMalloc(&out_d, sizeof(value_t)*n); CUERR

  TIMERSTART(device_reduce)
  reduce(in_d, b, n, out_d);
  TIMERSTOP(device_reduce)

  cudaMemcpy(out, out_d, sizeof(value_t)*n, D2H); CUERR

  //check results
  const op_t op = op_t();
  value_t * check_out = new value_t[n];
  TIMERSTART(host_reduce)
  #pragma omp parallel for
  for(index_t i = 0; i < n; i++)
  {
    value_t _out = op_t::identity();
    for(index_t j = 0; j < b; j++)
    {
      _out = op(_out, in[i*b + j]);
    }
    check_out[i] = _out;
  }
  TIMERSTOP(host_reduce)
  for(index_t i = 0; i < n; i++)
  {
    if(out[i] != check_out[i]) printf("Error at id %llu: device:%u\thost:%u\n", i, out[i], check_out[i]);
  }

  cudaFree(in_d); CUERR
  cudaFree(out_d); CUERR
  delete[] in;
  delete[] out;
  delete[] check_out;
}
