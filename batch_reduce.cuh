#ifndef BATCH_REDUCE_CUH
#define BATCH_REDUCE_CUH

/////////////////////////////////////////////////
//  Author: Daniel Jünger                      //
//  Github: github.com/sleeepyjack/batchreduce //
//  Data  : 21. Oct. 2016                      //
/////////////////////////////////////////////////

#include "cuda_helpers.cuh"

template <
  typename index_t,
  typename value_t,
  typename op_t
  >
GLOBALQUALIFIER void reduce_kernel(const value_t * in, index_t b, index_t n, value_t * out, op_t op)
{
  index_t warp, lane, base;
  value_t _out;
  for(index_t thid = blockIdx.x*blockDim.x + threadIdx.x; thid < n*32; thid += blockDim.x * gridDim.x)
  {
      warp = thid / 32;
      lane = thid % 32;
      base = warp * b;

      if(warp >= n) return;

      _out = 0;

      #pragma unroll 32
      for(index_t i = 0; i < b; i += 32)
      {
          if(i+lane < b)
              _out = op(_out, in[base+i+lane]);
      }

      for (index_t offset = 16; offset > 0; offset /= 2)
          _out = op(_out, __shfl_down(_out, offset));

      if(lane == 0) out[warp] = _out;
  }
}

template <
  typename index_t,
  typename value_t,
  typename op_t,
  index_t block_size = 1024,
  index_t grid_size  = 1UL << 27
  >
struct BatchReduce
{
  const op_t op = op_t();

  void operator () (value_t * in, index_t b, index_t n, value_t * out) // in: input array; b: batch size; n: num batches; out: output array
  {
    reduce_kernel<<<SDIV((n*32 < grid_size) ? n*32 : grid_size, block_size), block_size>>>(in, b, n, out, op); CUERR
  }
};

#endif /* BATCH_REDUCE_CUH */
