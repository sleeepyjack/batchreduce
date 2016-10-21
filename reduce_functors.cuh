#ifndef REDUCE_FUNCTORS_CUH
#define REDUCE_FUNCTORS_CUH

/////////////////////////////////////////////////
//  Author: Daniel Jünger                      //
//  Github: github.com/sleeepyjack/ratchreduce //
//  Data  : 21. Oct. 2016                      //
/////////////////////////////////////////////////

#include "cuda_helpers.cuh"

template <typename value_t>
struct sum_op_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_t operator() (value_t & base, value_t x) const
    {
        return base + x;
    }
};

template <typename value_t>
struct max_op_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_t operator() (value_t & base, value_t x) const
    {
        return (x > base) ? x : base;
    }
};

template <typename value_t>
struct min_op_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_t operator() (value_t & base, value_t x) const
    {
        return (x < base) ? x : base;
    }
};


#endif /* REDUCE_FUNCTORS_CUH */
