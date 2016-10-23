#ifndef REDUCE_FUNCTORS_CUH
#define REDUCE_FUNCTORS_CUH

/////////////////////////////////////////////////
//  Author: Daniel Jünger                      //
//  Github: github.com/sleeepyjack/ratchreduce //
//  Data  : 21. Oct. 2016                      //
/////////////////////////////////////////////////

#include "cudahelpers/cuda_helpers.cuh"
#include <limits>

template <typename value_t>
struct sum_op_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_t operator() (value_t & base, value_t x) const
    {
        return base + x;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static const value_t identity()
    {
        return 0;
    }
};

template <typename value_t>
struct max_op_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_t operator() (value_t & base, value_t x) const
    {
        return (x > base) ? x : base;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static const value_t identity()
    {
        return std::numeric_limits<value_t>::min();
    }
};

template <typename value_t>
struct min_op_t {

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    value_t operator() (value_t & base, value_t x) const
    {
        return (x < base) ? x : base;
    }

    HOSTDEVICEQUALIFIER INLINEQUALIFIER
    static const value_t identity()
    {
        return std::numeric_limits<value_t>::max();
    }
};


#endif /* REDUCE_FUNCTORS_CUH */
