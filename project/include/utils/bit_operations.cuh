//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_BIT_OPERATIONS_H
#define MCTS_CHECKERS_BIT_OPERATIONS_H

#include <bit>
#include <cstddef>
#include <cuda_runtime_api.h>

constexpr size_t MIN_LSB = 1ULL;

// this func is used for setting bits
__device__ __host__ __forceinline__ void setBitAtIdx(size_t &a, const size_t idx) { a |= MIN_LSB << idx; }

// this func is used for checking bits
__device__ __host__ __forceinline__ bool checkBitAtIdx(const size_t a, const size_t idx)
{
    return a & (MIN_LSB << idx);
}

// this func toggles the bit at the given index (0 -> 1 or 1 -> 0)
__device__ __host__ __forceinline__ void flipBitAtIdx(size_t &a, const size_t idx) { a ^= (MIN_LSB << idx); }

// this func sets to 0 bit of given idx
__device__ __host__ __forceinline__ void resetBitAtIdx(size_t &a, const size_t idx) { a &= ~(MIN_LSB << idx); }

// this func returns number of 1's
__device__ __host__ __forceinline__ int popCount(const size_t a)
{
#if defined(__CUDA_ARCH__)
    return __popcll(a);
#else
    return std::popcount(a);
#endif
}

// TODO: check if indexing is the same across projects
// this func returns index of lsb index, 0-based
__device__ __host__ __forceinline__ int getLsb(const size_t a)
{
#if defined(__CUDA_ARCH__)
    return __ffsll(a) - 1; //__ffsll() is 1-based, so - 1
#else
    return std::countr_zero(a);
#endif
}

// retrieves lsb index and resets lsb
__device__ __host__ __forceinline__ int popLsb(size_t &a)
{
    const int idx = getLsb(a);
    a &= a - 1;
    return idx;
}

#endif // MCTS_CHECKERS_BIT_OPERATIONS_H
