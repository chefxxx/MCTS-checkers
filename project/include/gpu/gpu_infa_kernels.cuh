//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_INFRA_KERNELS_CUH
#define GPU_INFRA_KERNELS_CUH

#include <curand_kernel.h>

#include "constants.h"
#include "gpu_movegen.cuh"
#include "gpu_board.cuh"

// -------------------------------------
// Constant memory variables definitions
// -------------------------------------

__constant__ inline GPU_Board d_initBoard;

__global__ void rollout_kernel(curandState* t_stateBuff, const Colour t_startingTurn)
{
    // Initialize the kernel's variables
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const GPU_Board tmp_board = d_initBoard;
    curandState local = t_stateBuff[tid];

    // perform a random game
    const auto move = generate_random_move(&local, tmp_board, t_startingTurn);

    // save results
    t_stateBuff[tid] = local;
}

#endif
