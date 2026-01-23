//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_INFRA_KERNELS_CUH
#define GPU_INFRA_KERNELS_CUH

#include "gpu_board.cuh"
#include "constants.h"
#include "gpu_movegen.cuh"

// -------------------------------------
// Constant memory variables definitions
// -------------------------------------

__constant__ inline GPU_Board d_initBoard;

__global__ void rollout_kernel(const Colour t_startingTurn)
{
    const GPU_Board tmp_board = d_initBoard;
    const size_t mask = generate_random_move(tmp_board, t_startingTurn);
}

#endif
