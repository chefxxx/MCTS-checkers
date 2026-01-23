//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_INFRA_KERNELS_CUH
#define GPU_INFRA_KERNELS_CUH

#include "gpu_board.cuh"
#include "constants.h"

__constant__ inline GPU_Board d_initBoard;

__global__ void rollout_kernel(Colour t_startingTurn)
{
    if (d_initBoard.pawns[white] == 5614165 && d_initBoard.pawns[black] == 12273903276444876800) {
        printf("Starting board pawns match!\n");
    }
}

#endif
