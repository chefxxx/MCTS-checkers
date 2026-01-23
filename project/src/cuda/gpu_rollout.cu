//
// Created by chefxx on 22.01.2026.

#include <helper_cuda.h>

#include "gpu_rollout.cuh"
#include "gpu_board.cuh"
#include "gpu_infa_kernels.cuh"

/**
 * @brief Entry point of the rollout kernels.
 *
 * @param t_node
 * @return
 */
double rollout_gpu(const MctsNode *t_node)
{
    const Board h_board = t_node->current_board_state;
    const auto d_board = GPU_Board(h_board);
    checkCudaErrors(cudaMemcpyToSymbol(d_initBoard, &d_board, sizeof(GPU_Board)));
    rollout_kernel<<<1, 1>>>(t_node->turn_colour);
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    return 1.0;
}