//
// Created by chefxx on 22.01.2026.

#include <helper_cuda.h>

#include "gpu_board.cuh"
#include "gpu_infa_kernels.cuh"
#include "gpu_rollout.cuh"
#include "memory_cuda.cuh"

/**
 * @brief Entry point of the rollout kernels.
 *
 * @param t_node
 * @param d_states
 * @return
 */
double rollout_gpu(const MctsNode *t_node, const mem_cuda::unique_ptr<curandState> &d_states)
{
    // -------------
    // Prepare board
    // -------------
    const Board h_board = t_node->current_board_state;
    init_gpu_const_board(h_board);
    // -----------------
    // init random seeds
    // -----------------

    rollout_kernel<<<BLOCKS_PER_GRID, THREAD_PER_BLOCK>>>(d_states.get(), t_node->turn_colour, TODO);
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    return 1.0;
}