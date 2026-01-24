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
 * @param t_states
 * @param t_globalScore
 * @return
 */
double rollout_gpu(const MctsNode                          *t_node,
                   const mem_cuda::unique_ptr<curandState> &t_states,
                   const mem_cuda::unique_ptr<double>      &t_globalScore)
{
    // -------------
    // Prepare board
    // -------------
    const Board h_board = t_node->current_board_state;
    init_gpu_const_board(h_board);

    // ------------
    // reset global score
    // ------------
    reset_score<<<1, 1>>>(t_globalScore.get());
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    rollout_kernel<<<BLOCKS_PER_GRID, THREAD_PER_BLOCK>>>(t_states.get(), t_node->turn_colour, t_globalScore.get());
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    // --------------
    // copy the score
    // --------------
    double score = -1.0;
    checkCudaErrors(cudaMemcpy(&score, t_globalScore.get(), sizeof(double), cudaMemcpyDeviceToHost));
    score = score / static_cast<double>(BLOCKS_PER_GRID * THREAD_PER_BLOCK);
    return score;
}