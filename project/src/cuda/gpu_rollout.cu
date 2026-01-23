//
// Created by chefxx on 22.01.2026.

#include <helper_cuda.h>

#include "../../include/gpu/gpu_board.cuh"
#include "../../include/gpu/gpu_rollout.cuh"
#include "gpu_infa_kernels.cuh"
#include "memory_cuda.cuh"

constexpr int BLOCKS_PER_GRID = 1;
constexpr int THREAD_PER_BLOCK = 1;

/**
 * @brief Entry point of the rollout kernels.
 *
 * @param t_node
 * @return
 */
double rollout_gpu(const MctsNode *t_node)
{
    // -------------
    // Prepare board
    // -------------
    const Board h_board = t_node->current_board_state;
    const auto d_board = GPU_Board(h_board);
    checkCudaErrors(cudaMemcpyToSymbol(d_initBoard, &d_board, sizeof(GPU_Board)));

    // -----------------
    // init random seeds
    // -----------------
    // TODO: move this to GameManager class definition
    const auto d_states = mem_cuda::make_unique<curandState>(BLOCKS_PER_GRID * THREAD_PER_BLOCK);

    rollout_kernel<<<1, 1>>>(d_states.get(), t_node->turn_colour);
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    return 1.0;
}