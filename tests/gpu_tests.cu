//
// Created by chefxx on 23.01.2026.
//

#include <curand_kernel.h>
#include <gtest/gtest.h>

#include "gpu_infa_kernels.cuh"
#include "gpu_movegen.cuh"
#include "gpu_rollout.cuh"
#include "memory_cuda.cuh"

constexpr int TEST_BLOCKS = 1;
constexpr int TEST_THREADS = 1;

// TEST_F(GpuRolloutTests, firstTest)
// {
//     Board board;
//     board.initStartingBoard();
//     const MctsNode node{nullptr, board, white};
//     prepare_gpu_const_mem();
//     const auto result = rollout_gpu(&node);
// }

class GpuMovegenTests : public ::testing::Test
{
    void SetUp() override
    {
        prepare_gpu_const_mem();
    }
};

TEST_F(GpuMovegenTests, whitePawnMoveUpRight)
{
    // Note:
    //
    // Even thought the gpu move gen function is random, the board
    // is set up in such a way that there is only one possible move.
    //
    constexpr auto white_mask = 1ULL << 11;
    constexpr auto black_mask = 1ULL << 18 | 1ULL << 25;
    GPU_Board board;
    board.pawns[white] = white_mask;
    board.pawns[black] = black_mask;
    const auto d_boardPtr = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(TEST_THREADS * TEST_BLOCKS);
    setup_curand_kernel<<<TEST_BLOCKS, TEST_THREADS>>>(d_statesPtr.get(), time(nullptr));
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();
    test_kernel<<<TEST_BLOCKS, TEST_THREADS>>>(d_statesPtr.get(), d_boardPtr.get(), white, d_movePtr.get());
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    // Only possible move from this position
    constexpr size_t result_from = 1ULL << 11;
    constexpr size_t result_to = 1ULL << 20;
    ASSERT_EQ(result_from, h_result.from_mask);
    ASSERT_EQ(result_to, h_result.to_mask);
}

// 1. White Pawn UP_LEFT
TEST_F(GpuMovegenTests, whitePawnMoveUpLeft) {
    GPU_Board board;
    board.pawns[white]  = 1ULL << 18;
    board.pawns[black] |= 1ULL << 27;
    board.pawns[black] |= 1ULL << 36;

    const auto d_boardPtr = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>();
    setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), 1234);
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), white, d_movePtr.get());
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    ASSERT_EQ(1ULL << 18, h_result.from_mask);
    ASSERT_EQ(1ULL << 25, h_result.to_mask);
}


TEST_F(GpuMovegenTests, blackPawnMoveDownwardsLeft) {
    GPU_Board board;
    board.pawns[black]  = 1ULL << 45;
    board.pawns[white] |= 1ULL << 38;
    board.pawns[white] |= 1ULL << 31;

    const auto d_boardPtr  = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(1);
    setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), 4321);
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), black, d_movePtr.get());
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    ASSERT_EQ(1ULL << 45, h_result.from_mask);
    ASSERT_EQ(1ULL << 36, h_result.to_mask);
}

TEST_F(GpuMovegenTests, blackPawnMoveDownwardsRight) {
    GPU_Board board;
    board.pawns[black]  = 1ULL << 45;
    board.pawns[white] |= 1ULL << 36;
    board.pawns[white] |= 1ULL << 27;

    const auto d_boardPtr  = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(1);
    setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), 4321);
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), black, d_movePtr.get());
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();
    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    ASSERT_EQ(1ULL << 45, h_result.from_mask);
    ASSERT_EQ(1ULL << 38, h_result.to_mask);
}