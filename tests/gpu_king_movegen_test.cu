//
// Created by chefxx on 23.01.2026.
//

#include <curand_kernel.h>
#include <gtest/gtest.h>

#include "gpu_infa_kernels.cuh"
#include "gpu_movegen.cuh"
#include "gpu_rollout.cuh"
#include "memory_cuda.cuh"

class GpuKingsMovegenTests : public ::testing::Test
{
    void SetUp() override
    {
        init_gpu_movegen_const_mem();
    }
};

TEST_F(GpuKingsMovegenTests, kingQuietSlide) {
    GPU_Board board;
    board.kings[white] = 1ULL << 7;

    constexpr std::array seeds{111, 227, 342, 444, 555, 666, 777};
    const auto d_boardPtr = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(1);
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    for (int i = 0; i < seeds.size(); ++i) {
        constexpr std::array results{14, 49, 42, 35, 28, 56, 21};
        setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), seeds[i]);
        test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), white, d_movePtr.get());

        GPU_Move h_result;
        checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));
        ASSERT_EQ(h_result.to_mask, 1ULL << results[i]);
        ASSERT_EQ(h_result.from_mask, 1ULL << 7);
    }
}

TEST_F(GpuKingsMovegenTests, kingSinglePawnTakeDownRight) {
    GPU_Board board;
    board.kings[white] = 1ULL << 49;
    board.pawns[black] = 1ULL << 14;
    const auto d_boardPtr = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(1);
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), 111);
    test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), white, d_movePtr.get());

    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_result.from_mask, 1ULL << 49);
    ASSERT_EQ(h_result.to_mask, 1ULL << 7);
    ASSERT_EQ(h_result.captures_mask, 1ULL << 14);
}

TEST_F(GpuKingsMovegenTests, kingSingleKingTakeUpLeft) {
    GPU_Board board;
    board.kings[white] = 1ULL << 14;
    board.kings[black] = 1ULL << 49;
    const auto d_boardPtr = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(1);
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), 111);
    test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), white, d_movePtr.get());

    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_result.from_mask, 1ULL << 14);
    ASSERT_EQ(h_result.to_mask, 1ULL << 56);
    ASSERT_EQ(h_result.captures_mask, 1ULL << 49);
}

TEST_F(GpuKingsMovegenTests, kingDoubleTakeUpLeftDownLeft) {
    GPU_Board board;
    board.kings[white] = 1ULL << 6;
    board.kings[black] = 1ULL << 20;
    board.pawns[black] = 1ULL << 9;
    const auto d_boardPtr = mem_cuda::allocateAndCopyGPU_FromHostObject(board);
    const auto d_statesPtr = mem_cuda::make_unique<curandState>(1);
    const auto d_movePtr = mem_cuda::make_unique<GPU_Move>();

    setup_curand_kernel<<<1, 1>>>(d_statesPtr.get(), 111);
    test_kernel<<<1, 1>>>(d_statesPtr.get(), d_boardPtr.get(), white, d_movePtr.get());

    GPU_Move h_result;
    checkCudaErrors(cudaMemcpy(&h_result, d_movePtr.get(), sizeof(GPU_Move), cudaMemcpyDeviceToHost));

    ASSERT_EQ(h_result.from_mask, 1ULL << 6);
    ASSERT_EQ(h_result.to_mask, 1ULL << 0);
    ASSERT_EQ(h_result.captures_mask, 1ULL << 20 | 1ULL << 9);
}