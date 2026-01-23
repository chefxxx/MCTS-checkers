//
// Created by chefxx on 23.01.2026.
//

#include <gtest/gtest.h>

#include "gpu_rollout.cuh"

TEST(GpuRolloutTests, firstTest)
{
    Board board;
    board.initStartingBoard();
    const MctsNode node{nullptr, board, white};
    const auto result = rollout_gpu(&node);
}

