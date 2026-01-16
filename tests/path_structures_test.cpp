//
// Created by chefxx on 16.01.2026.
//

#include <gtest/gtest.h>

#include "move.h"

void ExpectVectorsEqual(const std::vector<int>& a, const std::vector<int>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "Mismatch at index " << i;
    }
}

TEST(MovePathTest, SimpleMoveNoCapture) {
    const std::vector<int> path = {10, 18};
    const LightMovePath light(path, false);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_FALSE(printer.capture);
    ExpectVectorsEqual(printer.positions, path);
}

// Test a capture move (single jump)
TEST(MovePathTest, SingleJumpCapture) {
    const std::vector<int> path = {10, 28}; // e.g., jumping over a piece
    const LightMovePath light(path, true);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_TRUE(printer.capture);
    ExpectVectorsEqual(printer.positions, path);
}

// Test a complex multi-jump (Draughts specialty)
TEST(MovePathTest, MultiJumpSequence) {
    const std::vector<int> path = {1, 10, 19, 28, 37};
    const LightMovePath light(path, true);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_TRUE(printer.capture);
    ExpectVectorsEqual(printer.positions, path);
}

// Test the boundary squares (0 and 63)
TEST(MovePathTest, BoundaryIndices) {
    const std::vector<int> path = {0, 63};
    const LightMovePath light(path, false);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_EQ(printer.positions[0], 0);
    EXPECT_EQ(printer.positions[1], 63);
}

// Test the limit of 10 squares
TEST(MovePathTest, MaxPathLength) {
    const std::vector<int> path = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const LightMovePath light(path, true);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_EQ(printer.positions.size(), 9);
    ExpectVectorsEqual(printer.positions, path);
}

// Test empty path (Sentinel or Move::None case)
TEST(MovePathTest, EmptyPath) {
    constexpr std::vector<int> path = {};
    const LightMovePath light(path, false);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_EQ(printer.positions.size(), 0);
    EXPECT_FALSE(printer.capture);
}