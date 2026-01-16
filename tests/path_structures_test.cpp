//
// Created by chefxx on 16.01.2026.
//

#include <gtest/gtest.h>

#include "move.h"

void ExpectVectorsEqual(const std::vector<int> &a, const std::vector<int> &b)
{
    ASSERT_EQ(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i], b[i]) << "Mismatch at index " << i;
    }
}

TEST(MovePathTest, SimpleMoveNoCapture)
{
    const std::vector<int> path = {10, 18};
    const LightMovePath    light(path, false);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_FALSE(printer.capture);
    ExpectVectorsEqual(printer.positions, path);
}

// Test a capture move (single jump)
TEST(MovePathTest, SingleJumpCapture)
{
    const std::vector<int> path = {10, 28}; // e.g., jumping over a piece
    const LightMovePath    light(path, true);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_TRUE(printer.capture);
    ExpectVectorsEqual(printer.positions, path);
}

// Test a complex multi-jump (Draughts specialty)
TEST(MovePathTest, MultiJumpSequence)
{
    const std::vector<int> path = {1, 10, 19, 28, 37};
    const LightMovePath    light(path, true);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_TRUE(printer.capture);
    ExpectVectorsEqual(printer.positions, path);
}

// Test the boundary squares (0 and 63)
TEST(MovePathTest, BoundaryIndices)
{
    const std::vector<int> path = {0, 63};
    const LightMovePath    light(path, false);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_EQ(printer.positions[0], 0);
    EXPECT_EQ(printer.positions[1], 63);
}

// Test the limit of 10 squares
TEST(MovePathTest, MaxPathLength)
{
    const std::vector<int> path = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const LightMovePath    light(path, true);
    const PrintingMovePath printer(light.packed_path);

    EXPECT_EQ(printer.positions.size(), 9);
    ExpectVectorsEqual(printer.positions, path);
}

// Test empty path (Sentinel or Move::None case)
TEST(MovePathTest, EmptyPath)
{
    constexpr std::vector<int> path = {};
    const LightMovePath        light(path, false);
    const PrintingMovePath     printer(light.packed_path);

    EXPECT_EQ(printer.positions.size(), 0);
    EXPECT_FALSE(printer.capture);
}

TEST(MovePrintingTest, SimpleMoveFormat)
{
    // a1 (0) to b2 (9)
    const LightMovePath light({0, 9}, false);
    const auto          printer = PrintingMovePath(light.packed_path);

    EXPECT_EQ(stringMove(printer), "a1-b2");
}

// Test 2: Single Capture (Delimiter ':')
TEST(MovePrintingTest, SimpleCaptureFormat)
{
    // c3 (18) to e5 (36)
    const LightMovePath light({18, 36}, true);
    const auto          printer = PrintingMovePath(light.packed_path);

    EXPECT_EQ(stringMove(printer), "c3:e5");
}

// Test 3: Multi-Jump Sequence (Draughts style)
TEST(MovePrintingTest, MultiJumpFormat)
{
    // a1 (0) -> c3 (18) -> e5 (36)
    const LightMovePath light({0, 18, 36}, true);
    const auto          printer = PrintingMovePath(light.packed_path);

    EXPECT_EQ(stringMove(printer), "a1:c3:e5");
}

// Test 4: Board Boundaries
TEST(MovePrintingTest, BoundarySquares)
{
    const LightMovePath light({0, 63}, false);
    const auto          printer = PrintingMovePath(light.packed_path);

    EXPECT_EQ(stringMove(printer), "a1-h8");
}

// Test 5: Long Sequence (Max 9)
TEST(MovePrintingTest, MaxPathPrinting)
{
    // a1, b1, c1, d1, e1, f1, g1, h1, h2
    const LightMovePath light({0, 1, 2, 3, 4, 5, 6, 7, 56}, true);
    const auto          printer = PrintingMovePath(light.packed_path);

    EXPECT_EQ(stringMove(printer), "a1:b1:c1:d1:e1:f1:g1:h1:a8");
}