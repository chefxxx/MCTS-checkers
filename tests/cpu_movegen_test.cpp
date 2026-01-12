//
// Created by chefxx on 13.11.2025.
//

#include <gtest/gtest.h>

#include "cpu_movegen.h"

// white: 10, 14, 34
// black: 19, 43, 23
// white on the move
// result: 10, 34
TEST(CPU_movegenTest, getPawnsAttackMask_returns_upRightAttacks)
{
    constexpr auto white_mask  = (1ULL << 10) | (1ULL << 14) | (1ULL << 34);
    constexpr auto black_mask  = (1ULL << 19) | (1ULL << 23) | (1ULL << 43);
    constexpr auto result_mask = (1ULL << 10) | (1ULL << 34);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto     test_case   = getPawnsAttackMask(white_mask, black_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 18
// black: 25
// landing: 32 (empty)
// result: 18 (the jumper)
TEST(CPU_movegenTest, getPawnsAttackMask_returns_upLeftAttacks)
{
    constexpr uint64_t white_mask  = (1ULL << 18);
    constexpr uint64_t black_mask  = (1ULL << 25);
    constexpr uint64_t result_mask = (1ULL << 18);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsAttackMask(white_mask, black_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// black: 42
// white: 35
// landing: 28 (empty)
// result: 42 (the jumper)
TEST(CPU_movegenTest, getPawnsAttackMask_returns_downRightAttacks)
{
    constexpr uint64_t black_mask  = (1ULL << 42);
    constexpr uint64_t white_mask  = (1ULL << 35);
    constexpr uint64_t result_mask = (1ULL << 42);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    // We pass black_mask as t_mask because black is the "attacker" here
    const auto test_case = getPawnsAttackMask(black_mask, white_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 16 (File A)
// black: 25
// result: 0 (cannot jump left from File A)
TEST(CPU_movegenTest, getPawnsAttackMask_does_not_wrap_fileA)
{
    constexpr uint64_t white_mask  = (1ULL << 16);
    constexpr uint64_t black_mask  = (1ULL << 25);
    constexpr uint64_t result_mask = 0ULL;
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsAttackMask(black_mask, white_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 10, 14
// black: 19, 21
// landing: 28 (empty)
// result: 10, 14
TEST(CPU_movegenTest, getPawnsAttackMask_returns_multiple_attackers)
{
    constexpr uint64_t white_mask  = (1ULL << 10) | (1ULL << 14);
    constexpr uint64_t black_mask  = (1ULL << 19) | (1ULL << 21);
    constexpr uint64_t result_mask = (1ULL << 10) | (1ULL << 14);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsAttackMask(white_mask, black_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 19, 26, 28
// pawn 19 is blocked by 26 and 28,
// so the only pawns to mover are 26 and 28
TEST(CPU_movegenTest, getPawnsMovesMask_withBlockedWhitePawn)
{
    constexpr uint64_t white_mask  = (1ULL << 19) | (1ULL << 26) | (1ULL << 28);
    constexpr uint64_t black_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 26) | (1ULL << 28);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsMovesMask(white_mask, empty_mask, white);
    ASSERT_EQ(test_case, result_mask);
}

// black: 35
// white: 0
// Result: 35 should be able to move Down-Left (26) or Down-Right (28).
TEST(CPU_movegenTest, getPawnsMovesMask_blackMovesDown)
{
    constexpr uint64_t black_mask  = (1ULL << 35);
    constexpr uint64_t white_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 35);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsMovesMask(black_mask, empty_mask, black);
    ASSERT_EQ(test_case, result_mask);
}

// white: 15, 22
// Result: 15 cannot move, but 22 can
// The NOT_FILE_H mask should prevent 15 from moving to 24
TEST(CPU_movegenTest, getPawnsMovesMask_preventWrapFileH)
{
    constexpr uint64_t white_mask  = (1ULL << 15) | (1ULL << 22);
    constexpr uint64_t black_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 22);
    constexpr auto empty_mask = ~(white_mask | black_mask);

    // We check if it correctly identifies 15 as a not mover
    // without allowing it to "teleport" across the board.
    const auto test_case = getPawnsMovesMask(white_mask, empty_mask, white);
    ASSERT_EQ(test_case, result_mask);
}

// black: 44, 35, 37
// Result: 35, 37
TEST(CPU_movegenTest, getPawnsMovesMask_fULLyBlocked)
{
    constexpr uint64_t black_mask  = (1ULL << 44) | (1ULL << 35) | (1ULL << 37);
    constexpr uint64_t white_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 35) | (1ULL << 37);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsMovesMask(black_mask, empty_mask, black);
    ASSERT_EQ(test_case, result_mask);
}

// white: 27
// empty: all squares except 27
// expected: moves from 27 to 34 and 27 to 36
TEST(CPU_movegenTest, createOnePawnsMove_whiteCenter)
{
    constexpr int start_idx = 27;
    constexpr uint64_t white_mask = (1ULL << start_idx);
    constexpr uint64_t empty_mask = ~white_mask;

    std::vector<Move> moves;
    createOnePawnMoves(moves, start_idx, empty_mask, white);

    ASSERT_EQ(moves.size(), 2);
    // Move to 34 (Up-Left: 27+7) and 36 (Up-Right: 27+9)
    EXPECT_TRUE(moves[0].to_mask == 1ULL << 34);
    EXPECT_TRUE(moves[1].to_mask == 1ULL << 36);
}

// white: 18, 20
// expected: 4 total moves (2 from each piece)
TEST(CPU_movegenTest, createAllPawnMoves_multiplePieces)
{
    constexpr uint64_t white_mask = (1ULL << 18) | (1ULL << 20);
    constexpr uint64_t empty_mask = ~white_mask;

    // We assume getPawnsMovesMask correctly identifies both 18 and 20 as movers
    constexpr uint64_t movers_mask = white_mask;

    std::vector<Move> all_moves;
    createAllPawnsMoves(all_moves, movers_mask, empty_mask, white);

    // Each piece has 2 diagonal forward moves
    EXPECT_EQ(all_moves.size(), 4);
    EXPECT_TRUE(all_moves[0].to_mask == 1ULL << 25 && all_moves[0].from_mask == 1ULL << 18);
    EXPECT_TRUE(all_moves[1].to_mask == 1ULL << 27 && all_moves[1].from_mask == 1ULL << 18);
    EXPECT_TRUE(all_moves[2].to_mask == 1ULL << 27 && all_moves[2].from_mask == 1ULL << 20);
    EXPECT_TRUE(all_moves[3].to_mask == 1ULL << 29 && all_moves[3].from_mask == 1ULL << 20);
}