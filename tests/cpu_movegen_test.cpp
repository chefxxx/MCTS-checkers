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
    constexpr uint64_t white_mask  = (1ull << 19) | (1ull << 26) | (1ull << 28);
    constexpr uint64_t black_mask  = 0ull;
    constexpr uint64_t result_mask = (1ull << 26) | (1ull << 28);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsMovesMask(white_mask, empty_mask, white);
    ASSERT_EQ(test_case, result_mask);
}

// black: 35
// white: 0
// Result: 35 should be able to move Down-Left (26) or Down-Right (28).
TEST(CPU_movegenTest, getPawnsMovesMask_blackMovesDown)
{
    constexpr uint64_t black_mask  = (1ull << 35);
    constexpr uint64_t white_mask  = 0ull;
    constexpr uint64_t result_mask = (1ull << 35);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsMovesMask(black_mask, empty_mask, black);
    ASSERT_EQ(test_case, result_mask);
}

// white: 15, 22
// Result: 15 cannot move, but 22 can
// The NOT_FILE_H mask should prevent 15 from moving to 24
TEST(CPU_movegenTest, getPawnsMovesMask_preventWrapFileH)
{
    constexpr uint64_t white_mask  = (1ull << 15) | (1ull << 22);
    constexpr uint64_t black_mask  = 0ull;
    constexpr uint64_t result_mask = (1ull << 22);
    constexpr auto empty_mask = ~(white_mask | black_mask);

    // We check if it correctly identifies 15 as a not mover
    // without allowing it to "teleport" across the board.
    const auto test_case = getPawnsMovesMask(white_mask, empty_mask, white);
    ASSERT_EQ(test_case, result_mask);
}

// black: 44, 35, 37
// Result: 35, 37
TEST(CPU_movegenTest, getPawnsMovesMask_fullyBlocked)
{
    constexpr uint64_t black_mask  = (1ull << 44) | (1ull << 35) | (1ull << 37);
    constexpr uint64_t white_mask  = 0ull;
    constexpr uint64_t result_mask = (1ull << 35) | (1ull << 37);
    constexpr auto empty_mask = ~(white_mask | black_mask);
    const auto test_case = getPawnsMovesMask(black_mask, empty_mask, black);
    ASSERT_EQ(test_case, result_mask);
}