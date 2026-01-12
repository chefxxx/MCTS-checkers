//
// Created by chefxx on 13.11.2025.
//

#include <gtest/gtest.h>

#include "cpu_movegen.h"

// white: 10, 14, 34
// black: 19, 43, 23
// white on the move
// result: 10, 34
TEST(CPU_movegenTest, return_upRightAttacks)
{
    constexpr auto white_mask  = (1ULL << 10) | (1ULL << 14) | (1ULL << 34);
    constexpr auto black_mask  = (1ULL << 19) | (1ULL << 23) | (1ULL << 43);
    constexpr auto result_mask = (1ULL << 10) | (1ULL << 34);
    const auto     test_case   = getPawnsAttackMask(white_mask, black_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 18
// black: 25
// landing: 32 (empty)
// result: 18 (the jumper)
TEST(CPU_movegenTest, returns_upLeftAttacks)
{
    constexpr uint64_t white_mask  = (1ULL << 18);
    constexpr uint64_t black_mask  = (1ULL << 25);
    constexpr uint64_t result_mask = (1ULL << 18);

    const auto test_case = getPawnsAttackMask(white_mask, black_mask);
    EXPECT_EQ(result_mask, test_case);
}

// black: 42
// white: 35
// landing: 28 (empty)
// result: 42 (the jumper)
TEST(CPU_movegenTest, returns_downRightAttacks)
{
    constexpr uint64_t black_mask  = (1ULL << 42);
    constexpr uint64_t white_mask  = (1ULL << 35);
    constexpr uint64_t result_mask = (1ULL << 42);

    // We pass black_mask as t_mask because black is the "attacker" here
    const auto test_case = getPawnsAttackMask(black_mask, white_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 16 (File A)
// black: 25
// result: 0 (cannot jump left from File A)
TEST(CPU_movegenTest, does_not_wrap_fileA)
{
    constexpr uint64_t white_mask  = (1ULL << 16);
    constexpr uint64_t black_mask  = (1ULL << 25);
    constexpr uint64_t result_mask = 0ULL;

    const auto test_case = getPawnsAttackMask(black_mask, white_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 10, 14
// black: 19, 21
// landing: 28 (empty)
// result: 10, 14
TEST(CPU_movegenTest, returns_multiple_attackers)
{
    constexpr uint64_t white_mask  = (1ULL << 10) | (1ULL << 14);
    constexpr uint64_t black_mask  = (1ULL << 19) | (1ULL << 21);
    constexpr uint64_t result_mask = (1ULL << 10) | (1ULL << 14);

    const auto test_case = getPawnsAttackMask(white_mask, black_mask);
    EXPECT_EQ(result_mask, test_case);
}