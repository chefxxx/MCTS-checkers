//
// Created by chefxx on 14.01.2026.
//

#include <gtest/gtest.h>

#include "cpu_movegen.h"

// 1. Empty Board Test: King in the center (D4 - index 27)
TEST(KingMoveGenTest, CenterEmptyBoard)
{
    constexpr int      kingSq     = 27;
    constexpr uint64_t boardState = (1ULL << kingSq);

    const uint64_t dAttacks = diagonalKingMask(boardState, kingSq);
    const uint64_t aAttacks = antiDiagonalKingMask(boardState, kingSq);

    // Diagonal (/) expected bits: 0, 9, 18, 36, 45, 54, 63
    constexpr uint64_t expectedD =
        1ULL << 0 | 1ULL << 9 | 1ULL << 18 | 1ULL << 36 | 1ULL << 45 | 1ULL << 54 | 1ULL << 63;
    // Anti-diagonal (\) expected bits: 6, 13, 20, 34, 41, 48
    constexpr uint64_t expectedA = 1ULL << 6 | 1ULL << 13 | 1ULL << 20 | 1ULL << 34 | 1ULL << 41 | 1ULL << 48;

    EXPECT_EQ(dAttacks, expectedD);
    EXPECT_EQ(aAttacks, expectedA);
}

// 2. Blocking with Both Colored Pieces
TEST(KingMoveGenTest, BlockingBothColors)
{
    constexpr int      kingSq       = 27;           // D4
    constexpr uint64_t whiteBlocker = (1ULL << 45); // F6 (Diagonal NE)
    constexpr uint64_t blackBlocker = (1ULL << 13); // F2 (Anti-Diagonal SE)

    // King + White Blocker + Black Blocker
    constexpr uint64_t boardState = (1ULL << kingSq) | whiteBlocker | blackBlocker;

    const uint64_t dAttacks = diagonalKingMask(boardState, kingSq);
    const uint64_t aAttacks = antiDiagonalKingMask(boardState, kingSq);

    constexpr uint64_t expectedD = 1ULL << 0 | 1ULL << 9 | 1ULL << 18 | 1ULL << 36 | 1ULL << 45;
    constexpr uint64_t expectedA = 1ULL << 13 | 1ULL << 20 | 1ULL << 34 | 1ULL << 41 | 1ULL << 48;

    EXPECT_EQ(dAttacks, expectedD);
    EXPECT_EQ(aAttacks, expectedA);
}

// 3. Corner Cases: A1 (0) and H8 (63)
TEST(KingMoveGenTest, CornerCases)
{
    // Bottom Left (A1)
    constexpr uint64_t a1Board = (1ULL << 0);
    const uint64_t     d0      = diagonalKingMask(a1Board, 0);
    const uint64_t     a0      = antiDiagonalKingMask(a1Board, 0);

    constexpr uint64_t expectedD =
        1ULL << 9 | 1ULL << 18 | 1ULL << 27 | 1ULL << 36 | 1ULL << 45 | 1ULL << 54 | 1ULL << 63;

    constexpr uint64_t expectedA = 0ULL;

    EXPECT_EQ(d0, expectedD);
    EXPECT_EQ(a0, expectedA);
}

TEST(KingsAttackMaskTest, EmptyBoardNoAttacks)
{
    constexpr uint64_t kingSq     = 27; // D4
    constexpr uint64_t kings      = (1ULL << kingSq);
    constexpr uint64_t boardState = kings;
    constexpr uint64_t opponents  = 0ULL;

    const uint64_t result = getKingsAttackMask(kings, boardState, opponents);
    EXPECT_EQ(result, 0ULL);
}

TEST(KingsAttackMaskTest, SimpleCapture)
{
    constexpr int      k_idx      = 18; // C3
    constexpr int      v_idx      = 27; // D4
    constexpr uint64_t kings      = (1ULL << k_idx);
    constexpr uint64_t opponents  = (1ULL << v_idx);
    constexpr uint64_t boardState = kings | opponents; // E5 is empty

    const uint64_t result = getKingsAttackMask(kings, boardState, opponents);
    EXPECT_EQ(result, kings);
}

TEST(KingsAttackMaskTest, FlyingKingLongCapture)
{
    constexpr int      k_idx      = 0;  // A1
    constexpr int      v_idx      = 45; // F6
    constexpr uint64_t kings      = (1ULL << k_idx);
    constexpr uint64_t opponents  = (1ULL << v_idx);
    constexpr uint64_t boardState = kings | opponents; // Square 54 (G7) is empty

    const uint64_t result = getKingsAttackMask(kings, boardState, opponents);
    EXPECT_EQ(result, kings);
}

TEST(KingsAttackMaskTest, BlockedLandingSquare)
{
    constexpr int      k_idx      = 18; // C3
    constexpr int      v_idx      = 27; // D4
    constexpr int      blocker    = 36; // E5
    constexpr uint64_t kings      = (1ULL << k_idx) | (1ULL << blocker);
    constexpr uint64_t opponents  = (1ULL << v_idx);
    constexpr uint64_t boardState = kings | opponents;

    const uint64_t result = getKingsAttackMask(kings, boardState, opponents);
    EXPECT_EQ(result, 0ULL);
}

TEST(KingsAttackMaskTest, VictimOnEdgeNoJump)
{
    constexpr int      k_idx      = 54; // G7
    constexpr int      v_idx      = 63; // H8
    constexpr uint64_t kings      = (1ULL << k_idx);
    constexpr uint64_t opponents  = (1ULL << v_idx);
    constexpr uint64_t boardState = kings | opponents;

    const uint64_t result = getKingsAttackMask(kings, boardState, opponents);
    EXPECT_EQ(result, 0ULL);
}

TEST(KingsAttackMaskTest, MultipleKingsMixed)
{
    constexpr uint64_t kingAttacker = (1ULL << 18); // C3 -> jump D4 to E5
    constexpr uint64_t kingBlocked  = (1ULL << 0);  // A1 -> victim B2 but C3 is kingAttacker
    constexpr uint64_t victim1      = (1ULL << 27); // D4
    constexpr uint64_t victim2      = (1ULL << 9);  // B2

    constexpr uint64_t kings      = kingAttacker | kingBlocked;
    constexpr uint64_t opponents  = victim1 | victim2;
    constexpr uint64_t boardState = kings | opponents;

    const uint64_t result = getKingsAttackMask(kings, boardState, opponents);

    EXPECT_TRUE(result & kingAttacker);
    EXPECT_FALSE(result & kingBlocked);
    EXPECT_EQ(__builtin_popcountll(result), 1);
}