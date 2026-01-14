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
}

TEST(KingsQuietMovesTest, CenterEmptyBoard) {
    std::vector<Move> moves;
    constexpr int kingSq = 27; // D4
    constexpr uint64_t kings = (1ULL << kingSq);
    constexpr uint64_t boardState = kings; // Empty board

    createAllKingsQuietMoves(moves, kings, boardState);

    EXPECT_EQ(moves.size(), 13);
    for (size_t i = 0; i < moves.size(); ++i) {
        constexpr std::array results = {0, 6, 9, 13, 18, 20, 34, 36, 41, 45, 48, 54, 63};
        EXPECT_EQ(moves[i].from_mask, 1ULL << kingSq);
        EXPECT_EQ(moves[i].to_mask, 1ULL << results[i]);
        EXPECT_EQ(moves[i].positions[0], kingSq);
        EXPECT_EQ(moves[i].positions[1], results[i]);
    }
}

TEST(KingsQuietMovesTest, BlockedByFriendly) {
    std::vector<Move> moves;
    constexpr int kingSq = 27;   // D4
    constexpr int blocker = 36;  // E5 (Directly NE)
    constexpr uint64_t kings = (1ULL << kingSq);
    constexpr uint64_t boardState = kings | (1ULL << blocker);

    createAllKingsQuietMoves(moves, kings, boardState);

    // NE direction (36, 45, 54, 63) is now blocked.
    // Original: {0, 6, 9, 13, 18, 20, 34, 36, 41, 45, 48, 54, 63}
    // New: Removes 36, 45, 54, 63. Total size: 9
    EXPECT_EQ(moves.size(), 9);

    for (size_t i = 0; i < moves.size(); ++i) {
        constexpr std::array expectedResults = {0, 6, 9, 13, 18, 20, 34, 41, 48};
        EXPECT_EQ(moves[i].from_mask, 1ULL << kingSq);
        EXPECT_EQ(moves[i].to_mask, 1ULL << expectedResults[i]);
        EXPECT_EQ(moves[i].positions[0], kingSq);
        EXPECT_EQ(moves[i].positions[1], expectedResults[i]);
    }
}

TEST(KingsQuietMovesTest, EdgeMobilityA4) {
    std::vector<Move> moves;
    constexpr int kingSq = 24; // A4
    constexpr uint64_t kings = (1ULL << kingSq);
    constexpr uint64_t boardState = kings;

    createAllKingsQuietMoves(moves, kings, boardState);

    // Diagonals from A4 (24):
    // NE: 33, 42, 51, 60
    // SE: 17, 10, 3
    // (NW and SW are off-board)
    EXPECT_EQ(moves.size(), 7);

    for (size_t i = 0; i < moves.size(); ++i) {
        constexpr std::array results = {3, 10, 17, 33, 42, 51, 60};
        EXPECT_EQ(moves[i].to_mask, 1ULL << results[i]);
    }
}

TEST(KingsQuietMovesTest, BlockedByOpponent) {
    std::vector<Move> moves;
    constexpr int kingSq = 18;    // C3
    constexpr int opponent1 = 9;   // B2 (Directly SW)
    constexpr uint64_t kings = (1ULL << kingSq);
    constexpr uint64_t boardState = kings | (1ULL << opponent1);

    // Important note:
    //
    // createAllKingsQuiteMoves function does not check for the
    // attack possibilities, that is in this case it is enough to
    // add blocking piece at 9.
    createAllKingsQuietMoves(moves, kings, boardState);

    // Expected destinations in LSB-to-MSB order:
    // SE direction: 4, 11
    // NW direction: 25, 32, 39, 46, 53, 60
    // NE direction: 27, 36, 45, 54, 63
    // Note: 9 (Opponent) and 0 (Behind opponent) are excluded.
    constexpr std::array results = {4, 11, 25, 27, 32, 36, 45, 54, 63};

    ASSERT_EQ(moves.size(), results.size());
    for (size_t i = 0; i < moves.size(); ++i) {
        EXPECT_EQ(moves[i].from_mask, 1ULL << kingSq);
        EXPECT_EQ(moves[i].to_mask, 1ULL << results[i]);
        EXPECT_EQ(moves[i].positions[0], kingSq);
        EXPECT_EQ(moves[i].positions[1], results[i]);
    }
}

TEST(KingsQuietMovesTest, MultipleKingsCorner) {
    std::vector<Move> moves;
    constexpr int k1 = 0;  // A1
    constexpr int k2 = 63; // H8
    constexpr uint64_t kings = (1ULL << k1) | (1ULL << k2);
    constexpr uint64_t boardState = kings;

    createAllKingsQuietMoves(moves, kings, boardState);

    // Note: Because the other king is on the board, the long diagonal is blocked
    // at the very last square for both.
    constexpr std::array resultsK1 = {9, 18, 27, 36, 45, 54};

    // Total moves should be 12.
    ASSERT_EQ(moves.size(), 12);

    // Verify first 6 moves (from King 0)
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(moves[i].from_mask, 1ULL << k1);
        EXPECT_EQ(moves[i].to_mask, 1ULL << resultsK1[i]);
        EXPECT_EQ(moves[i].positions[0], k1);
    }

    for (size_t i = 0; i < 6; ++i) {
        const size_t moveIdx = i + 6;
        EXPECT_EQ(moves[moveIdx].from_mask, 1ULL << k2);
        EXPECT_EQ(moves[moveIdx].to_mask, 1ULL << resultsK1[i]);
        EXPECT_EQ(moves[moveIdx].positions[0], k2);
    }
}