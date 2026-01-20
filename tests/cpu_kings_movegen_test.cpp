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

TEST(KingsQuietMovesTest, CenterEmptyBoard)
{
    std::vector<Move>  moves;
    constexpr int      kingSq     = 27; // D4
    constexpr uint64_t kings      = (1ULL << kingSq);
    constexpr uint64_t boardState = kings; // Empty board

    createAllKingsQuietMoves(moves, kings, boardState);

    EXPECT_EQ(moves.size(), 13);
    for (size_t i = 0; i < moves.size(); ++i) {
        constexpr std::array results = {0, 6, 9, 13, 18, 20, 34, 36, 41, 45, 48, 54, 63};
        auto mv = PrintingMovePath(moves[i].path.packed_path);
        EXPECT_EQ(moves[i].from_mask, 1ULL << kingSq);
        EXPECT_EQ(moves[i].to_mask, 1ULL << results[i]);
        EXPECT_EQ(mv.positions[0], kingSq);
        EXPECT_EQ(mv.positions[1], results[i]);
    }
}

TEST(KingsQuietMovesTest, BlockedByFriendly)
{
    std::vector<Move>  moves;
    constexpr int      kingSq     = 27; // D4
    constexpr int      blocker    = 36; // E5 (Directly NE)
    constexpr uint64_t kings      = (1ULL << kingSq);
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
        auto mv = PrintingMovePath(moves[i].path.packed_path);
        EXPECT_EQ(mv.positions[0], kingSq);
        EXPECT_EQ(mv.positions[1], expectedResults[i]);
    }
}

TEST(KingsQuietMovesTest, EdgeMobilityA4)
{
    std::vector<Move>  moves;
    constexpr int      kingSq     = 24; // A4
    constexpr uint64_t kings      = (1ULL << kingSq);
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

TEST(KingsQuietMovesTest, BlockedByOpponent)
{
    std::vector<Move>  moves;
    constexpr int      kingSq     = 18; // C3
    constexpr int      opponent1  = 9;  // B2 (Directly SW)
    constexpr uint64_t kings      = (1ULL << kingSq);
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
        auto mv = PrintingMovePath(moves[i].path.packed_path);
        EXPECT_EQ(mv.positions[0], kingSq);
        EXPECT_EQ(mv.positions[1], results[i]);
    }
}

TEST(KingsQuietMovesTest, MultipleKingsCorner)
{
    std::vector<Move>  moves;
    constexpr int      k1         = 0;  // A1
    constexpr int      k2         = 63; // H8
    constexpr uint64_t kings      = (1ULL << k1) | (1ULL << k2);
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
        auto mv = PrintingMovePath(moves[i].path.packed_path);
        EXPECT_EQ(mv.positions[0], k1);
    }

    for (size_t i = 0; i < 6; ++i) {
        const size_t moveIdx = i + 6;
        EXPECT_EQ(moves[moveIdx].from_mask, 1ULL << k2);
        EXPECT_EQ(moves[moveIdx].to_mask, 1ULL << resultsK1[i]);
        auto mv = PrintingMovePath(moves[moveIdx].path.packed_path);
        EXPECT_EQ(mv.positions[0], k2);
    }
}

TEST(KingsAttackTest, SingleFlyingJumpMultipleLandings)
{
    std::vector<Move> moves;
    std::vector<int>  path;
    constexpr int     kingSq   = 0;  // A1
    constexpr int     victimSq = 18; // C3
    // Potential landings: 27 (D4), 36 (E5), 45 (F6), 54 (G7), 63 (H8)

    constexpr uint64_t kings     = (1ULL << kingSq);
    constexpr uint64_t opponents = (1ULL << victimSq);
    constexpr uint64_t board     = kings | opponents;

    path.push_back(kingSq);
    recursiveCreateAllKingsAttacks(moves, path, kingSq, board, opponents, kings, 0);

    // There should be 5 possible moves (one for each landing square behind C3)
    ASSERT_EQ(moves.size(), 5);

    // Sort or check specific landing squares
    for (size_t i = 0; i < moves.size(); ++i) {
        constexpr std::array expectedLandings = {27, 36, 45, 54, 63};
        EXPECT_EQ(moves[i].to_mask, 1ULL << expectedLandings[i]);
        EXPECT_EQ(moves[i].captures_mask, 1ULL << victimSq);
    }
}

TEST(KingsAttackTest, MandatoryDoubleJump)
{
    std::vector<Move> moves;
    std::vector<int>  path;

    constexpr int kingSq  = 0;  // A1
    constexpr int victim1 = 18; // C3
    constexpr int victim2 = 45; // F6
    // Logic: 0 jumps 18, must land on 27 (D4) to be able to jump 45.
    // From 27, it jumps 45 and can land on 54 (G7) or 63 (H8).

    constexpr uint64_t kings     = (1ULL << kingSq);
    constexpr uint64_t opponents = (1ULL << victim1) | (1ULL << victim2);
    constexpr uint64_t board     = kings | opponents;

    path.push_back(kingSq);
    recursiveCreateAllKingsAttacks(moves, path, kingSq, board, opponents, kings, 0);

    // If the King lands on 36, 45, 54, 63 after the first jump, it cannot jump victim2.
    // But in most rulesets, you must choose the path that continues the jump.
    // Therefore, only moves ending at 54 or 63 (after jumping BOTH) are valid.

    EXPECT_EQ(moves.size(), 4);

    for (int i = 0; i < 4; ++i) {
        constexpr std::array expectedMidPoints{27, 27, 36, 36};
        constexpr std::array expectedLandings{54, 63, 54, 63};
        EXPECT_EQ(moves[i].to_mask, 1ULL << expectedLandings[i]);
        EXPECT_EQ(moves[i].from_mask, 1ULL << kingSq);
        auto mv = PrintingMovePath(moves[i].path.packed_path);
        EXPECT_EQ(mv.positions[0], kingSq);
        EXPECT_EQ(mv.positions[1], expectedMidPoints[i]);
        EXPECT_EQ(mv.positions[2], expectedLandings[i]);
    }
}

TEST(KingsAttackTest, MultipleChoicesAndGhostPieceBlocker)
{
    std::vector<Move> moves;
    std::vector<int>  path;

    constexpr int kingSq = 2;
    // King jumps 9, lands on 18. Now 18 wants to jump 27, but 9 is still there!
    // The King's ray should be blocked by the 'ghost' at 9 if it tries to look "backwards".

    constexpr uint64_t kings     = (1ULL << kingSq);
    constexpr uint64_t opponents = 1ULL << 20 | 1ULL << 52 | 1ULL << 50 | 1ULL << 27;
    constexpr uint64_t board     = kings | opponents;
    path.push_back(kingSq);

    recursiveCreateAllKingsAttacks(moves, path, kingSq, board, opponents, kings, 0);
    const auto mv0 = PrintingMovePath(moves[0].path.packed_path);
    const auto mv1 = PrintingMovePath(moves[1].path.packed_path);
    const auto mv2 = PrintingMovePath(moves[2].path.packed_path);
    EXPECT_EQ(moves.size(), 3);
    for (int i = 0; i < 3; ++i) {
        constexpr std::array expectedMove1{2, 29, 57};
        EXPECT_EQ(mv0.positions[i], expectedMove1[i]);
    }
    for (int i = 0; i < 4; ++i) {
        constexpr std::array expectedMove3 = {2, 38, 59, 41};
        constexpr std::array expectedMove2 = {2, 38, 59, 32};
        EXPECT_EQ(mv1.positions[i], expectedMove2[i]);
        EXPECT_EQ(mv2.positions[i], expectedMove3[i]);
    }
}

TEST(KingsAttackTest, FourWayBranchingAndCompletion)
{
    std::vector<Move> moves;
    std::vector<int>  path;

    constexpr int kingSq = 27; // D4

    // Opponent pieces
    constexpr int vNE   = 36; // E5
    constexpr int vNW   = 34; // C5
    constexpr int vSE_1 = 20; // E3
    constexpr int vSE_2 = 18; // G3 (Actually let's use 11 for a cleaner jump)
    // Note: To make SE a double jump: 27 jumps 20, lands on 13.
    // From 13, it jumps 11 and lands on 4.

    constexpr uint64_t kings     = (1ULL << kingSq);
    constexpr uint64_t opponents = (1ULL << vNE) | (1ULL << vNW) | (1ULL << vSE_1) | (1ULL << vSE_2);
    constexpr uint64_t board     = kings | opponents;

    path.push_back(kingSq);
    recursiveCreateAllKingsAttacks(moves, path, kingSq, board, opponents, kings, 0);

    // Expected Results:
    // 1. NE Branch: One jump to F6(45), G7(54), H8(63). (3 moves)
    // 2. NW Branch: One jump to B6(41), A7(48). (2 moves)
    // 3. SE Branch: Must jump 20, land on 13, then jump 11, then land on 4 or 2. (2 moves)
    // Total expected moves: 7

    ASSERT_EQ(moves.size(), 9);
    for (int i = 0; i < 9; ++i) {
        constexpr std::array expectedTakes{18, 18, 20, 20, 34, 34, 36, 36, 36};
        constexpr std::array expectedTo{0, 9, 6, 13, 41, 48, 45, 54, 63};
        EXPECT_EQ(moves[i].to_mask, 1ULL << expectedTo[i]);
        const auto mv = PrintingMovePath(moves[i].path.packed_path);
        EXPECT_EQ(mv.positions[1], expectedTo[i]);
        EXPECT_EQ(moves[i].captures_mask, 1ULL << expectedTakes[i]);
    }
}