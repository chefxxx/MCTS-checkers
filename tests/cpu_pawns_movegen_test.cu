//
// Created by chefxx on 13.11.2025.
//

#include <gtest/gtest.h>

#include "cpu_movegen.h"

// white: 10, 14, 34
// black: 19, 43, 23
// white on the move
// result: 10, 34
TEST(PawnsCPU_MovegenTest, getPawnsAttackMask_returns_upRightAttacks)
{
    constexpr auto white_mask  = (1ULL << 10) | (1ULL << 14) | (1ULL << 34);
    constexpr auto black_mask  = (1ULL << 19) | (1ULL << 23) | (1ULL << 43);
    constexpr auto result_mask = (1ULL << 10) | (1ULL << 34);
    constexpr auto empty_mask  = ~(white_mask | black_mask);
    const auto     test_case   = getPawnsAttackMask(white_mask, black_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 18
// black: 25
// landing: 32 (empty)
// result: 18 (the jumper)
TEST(PawnsCPU_MovegenTest, getPawnsAttackMask_returns_upLeftAttacks)
{
    constexpr uint64_t white_mask  = (1ULL << 18);
    constexpr uint64_t black_mask  = (1ULL << 25);
    constexpr uint64_t result_mask = (1ULL << 18);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    const auto         test_case   = getPawnsAttackMask(white_mask, black_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// black: 42
// white: 35
// landing: 28 (empty)
// result: 42 (the jumper)
TEST(PawnsCPU_MovegenTest, getPawnsAttackMask_returns_downRightAttacks)
{
    constexpr uint64_t black_mask  = (1ULL << 42);
    constexpr uint64_t white_mask  = (1ULL << 35);
    constexpr uint64_t result_mask = (1ULL << 42);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    // We pass black_mask as t_mask because black is the "attacker" here
    const auto test_case = getPawnsAttackMask(black_mask, white_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 16 (File A)
// black: 25
// result: 0 (cannot jump left from File A)
TEST(PawnsCPU_MovegenTest, getPawnsAttackMask_does_not_wrap_fileA)
{
    constexpr uint64_t white_mask  = (1ULL << 16);
    constexpr uint64_t black_mask  = (1ULL << 25);
    constexpr uint64_t result_mask = 0ULL;
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    const auto         test_case   = getPawnsAttackMask(black_mask, white_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 10, 14
// black: 19, 21
// landing: 28 (empty)
// result: 10, 14
TEST(PawnsCPU_MovegenTest, getPawnsAttackMask_returns_multiple_attackers)
{
    constexpr uint64_t white_mask  = (1ULL << 10) | (1ULL << 14);
    constexpr uint64_t black_mask  = (1ULL << 19) | (1ULL << 21);
    constexpr uint64_t result_mask = (1ULL << 10) | (1ULL << 14);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    const auto         test_case   = getPawnsAttackMask(white_mask, black_mask, empty_mask);
    EXPECT_EQ(result_mask, test_case);
}

// white: 19, 26, 28
// pawn 19 is blocked by 26 and 28,
// so the only pawns to mover are 26 and 28
TEST(PawnsCPU_MovegenTest, getPawnsMovesMask_withBlockedWhitePawn)
{
    constexpr uint64_t white_mask  = (1ULL << 19) | (1ULL << 26) | (1ULL << 28);
    constexpr uint64_t black_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 26) | (1ULL << 28);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    const auto         test_case   = getPawnsQuietMask(white_mask, empty_mask, white);
    ASSERT_EQ(test_case, result_mask);
}

// black: 35
// white: 0
// Result: 35 should be able to move Down-Left (26) or Down-Right (28).
TEST(PawnsCPU_MovegenTest, getPawnsMovesMask_blackMovesDown)
{
    constexpr uint64_t black_mask  = (1ULL << 35);
    constexpr uint64_t white_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 35);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    const auto         test_case   = getPawnsQuietMask(black_mask, empty_mask, black);
    ASSERT_EQ(test_case, result_mask);
}

// white: 15, 22
// Result: 15 cannot move, but 22 can
// The NOT_FILE_H mask should prevent 15 from moving to 24
TEST(PawnsCPU_MovegenTest, getPawnsMovesMask_preventWrapFileH)
{
    constexpr uint64_t white_mask  = (1ULL << 15) | (1ULL << 22);
    constexpr uint64_t black_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 22);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);

    // We check if it correctly identifies 15 as a not mover
    // without allowing it to "teleport" across the board.
    const auto test_case = getPawnsQuietMask(white_mask, empty_mask, white);
    ASSERT_EQ(test_case, result_mask);
}

// black: 44, 35, 37
// Result: 35, 37
TEST(PawnsCPU_MovegenTest, getPawnsMovesMask_fULLyBlocked)
{
    constexpr uint64_t black_mask  = (1ULL << 44) | (1ULL << 35) | (1ULL << 37);
    constexpr uint64_t white_mask  = 0ULL;
    constexpr uint64_t result_mask = (1ULL << 35) | (1ULL << 37);
    constexpr auto     empty_mask  = ~(white_mask | black_mask);
    const auto         test_case   = getPawnsQuietMask(black_mask, empty_mask, black);
    ASSERT_EQ(test_case, result_mask);
}

// white: 27
// empty: all squares except 27
// expected: moves from 27 to 34 and 27 to 36
TEST(PawnsCPU_MovegenTest, createOnePawnsMove_whiteCenter)
{
    constexpr int      start_idx  = 27;
    constexpr uint64_t white_mask = (1ULL << start_idx);
    constexpr uint64_t empty_mask = ~white_mask;

    std::vector<Move> moves;
    createOnePawnQuietMoves(moves, start_idx, empty_mask, white);

    ASSERT_EQ(moves.size(), 2);
    // Move to 34 (Up-Left: 27+7) and 36 (Up-Right: 27+9)
    EXPECT_TRUE(moves[0].to_mask == 1ULL << 34);
    EXPECT_TRUE(moves[1].to_mask == 1ULL << 36);
}

// white: 18, 20
// expected: 4 total moves (2 from each piece)
TEST(PawnsCPU_MovegenTest, createAllPawnMoves_multiplePieces)
{
    constexpr uint64_t white_mask = (1ULL << 18) | (1ULL << 20);
    constexpr uint64_t empty_mask = ~white_mask;

    // We assume getPawnsMovesMask correctly identifies both 18 and 20 as movers
    constexpr uint64_t movers_mask = white_mask;

    std::vector<Move> all_moves;
    createAllPawnsQuietMoves(all_moves, movers_mask, empty_mask, white);

    // Each piece has 2 diagonal forward moves
    EXPECT_EQ(all_moves.size(), 4);
    EXPECT_TRUE(all_moves[0].to_mask == 1ULL << 25 && all_moves[0].from_mask == 1ULL << 18);
    EXPECT_TRUE(all_moves[1].to_mask == 1ULL << 27 && all_moves[1].from_mask == 1ULL << 18);
    EXPECT_TRUE(all_moves[2].to_mask == 1ULL << 27 && all_moves[2].from_mask == 1ULL << 20);
    EXPECT_TRUE(all_moves[3].to_mask == 1ULL << 29 && all_moves[3].from_mask == 1ULL << 20);
}

TEST(PawnsCPU_MovegenTest, recursiveCreatePawnsAttacks_ZigZag)
{
    // Indices for 8x8 setup
    constexpr int start_idx   = 9;  // B2
    constexpr int victim1_idx = 18; // C3 (Up-Right)
    constexpr int victim2_idx = 34; // C5 (Up-Left)
    constexpr int victim3_idx = 50; // C7
    constexpr int final_idx   = 59; // D8

    constexpr size_t white_start_mask = 1ULL << start_idx;
    constexpr size_t black_pieces     = 1ULL << victim1_idx | 1ULL << victim2_idx | 1ULL << victim3_idx;
    constexpr size_t empty_files      = ~(white_start_mask | black_pieces);

    std::vector<Move> result_moves;
    std::vector       path{start_idx};
    // EXECUTE
    recursiveCreatePawnsAttacks(result_moves,
                                path,
                                start_idx,
                                black_pieces,
                                empty_files, // No captures yet
                                0ULL,        // Original start
                                white_start_mask,
                                white);

    // VERIFY
    // There should be exactly one legal path found
    ASSERT_EQ(result_moves.size(), 1);

    constexpr size_t expected_captures = black_pieces;
    const Move      &m                 = result_moves[0];

    EXPECT_EQ(m.from_mask, white_start_mask);
    EXPECT_EQ(m.to_mask, (1ULL << final_idx));
    EXPECT_EQ(m.captures_mask, expected_captures);
}

TEST(PawnsCPU_MovegenTest, recursiveCreatePawnsAttacks_simpleBranching)
{
    // SETUP
    constexpr int    start_idx        = 8;
    constexpr size_t white_start_mask = (1ULL << start_idx);

    // Opponents placed to create a fork in the path
    constexpr size_t victim1 = (1ULL << 17); // B3
    constexpr size_t victim2 = (1ULL << 35); // D5
    constexpr size_t victim3 = (1ULL << 19); // D3

    constexpr size_t opponent_pieces = victim1 | victim2 | victim3;
    constexpr size_t empty_files     = ~(white_start_mask | opponent_pieces);

    std::vector<Move> result_moves;
    std::vector       path{start_idx};
    // EXECUTE
    recursiveCreatePawnsAttacks(result_moves,
                                path,
                                start_idx,
                                opponent_pieces,
                                empty_files, // No captures yet
                                0ULL,        // Original start
                                white_start_mask,
                                white);

    // VERIFY
    // We expect 2 distinct terminal moves from the same starting piece
    ASSERT_EQ(result_moves.size(), 2);

    // since the choice of directions while searching for moves
    // is deterministic, always goes UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT
    // we can do assertions here

    const auto move1 = result_moves[0];
    ASSERT_EQ(move1.from_mask, white_start_mask);
    ASSERT_EQ(move1.to_mask, 1ULL << 44);
    ASSERT_EQ(move1.captures_mask, victim1 | victim2);

    const auto move2 = result_moves[1];
    ASSERT_EQ(move2.from_mask, white_start_mask);
    ASSERT_EQ(move2.to_mask, 1ULL << 12);
    ASSERT_EQ(move2.captures_mask, victim1 | victim3);
}

TEST(PawnsCPU_MovegenTest, diamondCase)
{
    // SETUP
    constexpr int    start_idx        = 11;
    constexpr size_t white_start_mask = (1ULL << start_idx);

    constexpr size_t victim1 = (1ULL << 20);
    constexpr size_t victim2 = (1ULL << 36);
    constexpr size_t victim3 = (1ULL << 34);
    constexpr size_t victim4 = (1ULL << 18);

    constexpr size_t opponent_pieces = victim1 | victim2 | victim3 | victim4;
    constexpr size_t empty_files     = ~(white_start_mask | opponent_pieces);

    std::vector<Move> result_moves;
    std::vector       path{start_idx};
    // EXECUTE
    recursiveCreatePawnsAttacks(result_moves,
                                path,
                                start_idx,
                                opponent_pieces,
                                empty_files, // No captures yet
                                0ULL,        // Original start
                                white_start_mask,
                                white);

    ASSERT_EQ(result_moves.size(), 2);
}