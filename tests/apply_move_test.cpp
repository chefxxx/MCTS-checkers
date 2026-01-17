//
// Created by chefxx on 17.01.2026.
//

#include <gtest/gtest.h>

// 1. White normal move and promotes
// TEST(PawnsCPU_MovegenTest, WhiteSlidePromotes)
// {
//     constexpr int      start_idx = 49; // B7
//     constexpr uint64_t empty     = ~(1ULL << 49);
//     std::vector<Move>  moves;
//
//     createOnePawnQuietMoves(moves, start_idx, empty, white);
//     // Expected: 49 -> 56 (Up-Left) or 49 -> 58 (Up-Right). Both are Rank 8.
//     for (const auto &m : moves) {
//         EXPECT_TRUE(m.is_promotion);
//     }
// }
//
// // 2. White attack move and promotes
// TEST(PawnsCPU_MovegenTest, WhiteJumpPromotes)
// {
//     constexpr int start_idx   = 42; // C6
//     constexpr int victim_idx  = 51; // D7
//     constexpr int landing_idx = 60; // E8 (Rank 8)
//
//     constexpr size_t  opponents = (1ULL << victim_idx);
//     constexpr size_t  empty     = ~((1ULL << start_idx) | opponents);
//     std::vector<Move> moves;
//     std::vector       path{start_idx};
//
//     recursiveCreatePawnsAttacks(moves, path, start_idx, opponents, empty, 0, (1ULL << start_idx), white);
//
//     ASSERT_EQ(moves.size(), 1);
//     EXPECT_EQ(getLsb(moves[0].to_mask), landing_idx);
//     EXPECT_TRUE(moves[0].is_promotion);
// }
//
// // 3. White attack move THROUGH promote square, but does not promote
// TEST(PawnsCPU_MovegenTest, WhiteJumpThroughPromoteNoPromotion)
// {
//     constexpr int start_idx     = 42; // C6
//     constexpr int victim1       = 51; // D7
//     constexpr int victim2       = 53; // F7
//     constexpr int final_landing = 46; // G6 (Rank 6 - Final)
//
//     constexpr size_t  opponents = (1ULL << victim1) | (1ULL << victim2);
//     constexpr size_t  empty     = ~((1ULL << start_idx) | opponents);
//     std::vector<Move> moves;
//     std::vector       path{start_idx};
//
//     recursiveCreatePawnsAttacks(moves, path, start_idx, opponents, empty, 0, (1ULL << start_idx), white);
//
//     ASSERT_EQ(moves.size(), 1);
//     EXPECT_EQ(getLsb(moves[0].to_mask), final_landing);
//     // Must be false because it landed on Rank 6 at the end of the turn
//     EXPECT_FALSE(moves[0].is_promotion);
// }
//
// // 4. White attack move and lands on Row 1 - does not promote
// TEST(PawnsCPU_MovegenTest, WhiteJumpBackwardsNoPromotion)
// {
//     constexpr int start_idx   = 18; // C3
//     constexpr int victim_idx  = 9;  // B2
//     constexpr int landing_idx = 0;  // A1 (Rank 1)
//
//     constexpr size_t  opponents = (1ULL << victim_idx);
//     constexpr size_t  empty     = ~((1ULL << start_idx) | opponents);
//     std::vector<Move> moves;
//     std::vector       path{start_idx};
//     recursiveCreatePawnsAttacks(moves, path, start_idx, opponents, empty, 0, (1ULL << start_idx), white);
//
//     ASSERT_EQ(moves.size(), 1);
//     EXPECT_EQ(getLsb(moves[0].to_mask), landing_idx);
//     EXPECT_FALSE(moves[0].is_promotion);
// }

// 1. Black normal move and promotes
// TEST(PawnsCPU_MovegenTest, BlackSlidePromotes)
// {
//     constexpr int      start_idx = 9; // B2
//     constexpr uint64_t empty     = ~(1ULL << 9);
//     std::vector<Move>  moves;
//
//     createOnePawnQuietMoves(moves, start_idx, empty, black);
//     // Expected: 9 -> 0 (Down-Left) or 9 -> 2 (Down-Right). Both Rank 1.
//     for (const auto &m : moves) {
//         EXPECT_TRUE(m.is_promotion);
//     }
// }
//
// // 2. Black attack move and promotes
// TEST(PawnsCPU_MovegenTest, BlackJumpPromotes)
// {
//     constexpr int start_idx  = 18; // C3
//     constexpr int victim_idx = 9;  // B2
//
//     constexpr size_t  opponents = (1ULL << victim_idx);
//     constexpr size_t  empty     = ~((1ULL << start_idx) | opponents);
//     std::vector<Move> moves;
//     std::vector       path{start_idx};
//
//     recursiveCreatePawnsAttacks(moves, path, start_idx, opponents, empty, 0, (1ULL << start_idx), black);
//
//     ASSERT_EQ(moves.size(), 1);
//     EXPECT_TRUE(moves[0].is_promotion);
// }
//
// // 3. Black attack move THROUGH promote square, but does not promote
// TEST(PawnsCPU_MovegenTest, BlackJumpThroughPromoteNoPromotion)
// {
//     constexpr int start_idx     = 16; // A3
//     constexpr int victim1       = 9;  // B2
//     constexpr int victim2       = 11; // D2
//     constexpr int final_landing = 20; // E3 (Rank 3 - Final)
//
//     constexpr size_t  opponents = (1ULL << victim1) | (1ULL << victim2);
//     constexpr size_t  empty     = ~((1ULL << start_idx) | opponents);
//     std::vector<Move> moves;
//     std::vector       path{start_idx};
//
//     recursiveCreatePawnsAttacks(moves, path, start_idx, opponents, empty, 0, (1ULL << start_idx), black);
//
//     ASSERT_EQ(moves.size(), 1);
//     EXPECT_EQ(getLsb(moves[0].to_mask), final_landing);
//     EXPECT_FALSE(moves[0].is_promotion);
// }
//
// // 4. Black attack move and lands on Row 8 - does not promote
// TEST(PawnsCPU_MovegenTest, BlackJumpBackwardsNoPromotion)
// {
//     constexpr int start_idx  = 42; // C6
//     constexpr int victim_idx = 51; // D7
//
//     constexpr size_t  opponents = (1ULL << victim_idx);
//     constexpr size_t  empty     = ~((1ULL << start_idx) | opponents);
//     std::vector<Move> moves;
//     std::vector       path{start_idx};
//
//     recursiveCreatePawnsAttacks(moves, path, start_idx, opponents, empty, 0, (1ULL << start_idx), black);
//
//     ASSERT_EQ(moves.size(), 1);
//     EXPECT_FALSE(moves[0].is_promotion);
// }