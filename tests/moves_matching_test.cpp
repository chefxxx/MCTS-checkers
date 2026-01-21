#include <gtest/gtest.h>
#include <vector>

#include "game_engine.h"
#include "move.h"
//
// class MoveConstructorTest : public ::testing::Test {
// protected:
//     // Helper to compare paths easily
//     void AssertPathsEqual(const Move& m1, const Move& m2) {
//         // Assuming LightMovePath has an equality operator
//         EXPECT_EQ(m1.path, m2.path) << "Paths do not match!";
//         EXPECT_EQ(m1.from_mask, m2.from_mask);
//         EXPECT_EQ(m1.to_mask, m2.to_mask);
//         EXPECT_EQ(m1.captures_mask, m2.captures_mask);
//     }
// };
//
// // 1. Test Simple (Quiet) Move
// TEST_F(MoveConstructorTest, QuietMoveConsistency) {
//     constexpr int from = 18; // e.g., c3
//     constexpr int to = 27;   // e.g., d4
//
//     // AI version
//     const Move aiMove(from, to);
//     const auto playerMove = processMoveString("c3-d4");
//     AssertPathsEqual(aiMove, playerMove.value());
// }
//
// // 2. Test Single Capture
// TEST_F(MoveConstructorTest, SingleCaptureConsistency) {
//     // Example: c3 (18) jumps over d4 (27) to e5 (36)
//     constexpr int from = 18;
//     constexpr int mid = 27;
//     constexpr int to = 36;
//     constexpr size_t captures = (1ULL << mid);
//     const std::vector path = {from, to};
//
//     // AI version (Constructor 2)
//     const Move aiMove(captures, path);
//     const auto playerMove = processMoveString("c3:e5");
//
//     AssertPathsEqual(aiMove, playerMove.value());
// }
//
// // 3. Test Multi-Capture (The likely point of failure)
// TEST_F(MoveConstructorTest, MultiCaptureConsistency) {
//     const std::vector fullPath = {18, 36, 54};
//     constexpr size_t captures = (1ULL << 27) | (1ULL << 45);
//
//     const Move aiMove(captures, fullPath);
//     const auto playerMove = processMoveString("c3:e5:g7");
//     AssertPathsEqual(aiMove, playerMove.value());
// }
//
//
// TEST_F(MoveConstructorTest, blackMoveConstructorTest) {
//     constexpr int from = 41;
//     constexpr int to = 32;
//
//     const Move aiMove(from, to);
//     const auto playerMove = processMoveString("b6-a5");
//     AssertPathsEqual(aiMove, playerMove.value());
// }