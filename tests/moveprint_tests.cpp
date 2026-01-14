//
// Created by chefxx on 14.01.2026.
//

#include <gtest/gtest.h>
#include "move.h"

TEST(MovePrintingTest, NormalMoveSlide) {
    constexpr int start_idx = 9;  // b2
    constexpr int end_idx = 18;   // c3

    // Slide Move: captures_mask is 0
    const Move m(1ULL << start_idx, 1ULL << end_idx, false, start_idx, end_idx);

    // Simulation of your printing loop
    const auto result = stringMove(m);

    EXPECT_EQ(result, "b2-c3");
}

TEST(MovePrintingTest, SimpleAttackJump) {
    constexpr int start_idx = 0;   // a1
    constexpr int end_idx = 18;    // c3
    const std::vector path = {start_idx, end_idx};
    constexpr auto captures = 1ULL << 9;

    // Jump Move: captures_mask is non-zero
    const Move m(1ULL << start_idx, 1ULL << end_idx, captures,false, path);
    const auto result = stringMove(m);

    EXPECT_EQ(result, "a1:c3");
}

TEST(MovePrintingTest, MultipleAttackJump) {
    const std::vector path = {0, 18, 4, 22}; // a1, c3, e1, g3

    // Multiple Jump: captures_mask would have 3 bits set
    const Move m(1ULL << 0, 1ULL << 22, 1ULL << 9 | 1ULL << 11 | 1ULL << 13,false, path);
    const auto result = stringMove(m);

    EXPECT_EQ(result, "a1:c3:e1:g3");
}