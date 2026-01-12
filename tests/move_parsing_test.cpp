//
// Created by chefxx on 13.11.2025.
//

#include <gtest/gtest.h>

#include "game_engine.h"

TEST(MoveParse, simpleParse)
{
    const std::string normalMove{"a1-b2"};
    const std::string takeMove{"c3:e5"};
    const std::string doubleTakeMove{"c1:e3:g1"};

    const auto mv0 = parseMove(normalMove).value();
    ASSERT_EQ(mv0.positions[0], 0);
    ASSERT_EQ(mv0.positions[1], 9);
    ASSERT_EQ(mv0.kind, MoveKind::normal);

    const auto mv1 = parseMove(takeMove).value();
    ASSERT_EQ(mv1.positions[0], 18);
    ASSERT_EQ(mv1.positions[1], 36);
    ASSERT_EQ(mv1.kind, MoveKind::take);

    const auto mv2 = parseMove(doubleTakeMove).value();
    ASSERT_EQ(mv2.positions[0], 2);
    ASSERT_EQ(mv2.positions[1], 20);
    ASSERT_EQ(mv2.positions[2], 6);
    ASSERT_EQ(mv2.kind, MoveKind::take);
}
