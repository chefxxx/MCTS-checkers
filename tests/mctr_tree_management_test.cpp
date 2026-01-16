//
// Created by chefxx on 16.01.2026.
//

#include <gtest/gtest.h>

#include "mcts_tree.h"

TEST(MctsTreeBasicTests, creationTest)
{
    Board board;
    board.initStartingBoard();
    const MctsTree mcts_tree{board, white};
    ASSERT_FALSE(mcts_tree.root == nullptr);
}

