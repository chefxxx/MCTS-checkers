//
// Created by chefxx on 13.11.2025.
//

#include <gtest/gtest.h>
#include "board_infra.cuh"

TEST(BoardPrint, SimplePrint)
{
    const Board board{Colour::white};
    board.printBoard();
}