//
// Created by chefxx on 8.01.2026.
//

#ifndef BOARD_INFRA_H
#define BOARD_INFRA_H

#include <array>
#include <string>

struct Board
{
    // this is the state of the board
    // 0 represents black pieces
    // 1 represents white pieces
    std::array<size_t, 2> pawns{};
    std::array<size_t, 2> kings{};
};

#endif