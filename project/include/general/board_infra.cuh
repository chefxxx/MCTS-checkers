//
// Created by chefxx on 8.01.2026.
//

#ifndef BOARD_INFRA_H
#define BOARD_INFRA_H

#include <array>
#include "constants.h"

struct Board
{
    // this is the state of the board
    // 0 represents black pieces
    // 1 represents white pieces
    std::array<size_t, 2> pawns{};
    std::array<size_t, 2> kings{};

    void initStartingBoard()
    {
        constexpr size_t firstRowMask  = 85;
        constexpr size_t secondRowMask = 170;

        pawns[white] = 0ull;
        pawns[black] = 0ull;

        // initialize white player rocks
        pawns[white] |= firstRowMask;
        pawns[white] |= secondRowMask << 8;
        pawns[white] |= firstRowMask << 16;

        // initialize black player rocks
        pawns[black] |= secondRowMask << 40;
        pawns[black] |= firstRowMask << 48;
        pawns[black] |= secondRowMask << 56;
    }
};

#endif