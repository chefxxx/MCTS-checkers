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

    uint16_t kings_quiet_moves = 0;

    void initStartingBoard()
    {
        constexpr size_t firstRowMask  = 85;
        constexpr size_t secondRowMask = 170;

        pawns[white]      = 0ULL;
        pawns[black]      = 0ULL;
        kings_quiet_moves = 0;
        kings[white]      = 0ULL;
        kings[black]      = 0ULL;

        // initialize white player rocks
        pawns[white] |= firstRowMask;
        pawns[white] |= secondRowMask << 8;
        pawns[white] |= firstRowMask << 16;

        // initialize black player rocks
        pawns[black] |= secondRowMask << 40;
        pawns[black] |= firstRowMask << 48;
        pawns[black] |= secondRowMask << 56;
    }

    bool operator==(const Board &other) const
    {
        return pawns[white] == other.pawns[white] && pawns[black] == other.pawns[black]
            && kings[white] == other.kings[white] && kings[black] == other.kings[black]
            && kings_quiet_moves == other.kings_quiet_moves;
    }
    bool operator!=(const Board &other) const { return !(*this == other); }
};


#endif