//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_BOARD_CUH
#define GPU_BOARD_CUH

#include "board_infra.h"
#include "constants.h"

struct GPU_Board
{
    GPU_Board() = default;
    __host__ explicit GPU_Board(const Board &t_hostBoard)
    {
        pawns[white] = t_hostBoard.pawns[white];
        pawns[black] = t_hostBoard.pawns[black];
        kings[white] = t_hostBoard.kings[white];
        kings[black] = t_hostBoard.kings[black];
        kings[white] = t_hostBoard.kings_quiet_moves;
    }
    size_t   pawns[2]{};
    size_t   kings[2]{};
    uint16_t kings_quiet_moves = 0;
};

#endif