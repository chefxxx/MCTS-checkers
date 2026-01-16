//
// Created by chefxx on 16.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_CHECKERS_ENGINE_H
#define MCTS_CHECKERS_CPU_CHECKERS_ENGINE_H

#include "board_infra.cuh"
#include "move.h"

Board applyMove(const Board &t_board, const Move &t_move, Colour t_colour);
void checkEndOfGameConditions();

#endif // MCTS_CHECKERS_CPU_CHECKERS_ENGINE_H
