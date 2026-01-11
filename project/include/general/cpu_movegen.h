//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_MOVEGEN_H
#define MCTS_CHECKERS_CPU_MOVEGEN_H

#include <vector>

#include "board_infra.cuh"
#include "move.h"



std::vector<Move> generateAllPossibleMoves(const Board &board);
size_t getNormalMoves(size_t t_boardState, Colour t_color);

#endif // MCTS_CHECKERS_CPU_MOVEGEN_H
