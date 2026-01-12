//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_MOVEGEN_H
#define MCTS_CHECKERS_CPU_MOVEGEN_H

#include <vector>

#include "board_infra.cuh"
#include "move.h"

// enum Colour {
//     black = 0,
//     white = 1,
// };

// An array to perform branchless
// sliding move generation.
//
// ------------------------------
constexpr int canMove[2][4] = {
    {0, 0, 1, 1}, // this represents black
    {1, 1, 0 ,0}  // this represents white
};

std::vector<Move> generateAllPossibleMoves(const Board &t_board, Colour t_color);
size_t            getPawnsAttackMask(size_t t_attacker, size_t t_opponent);
size_t            getPawnsMovesMask(Colour t_onMoveColour, size_t t_onMove, size_t t_opponent);
void              checkMultiAttacks();

#endif // MCTS_CHECKERS_CPU_MOVEGEN_H
