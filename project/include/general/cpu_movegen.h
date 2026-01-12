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
size_t            getPawnsAttackMask(size_t t_attackerPawns, size_t t_opponentPieces, size_t t_emptyFiles);
size_t            getPawnsMovesMask(size_t t_moversPawnsMask, size_t t_emptyFiles, Colour t_moversColour);
std::vector<Move> createAllPawnsAttacks(size_t t_onMove, size_t t_opponent, Colour t_onMoveColour);
Move              createOnePawnsAttack(size_t t_onMove, size_t t_opponent, int t_idx, Colour t_onMoveColour);
void createAllPawnMoves(std::vector<Move> &t_allMoves, size_t t_moversMask, size_t t_emptyFiles, Colour t_moversColour);
void createOnePawnsMove(std::vector<Move> &t_allMoves, int t_idx, size_t t_emptyFiles, Colour t_moversColour);

#endif // MCTS_CHECKERS_CPU_MOVEGEN_H
