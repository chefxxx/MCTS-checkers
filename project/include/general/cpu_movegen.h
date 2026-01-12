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

// An array to perform branchless
// updating of current attacker idx.
//
// ---------------------------------
constexpr int idxUpdate[4] = {
    18,
    14,
    -14,
    -18
};

std::vector<Move> generateAllPossibleMoves(const Board &t_board, Colour t_color);
size_t            getPawnsAttackMask(size_t t_attackerPawns, size_t t_opponentPieces, size_t t_emptyFiles);
size_t            getPawnsMovesMask(size_t t_moversPawnsMask, size_t t_emptyFiles, Colour t_moversColour);
void              createAllPawnsAttacks(std::vector<Move> &t_allMoves,
                                        size_t             t_attackersMask,
                                        size_t             t_opponentPieces,
                                        size_t             t_emptyFiles,
                                        Colour             t_attackersColour);
void recursiveCreatePawnsAttacks(std::vector<Move> &t_allMoves,
                                              int                t_idx,
                                              size_t             t_opponentPieces,
                                              size_t             t_emptyFiles,
                                              size_t             t_currentVictimsMask,
                                              Colour             t_attackerColour);
void createAllPawnsMoves(std::vector<Move> &t_allMoves, size_t t_moversMask, size_t t_emptyFiles, Colour t_moversColour);
void createOnePawnMoves(std::vector<Move> &t_allMoves, int t_idx, size_t t_emptyFiles, Colour t_moversColour);

size_t updateEmptyMaskAfterAttack(size_t t_emptyFiles, size_t t_attack, size_t t_jump, int t_idx);
#endif // MCTS_CHECKERS_CPU_MOVEGEN_H
