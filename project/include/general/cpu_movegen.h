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
size_t            getPawnsMovesMask(size_t t_moversPawns, size_t t_emptyFiles, Colour t_moversColour);
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
                                              size_t             t_originalStartingPositionMask,
                                              Colour             t_attackerColour);
void createAllPawnsMoves(std::vector<Move> &t_allMoves, size_t t_moversMask, size_t t_emptyFiles, Colour t_moversColour);
void createOnePawnMoves(std::vector<Move> &t_allMoves, int t_idx, size_t t_emptyFiles, Colour t_moversColour);
#endif // MCTS_CHECKERS_CPU_MOVEGEN_H
