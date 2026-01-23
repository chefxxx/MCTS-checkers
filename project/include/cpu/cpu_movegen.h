//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_MOVEGEN_H
#define MCTS_CHECKERS_CPU_MOVEGEN_H

#include <vector>

#include "bit_operations.cuh"
#include "board_infra.h"
#include "constants.h"
#include "move.h"

#if WIN32
#include <stdlib.h>
inline size_t byte_swap64(const size_t x) { return _byteswap_uint64(x); }
#else
#include <byteswap.h>
inline size_t byte_swap64(const size_t x) { return __bswap_64(x); }
#endif

std::vector<Move> generateAllPossibleMoves(const Board &t_board, Colour t_color);

// Pawns
size_t getPawnsQuietMask(size_t t_moversPawns, size_t t_emptyFiles, Colour t_moversColour);

// Kings move and attack mask creation functions
size_t diagonalKingMask(size_t t_boardState, int t_kingSquare);
size_t antiDiagonalKingMask(size_t t_boardState, int t_kingSquare);
size_t bothDiagonalsKingMask(size_t t_boardState, int t_kingSquare);
size_t getKingsAttackMask(size_t t_kingsMask, size_t t_boardState, size_t t_opponentsPieces);

// Kings moves creation functions
void createAllKingsAttacks(std::vector<Move> &t_allMoves,
                           size_t             t_kingsMask,
                           size_t             t_boardState,
                           size_t             t_opponentPieces);
void recursiveCreateAllKingsAttacks(std::vector<Move> &t_allMoves,
                                    std::vector<int>  &t_attackPath,
                                    int                t_kingIdx,
                                    size_t             t_boardState,
                                    size_t             t_opponentPieces,
                                    size_t             t_originalStartingPositionMask,
                                    size_t             t_currentVictims);

void createAllKingsQuietMoves(std::vector<Move> &t_allMoves, size_t t_kingsMask, size_t t_boardState);

// Pawns moves creation functions
void createAllPawnsAttacks(std::vector<Move> &t_allMoves,
                           size_t             t_attackersMask,
                           size_t             t_opponentPieces,
                           size_t             t_emptyFiles,
                           Colour             t_attackersColour);
void recursiveCreatePawnsAttacks(std::vector<Move> &t_allMoves,
                                 std::vector<int>  &t_currentPath,
                                 int                t_idx,
                                 size_t             t_opponentPieces,
                                 size_t             t_emptyFiles,
                                 size_t             t_currentVictimsMask,
                                 size_t             t_originalStartingPositionMask,
                                 Colour             t_attackerColour);
void createAllPawnsQuietMoves(std::vector<Move> &t_allMoves,
                              size_t             t_moversMask,
                              size_t             t_emptyFiles,
                              Colour             t_moversColour);
void createOnePawnQuietMoves(std::vector<Move> &t_allMoves, int t_idx, size_t t_emptyFiles, Colour t_moversColour);
#endif // MCTS_CHECKERS_CPU_MOVEGEN_H
