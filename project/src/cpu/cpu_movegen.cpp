//
// Created by chefxx on 11.01.2026.
//

#include "cpu_movegen.h"

std::vector<Move> generateAllPossibleMoves(const Board &t_board, const Colour t_color)
{
    std::vector<Move> result;
    // t_colour represents a player that is currently
    // on the move, I do not want to think of this func
    // just form one perspective

    // remember to correctly initialize all variables!
    const size_t pawns = t_board.pawns[t_color];
    // TODO: const size_t kings = t_board.m_kings[t_color];
    const size_t pieces = pawns;
    const size_t opponent_pawns = t_board.pawns[1 - t_color];
    // TODO: const size_t opponent_kings = t_board.m_kings[1 - t_color];
    const size_t opponent_pieces = opponent_pawns;
    const size_t empty = ~(pieces | opponent_pieces);

    if (const size_t pawns_attackers = getPawnsAttackMask(pawns, opponent_pieces, empty)) {
        // if there is a move that is a jump/attack we have to do this move
        createAllPawnsAttacks(result, pawns_attackers, opponent_pieces, empty, TODO);
    }
    else {
        // if there is no attack possibility we can think of sliding moves
        const size_t pawns_movers = getPawnsMovesMask(pawns, empty, t_color);
        createAllPawnsMoves(result, pawns_movers, empty, t_color);
    }
    return result;
}

size_t getPawnsAttackMask(const size_t t_attackerPawns, const size_t t_opponentPieces, const size_t t_emptyFiles)
{
    const size_t attacks_upRight   = (((t_attackerPawns & NOT_FILE_G & NOT_FILE_H) << 9 & t_opponentPieces) << 9 & t_emptyFiles) >> 18;
    const size_t attacks_downRight = (((t_attackerPawns & NOT_FILE_G & NOT_FILE_H) >> 7 & t_opponentPieces) >> 7 & t_emptyFiles) << 14;
    const size_t attacks_upLeft    = (((t_attackerPawns & NOT_FILE_A & NOT_FILE_B) << 7 & t_opponentPieces) << 7 & t_emptyFiles) >> 14;
    const size_t attacks_downLeft  = (((t_attackerPawns & NOT_FILE_A & NOT_FILE_B) >> 9 & t_opponentPieces) >> 9 & t_emptyFiles) << 18;
    return attacks_upRight | attacks_downRight | attacks_upLeft | attacks_downLeft;
}

size_t getPawnsMovesMask(const size_t t_moversPawnsMask, const size_t t_emptyFiles, const Colour t_moversColour)
{
    // It is assumed that program "sees" the board from white perspective
    // so for white pawns only "up" slides are permitted, meaning
    // that for black only "down" slides are permitted.

    const size_t move_upRight   = canMove[t_moversColour][UP_RIGHT] * ((t_moversPawnsMask & NOT_FILE_H) << 9 & t_emptyFiles) >> 9;
    const size_t move_upLeft    = canMove[t_moversColour][UP_LEFT] * ((t_moversPawnsMask & NOT_FILE_A) << 7 & t_emptyFiles) >> 7;
    const size_t move_downRight = canMove[t_moversColour][DOWN_RIGHT] * ((t_moversPawnsMask & NOT_FILE_H) >> 7 & t_emptyFiles) << 7;
    const size_t move_downLeft  = canMove[t_moversColour][DOWN_LEFT] * ((t_moversPawnsMask & NOT_FILE_A) >> 9 & t_emptyFiles) << 9;
    return move_upRight | move_upLeft | move_downRight | move_downLeft;
}

void createAllPawnsAttacks(std::vector<Move> &t_allMoves,
                           const size_t       t_attackersMask,
                           const size_t       t_opponentPieces,
                           const size_t       t_emptyFiles,
                           const Colour       t_attackersColour)
{
    size_t maskCopy = t_attackersMask;
    while (maskCopy) {
        const int attacker_idx = popLsb(maskCopy);
        recursiveCreatePawnsAttacks(t_allMoves, attacker_idx, t_opponentPieces, t_emptyFiles, TODO, TODO);
    }
}

void recursiveCreatePawnsAttacks(std::vector<Move> &t_allMoves,
                                 const int          t_idx,
                                 const size_t       t_opponentPieces,
                                 const size_t       t_emptyFiles,
                                 const size_t       t_currentVictimsMask,
                                 const Colour       t_attackerColour)
{
    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        // if in dir we have opponent piece
        const size_t attack_mask =
            canMove[t_attackerColour][dir] * t_opponentPieces & globalTables.NeighbourTable[t_idx][dir];
        // if in dir we have a free file to jump into
        const size_t jump_mask =
            canMove[t_attackerColour][dir] * globalTables.NeighbourTable[t_idx][dir] & t_emptyFiles;
        if (jump_mask && attack_mask) {
            // TODO: update masks
            const size_t new_empty = t_emptyFiles;
            const size_t new_opponents = t_opponentPieces;
            const size_t new_attacks = t_currentVictimsMask | attack_mask;
            recursiveCreatePawnsAttacks(t_allMoves, t_idx + idxUpdate[dir], new_opponents, new_empty, new_attacks, t_attackerColour);
        }
    }
    // TODO: add new move to vector

}

void createAllPawnsMoves(std::vector<Move> &t_allMoves,
                        const size_t       t_moversMask,
                        const size_t       t_emptyFiles,
                        const Colour       t_moversColour)
{
    size_t maskCopy = t_moversMask;
    while (maskCopy) {
        const int mover_idx = popLsb(maskCopy);
        createOnePawnMoves(t_allMoves, mover_idx, t_emptyFiles, t_moversColour);
    }
}

void createOnePawnMoves(std::vector<Move> &t_allMoves,
                        const int          t_idx,
                        const size_t       t_emptyFiles,
                        const Colour       t_moversColour)
{
    // check directions of moves and create them
    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        // ReSharper disable once CppTooWideScope
        const size_t move_mask = canMove[t_moversColour][dir] * globalTables.NeighbourTable[t_idx][dir] & t_emptyFiles;
        if (move_mask) {
            const auto move = Move(1ULL << t_idx, move_mask);
            t_allMoves.push_back(move);
        }
    }
}
