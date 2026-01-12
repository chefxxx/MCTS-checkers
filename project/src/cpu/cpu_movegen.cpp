//
// Created by chefxx on 11.01.2026.
//

#include "cpu_movegen.h"

// generate all possible jumps first
// if a jump is possible from given position only jump can be made
// how to keep track of it???
// bitmask with other pieces and then check

std::vector<Move> generateAllPossibleMoves(const Board &t_board, const Colour t_color)
{
    std::vector<Move> result;
    // t_colour represents a player that is currently
    // on the move, I do not want to think of this func
    // just form one perspective

    // remember to correctly initialize all variables!
    const size_t pawns = t_board.m_pawns[t_color];
    // TODO: const size_t kings = t_board.m_kings[t_color];
    const size_t pieces = pawns;
    const size_t opponent_pawns = t_board.m_pawns[1 - t_color];
    // TODO: const size_t opponent_kings = t_board.m_kings[1 - t_color];
    const size_t opponent_pieces = opponent_pawns;
    const size_t empty = ~(pieces | opponent_pieces);

    // ReSharper disable once CppTooWideScope
    const size_t pawns_attackers = getPawnsAttackMask(pawns, opponent_pieces, empty);
    if (pawns_attackers) {
        // if there is a move that is a jump/attack we have to do this move

    }
    else {
        // if there is no attack possibility we can think of sliding moves
        const size_t pawns_movers = getPawnsMovesMask(pawns, empty, t_color);
        const auto pawns_slide_moves = createAllPawnMoves(pawns_movers, empty, t_color);
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

    const size_t move_upRight   = canMove[t_moversColour][0] * ((t_moversPawnsMask & NOT_FILE_H) << 9 & t_emptyFiles) >> 9;
    const size_t move_upLeft    = canMove[t_moversColour][1] * ((t_moversPawnsMask & NOT_FILE_A) << 7 & t_emptyFiles) >> 7;
    const size_t move_downRight = canMove[t_moversColour][2] * ((t_moversPawnsMask & NOT_FILE_H) >> 7 & t_emptyFiles) << 7;
    const size_t move_downLeft  = canMove[t_moversColour][3] * ((t_moversPawnsMask & NOT_FILE_A) >> 9 & t_emptyFiles) << 9;
    return move_upRight | move_upLeft | move_downRight | move_downLeft;
}

std::vector<Move> createAllPawnMoves(const size_t t_moversMask, const size_t t_emptyFiles, const Colour t_moversColour)
{
    std::vector<Move> result;
    size_t maskCopy = t_moversMask;
    while (maskCopy) {
        const int mover_idx = popLsb(maskCopy);
        const auto moves = createOnePawnsMove(mover_idx, t_emptyFiles, t_moversColour);
    }
    return result;
}

std::vector<Move> createOnePawnsMove(const int t_idx, const size_t t_emptyFiles, const Colour t_moversColour)
{
    std::vector<Move> result;
    // check directions of moves and create them
    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        // ReSharper disable once CppTooWideScope
        const size_t move_mask = canMove[t_moversColour][dir] * globalTables.NeighbourTable[t_idx][dir] & t_emptyFiles;
        if (move_mask) {
            const auto move = Move(1ull << t_idx, move_mask);
            result.push_back(move);
        }
    }
    return result;
}
