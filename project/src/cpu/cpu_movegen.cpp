//
// Created by chefxx on 11.01.2026.
//

#include "cpu_movegen.h"

// generate all possible jumps first
// if a jump is possible from given position only jump can be made
// how to keep track of it???
// bitmask with other pieces and then check

size_t getPawnsAttackMask(const size_t t_attacker, const size_t t_opponent)
{
    const size_t empty             = ~(t_attacker | t_opponent);
    const size_t attacks_upRight   = (((t_attacker & NOT_FILE_G & NOT_FILE_H) << 9 & t_opponent) << 9 & empty) >> 18;
    const size_t attacks_downRight = (((t_attacker & NOT_FILE_G & NOT_FILE_H) >> 7 & t_opponent) >> 7 & empty) << 14;
    const size_t attacks_upLeft    = (((t_attacker & NOT_FILE_A & NOT_FILE_B) << 7 & t_opponent) << 7 & empty) >> 14;
    const size_t attacks_downLeft  = (((t_attacker & NOT_FILE_A & NOT_FILE_B) >> 9 & t_opponent) >> 9 & empty) << 18;
    return attacks_upRight | attacks_downRight | attacks_upLeft | attacks_downLeft;
}

size_t getPawnsMovesMask(const Colour t_onMoveColour, const size_t t_onMove, const size_t t_opponent)
{
    const size_t empty = ~(t_onMove | t_opponent);

    // It is assumed that program "sees" the board from white perspective
    // so for white pawns only "up" slides are permitted, meaning
    // that for black only "down" slides are permitted.

    const size_t move_upRight   = canMove[t_onMoveColour][0] * ((t_onMove & NOT_FILE_H) << 9 & empty) >> 9;
    const size_t move_upLeft    = canMove[t_onMoveColour][1] * ((t_onMove & NOT_FILE_A) << 7 & empty) >> 7;
    const size_t move_downRight = canMove[t_onMoveColour][2] * ((t_onMove & NOT_FILE_H) >> 7 & empty) << 7;
    const size_t move_downLeft  = canMove[t_onMoveColour][3] * ((t_onMove & NOT_FILE_A) >> 9 & empty) << 9;
    return move_upRight | move_upLeft | move_downRight | move_downLeft;
}


// std::vector<Move> getNormalMoves(const Board &board, Colour t_color) {}
