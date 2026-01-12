//
// Created by chefxx on 11.01.2026.
//

#include "cpu_movegen.h"

// size_t getNormalMoves(const Board& t_board, const Colour t_color)
// {
//     size_t mask = 0ull;
//     if (t_color == white) {
//         mask |= (t_board.m_pawns[white] & NOT_FILE_A) << 7;
//         mask |= (t_board.m_pawns[white] & NOT_FILE_H) << 9;
//     }
//     if (t_color == black) {
//         mask |= (t_board.m_pawns[black] & NOT_FILE_H) >> 7;
//         mask |= (t_board.m_pawns[black] & NOT_FILE_A) >> 9;
//     }
//     return mask;
// }

// std::vector<Move> generateAllPossibleMoves(const Board& t_board, Colour t_color)
// {
//     // generate all possible jumps first
//     // if a jump is possible from given position only jump can be made
//     // how to keep track of it???
//     // bitmask with other pieces and then check
//
// }

size_t getPawnsAttackMask(const size_t t_attacker, const size_t t_opponent)
{
    const size_t empty             = ~(t_attacker | t_opponent);
    const size_t attacks_upRight   = (((t_attacker & NOT_FILE_G & NOT_FILE_H) << 9 & t_opponent) << 9 & empty) >> 18;
    const size_t attacks_downRight = (((t_attacker & NOT_FILE_G & NOT_FILE_H) >> 7 & t_opponent) >> 7 & empty) << 14;
    const size_t attacks_upLeft    = (((t_attacker & NOT_FILE_A & NOT_FILE_B) << 7 & t_opponent) << 7 & empty) >> 14;
    const size_t attacks_downLeft  = (((t_attacker & NOT_FILE_A & NOT_FILE_B) >> 9 & t_opponent) >> 9 & empty) << 18;
    return attacks_upRight | attacks_downRight | attacks_upLeft | attacks_downLeft;
}


// std::vector<Move> getNormalMoves(const Board &board, Colour t_color) {}
