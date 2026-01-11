//
// Created by chefxx on 11.01.2026.
//

#include "cpu_movegen.h"

std::vector<Move> generateAllPossibleMoves(const Board &board)
{

}

size_t getNormalMoves(const Board& t_board, const Colour t_color)
{
    size_t mask = 0ull;
    if (t_color == white) {
        mask |= (t_board.m_pawns[white] & NOT_FILE_A) << 7;
        mask |= (t_board.m_pawns[white] & NOT_FILE_H) << 9;
    }
    if (t_color == black) {
        mask |= (t_board.m_pawns[black] & NOT_FILE_H) >> 7;
        mask |= (t_board.m_pawns[black] & NOT_FILE_A) >> 9;
    }
    return mask;
}

