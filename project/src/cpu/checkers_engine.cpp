//
// Created by chefxx on 16.01.2026.
//

#include "checkers_engine.h"

Board applyMove(const Board &t_board, const Move &t_move, const Colour t_colour)
{
    Board board_copy = t_board;
    board_copy.kings[t_colour] &= ~t_move.from_mask;
    board_copy.pawns[t_colour] &= ~t_move.from_mask;
    board_copy.kings[1 - t_colour] &= ~t_move.captures_mask;
    board_copy.pawns[1 - t_colour] &= ~t_move.captures_mask;
    board_copy.kings[t_colour] |= t_move.to_mask;
    board_copy.pawns[t_colour] |= t_move.to_mask;
    return board_copy;
}

GameState checkEndOfGameConditions()
{
    return CONTINUES;
}