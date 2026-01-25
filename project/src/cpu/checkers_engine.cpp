//
// Created by chefxx on 16.01.2026.

#include "checkers_engine.h"

#include "cpu_movegen.h"
#include "lookup_tables.h"

/**
 * @brief This function has to take care of the whole board state
 * during the move transition, that is, it is responsible for:
 *
 * 1. Tracking the pieces bitboards,
 * 2. Tracking the king moves without a take,
 * 3. Updating the board state.
 *
 * @param t_board
 * @param t_move
 * @param t_colour
 * @return
 */
std::optional<Board> applyMove(const Board &t_board, const Move &t_move, const Colour t_colour)
{
    Board board_copy = t_board;

    // Pawn moved
    if (t_move.from_mask & t_board.pawns[t_colour]) {
        board_copy.pawns[t_colour] &= ~t_move.from_mask;
        board_copy.pawns[t_colour] |= t_move.to_mask & promotion[t_colour] ? 0ULL : t_move.to_mask;
        board_copy.kings[t_colour] |= t_move.to_mask & promotion[t_colour] ? t_move.to_mask : 0ULL;
    }
    else {
        // King moved
        board_copy.kings[t_colour] &= ~t_move.from_mask;
        board_copy.kings[t_colour] |= t_move.to_mask;
    }

    // update kings quiet moves
    board_copy.kings_quiet_moves = (t_move.from_mask & t_board.kings[t_colour]) > 0 && t_move.captures_mask == 0ULL
                                     ? board_copy.kings_quiet_moves + 1
                                     : 0;

    // update opponents pieces with captures_mask, since it is 0ULL
    // if no attack move, then we can safely do it
    board_copy.kings[1 - t_colour] &= ~t_move.captures_mask;
    board_copy.pawns[1 - t_colour] &= ~t_move.captures_mask;

    return board_copy;
}


/**
 *
 * @param t_board
 * @param t_playerWhoJustMadeAMove
 * @return
 */
GameState checkEndOfGameConditions(const Board &t_board, const Colour t_playerWhoJustMadeAMove)
{
    if (t_board.kings_quiet_moves == DRAW_LIMIT) {
        return DRAW;
    }

    if (!(t_board.pawns[1 - t_playerWhoJustMadeAMove] | t_board.kings[1 - t_playerWhoJustMadeAMove])) {
        return WON;
    }

    if (const auto moves = generateAllPossibleMoves(t_board, static_cast<Colour>(1 - t_playerWhoJustMadeAMove));
        moves.empty()) {
        return WON;
    }
    return CONTINUES;
}