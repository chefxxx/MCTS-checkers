//
// Created by chefxx on 24.01.2026.
//

#ifndef GPU_CHECKERS_ENGINE_CUH
#define GPU_CHECKERS_ENGINE_CUH

#include <cuda_runtime_api.h>

#include "common.cuh"
#include "gpu_board.cuh"
#include "gpu_move.cuh"
#include "gpu_movegen.cuh"

__device__ __forceinline__ void apply_move_gpu(GPU_Board &t_board, const GPU_Move &t_move, const Colour t_colour)
{
    // Pawn moved
    if (t_move.from_mask & t_board.pawns[t_colour]) {
        t_board.pawns[t_colour] &= ~t_move.from_mask;
        t_board.pawns[t_colour] |= t_move.to_mask & promotion[t_colour] ? 0ULL : t_move.to_mask;
        t_board.kings[t_colour] |= t_move.to_mask & promotion[t_colour] ? t_move.to_mask : 0ULL;
    }
    else {
        // King moved
        t_board.kings[t_colour] &= ~t_move.from_mask;
        t_board.kings[t_colour] |= t_move.to_mask;
    }

    // update kings quiet moves
    t_board.kings_quiet_moves = (t_move.from_mask & t_board.kings[t_colour]) > 0 && t_move.captures_mask > 0
                                     ? t_board.kings_quiet_moves + 1
                                     : 0;

    // update opponents pieces with captures_mask, since it is 0ULL
    // if no attack move, then we can safely do it
    t_board.kings[1 - t_colour] &= ~t_move.captures_mask;
    t_board.pawns[1 - t_colour] &= ~t_move.captures_mask;
}

__device__ __forceinline__ GameState check_end_of_game_conditions(const GPU_Board &t_board, const Colour t_playerWhoJustMoved)
{
    if (t_board.kings_quiet_moves == DRAW_LIMIT) {
        return DRAW;
    }

    if (!(t_board.pawns[1 - t_playerWhoJustMoved] | t_board.kings[1 - t_playerWhoJustMoved])) {
        return WON;
    }

    const size_t pawns            = t_board.pawns[1 - t_playerWhoJustMoved];
    const size_t kings            = t_board.kings[1 - t_playerWhoJustMoved];
    const size_t pieces           = pawns | kings;
    const size_t opponent_pawns   = t_board.pawns[t_playerWhoJustMoved];
    const size_t opponent_kings   = t_board.kings[t_playerWhoJustMoved];
    const size_t opponent_pieces  = opponent_pawns | opponent_kings;
    const size_t all_board_pieces = pieces | opponent_pieces;
    const size_t empty            = ~all_board_pieces;

    const size_t possible_pawns_attack = getPawnsAttackMask(pawns, opponent_pieces, empty);
    const size_t possible_kings_attack = kings_attack_mask_gpu(kings, all_board_pieces, opponent_pieces);
    const size_t possible_pawns_quiet  = get_pawns_quiet_gpu(pawns, empty, static_cast<Colour>(1 - t_playerWhoJustMoved));
    const size_t possible_kings_quiet  = kings_quiet_mask_gpu(kings, all_board_pieces);

    if (const size_t any_move_mask =
            possible_kings_attack | possible_kings_quiet | possible_pawns_quiet | possible_pawns_attack;
        !any_move_mask)
        return WON;
    return CONTINUES;
}

#endif