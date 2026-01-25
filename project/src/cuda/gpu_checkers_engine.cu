#include "common.cuh"
#include "gpu_board.cuh"
#include "gpu_checkers_engine.cuh"
#include "gpu_move.cuh"
#include "gpu_movegen.cuh"
#include "helper_cuda.h"

// -------------------------------------
// Constant memory variables definitions
// -------------------------------------
__constant__ size_t d_promotion[2];

void init_promotion_const_mem()
{
    checkCudaErrors(cudaMemcpyToSymbol(d_promotion, promotion, 2 * sizeof(size_t)));
}

__device__ GPU_Board apply_move_gpu(const GPU_Board &t_board, const GPU_Move &t_move, const Colour t_colour)
{
    GPU_Board board_copy = t_board;

    // Pawn moved
    if (t_move.from_mask & t_board.pawns[t_colour]) {
        board_copy.pawns[t_colour] &= ~t_move.from_mask;
        board_copy.pawns[t_colour] |= t_move.to_mask & d_promotion[t_colour] ? 0ULL : t_move.to_mask;
        board_copy.kings[t_colour] |= t_move.to_mask & d_promotion[t_colour] ? t_move.to_mask : 0ULL;
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

__device__  GameState check_end_of_game_conditions(const GPU_Board &t_board, const Colour t_playerWhoJustMoved)
{
    if (t_board.kings_quiet_moves == DRAW_LIMIT) {
        return DRAW;
    }

    if (!(t_board.pawns[1 - t_playerWhoJustMoved] | t_board.kings[1 - t_playerWhoJustMoved])) {
        return WON;
    }

    return CONTINUES;
}