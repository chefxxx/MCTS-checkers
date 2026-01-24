//
// Created by chefxx on 23.01.2026.
//

#include <cassert>
#include <helper_cuda.h>

#include "bit_operations.cuh"
#include "common.cuh"
#include "gpu_movegen.cuh"
#include "lookup_tables.h"

// -------------------------------------
// Constant memory variables definitions
// -------------------------------------
__constant__ Direction d_diffToDir[128];
__constant__ size_t    d_NeighbourTable[64][4];
__constant__ size_t    d_JumpTable[64][4];
__constant__ size_t    d_diagonalMaskEx[64];
__constant__ size_t    d_anitDiagonalMaskEx[64];
__constant__ int       d_canMove[2][4];
__constant__ size_t    d_rayMask[64][4];

__host__ void prepare_gpu_const_mem()
{
    checkCudaErrors(cudaMemcpyToSymbol(d_diffToDir, globalTables.diffToDir, 128 * sizeof(Direction)));
    checkCudaErrors(cudaMemcpyToSymbol(d_NeighbourTable, globalTables.NeighbourTable, 256 * sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(d_JumpTable, globalTables.JumpTable, 256 * sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(d_diagonalMaskEx, globalTables.diagonalMaskEx, 64 * sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(d_anitDiagonalMaskEx, globalTables.anitDiagonalMaskEx, 64 * sizeof(size_t)));
    checkCudaErrors(cudaMemcpyToSymbol(d_canMove, canMove, 2 * 4 * sizeof(int)));
    checkCudaErrors(cudaMemcpyToSymbol(d_rayMask, globalTables.rayMasks, 256 * sizeof(size_t)));
}

__device__ size_t get_pawns_quiet_gpu(const size_t t_moversPawns,
                                      const size_t t_emptyFiles,
                                      const Colour t_moversColour)
{
    // It is assumed that program "sees" the board from white perspective
    // so for white pawns only "up" slides are permitted, meaning
    // that for black only "down" slides are permitted.
    const size_t move_upRight =
        d_canMove[t_moversColour][UP_RIGHT] * ((t_moversPawns & NOT_FILE_H) << 9 & t_emptyFiles) >> 9;
    const size_t move_upLeft =
        d_canMove[t_moversColour][UP_LEFT] * ((t_moversPawns & NOT_FILE_A) << 7 & t_emptyFiles) >> 7;
    const size_t move_downRight =
        d_canMove[t_moversColour][DOWN_RIGHT] * ((t_moversPawns & NOT_FILE_H) >> 7 & t_emptyFiles) << 7;
    const size_t move_downLeft =
        d_canMove[t_moversColour][DOWN_LEFT] * ((t_moversPawns & NOT_FILE_A) >> 9 & t_emptyFiles) << 9;
    return move_upRight | move_upLeft | move_downRight | move_downLeft;
}

// https://www.chessprogramming.org/Hyperbola_Quintessence
__device__ size_t diagonal_king_mask_gpu(const size_t t_boardState, const int t_kingSquare)
{
    size_t forward = t_boardState & d_diagonalMaskEx[t_kingSquare];
    size_t reverse = __nv_bswap64(forward);
    forward -= 1ULL << t_kingSquare;
    reverse -= __nv_bswap64(1ULL << t_kingSquare);
    forward ^= __nv_bswap64(reverse);
    forward &= d_diagonalMaskEx[t_kingSquare];
    return forward;
}

// https://www.chessprogramming.org/Hyperbola_Quintessence
__device__ size_t anti_diagonal_king_mask_gpu(const size_t t_boardState, const int t_kingSquare)
{
    size_t forward = t_boardState & d_anitDiagonalMaskEx[t_kingSquare];
    size_t reverse = __nv_bswap64(forward);
    forward -= 1ULL << t_kingSquare;
    reverse -= __nv_bswap64(1ULL << t_kingSquare);
    forward ^= __nv_bswap64(reverse);
    forward &= d_anitDiagonalMaskEx[t_kingSquare];
    return forward;
}

__device__ size_t both_diagonals_king_mask_gpu(const size_t t_boardState, const int t_kingSquare)
{
    return diagonal_king_mask_gpu(t_boardState, t_kingSquare) | anti_diagonal_king_mask_gpu(t_boardState, t_kingSquare);
}

__device__ size_t kings_attack_mask_gpu(size_t t_kingsMask, const size_t t_boardState, const size_t t_opponentsPieces)
{
    const size_t empty        = ~t_boardState;
    size_t       result_kings = 0ULL;
    while (t_kingsMask) {
        const int    k_idx            = popLsb(t_kingsMask);
        const size_t mask             = both_diagonals_king_mask_gpu(t_boardState, k_idx);
        size_t       possible_victims = mask & t_opponentsPieces;
        while (possible_victims) {
            const int v_idx = popLsb(possible_victims);
            const int diff  = v_idx - k_idx;
            if (const Direction dir = d_diffToDir[diff + 64]; d_NeighbourTable[v_idx][dir] & empty) {
                result_kings |= 1ULL << k_idx;
                break;
            }
        }
    }
    return result_kings;
}

__device__ size_t kings_quiet_mask_gpu(size_t t_kingsMask, const size_t t_boardState)
{
    size_t result_kings = 0ULL;
    while (t_kingsMask) {
        const int k_idx = popLsb(t_kingsMask);
        if (size_t moves_mask = both_diagonals_king_mask_gpu(t_boardState, k_idx) & ~t_boardState) {
            result_kings |= 1ULL << k_idx;
        }
    }
    return result_kings;
}

__device__ int pick_random_bit(size_t t_mask, curandState *t_state)
{
    const int count = popCount(t_mask);
    assert(count != 0);

    const int target = static_cast<int>(curand(t_state) % count);
    for (int i = 0; i < target; ++i) {
        t_mask &= t_mask - 1;
    }

    return getLsb(t_mask);
}

__device__ GPU_Move make_king_attack(curandState *t_state,
                                     const int    t_idx,
                                     size_t       t_boardState,
                                     const size_t t_opponentPieces)
{
    size_t current_victims = 0ULL;
    int    curr_pos        = t_idx;
    while (true) {
        size_t real_victims = 0ULL;
        size_t blockers = both_diagonals_king_mask_gpu(t_boardState, curr_pos) & t_opponentPieces & ~current_victims;
        while (blockers) {
            const int       v_idx = popLsb(blockers);
            const int       diff  = v_idx - curr_pos;
            const Direction dir   = d_diffToDir[diff + 64];
            real_victims |= d_NeighbourTable[v_idx][dir] & ~t_boardState ? 1ULL << v_idx : 0ULL;
        }
        if (real_victims == 0)
            break;
        const int       chosen_victim = pick_random_bit(real_victims, t_state);
        const int       diff          = chosen_victim - curr_pos;
        const Direction dir           = d_diffToDir[diff + 64];
        size_t          landing_mask =
            d_rayMask[chosen_victim][dir] & both_diagonals_king_mask_gpu(t_boardState, chosen_victim) & ~t_boardState;
        size_t attacks_possibilities_after_landing =
            kings_attack_mask_gpu(landing_mask, t_boardState, t_opponentPieces & ~(1ULL << chosen_victim));
        landing_mask = attacks_possibilities_after_landing ? attacks_possibilities_after_landing : landing_mask;

        // reset the bit that kings has gone from
        t_boardState = t_boardState & ~(1ULL << curr_pos);
        curr_pos     = pick_random_bit(landing_mask, t_state);

        // update the new position of the king
        t_boardState |= 1ULL << curr_pos;

        // update victims mask
        current_victims |= 1ULL << chosen_victim;
    }

    return GPU_Move{1ULL << t_idx, 1ULL << curr_pos, current_victims};
}

__device__ GPU_Move make_pawn_attack(curandState *t_state,
                                     const int    t_idx,
                                     const size_t t_opponentPieces,
                                     size_t       t_empty)
{
    size_t current_victims = 0ULL;
    int    curr_pos        = t_idx;
    while (true) {
        size_t attacks_destinations = 0ULL;

#pragma unroll
        for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
            const size_t victim_mask = d_NeighbourTable[curr_pos][dir] & t_opponentPieces & ~current_victims;
            const size_t jump_mask   = d_JumpTable[curr_pos][dir] & t_empty;
            attacks_destinations |= victim_mask && jump_mask ? jump_mask : 0ULL;
        }
        if (!attacks_destinations)
            break;

        const int       dest = pick_random_bit(attacks_destinations, t_state);
        const int       diff = dest - curr_pos;
        const Direction dir  = d_diffToDir[diff + 64];
        current_victims |= d_NeighbourTable[curr_pos][dir];
        t_empty |= 1ULL << curr_pos;
        curr_pos = dest;
        t_empty &= ~(1ULL << curr_pos);
    }
    return GPU_Move{1ULL << t_idx, 1ULL << curr_pos, current_victims};
}

__device__ GPU_Move make_king_quiet(curandState *t_state,
                                    const int    t_idx,
                                    const size_t t_boardState,
                                    const size_t t_empty)
{
    const size_t move_mask = both_diagonals_king_mask_gpu(t_boardState, t_idx) & t_empty; // & with empty
    assert(move_mask != 0);
    const int to = pick_random_bit(move_mask, t_state);
    return GPU_Move{1ULL << t_idx, 1ULL << to, 0ULL};
}

__device__ GPU_Move make_pawn_quiet(curandState *t_state, const int t_idx, const Colour t_colour, const size_t t_empty)
{
    size_t move_mask = 0ULL;
#pragma unroll
    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        move_mask |= d_canMove[t_colour][dir] * d_NeighbourTable[t_idx][dir] & t_empty;
    }
    assert(move_mask != 0);
    const int to = pick_random_bit(move_mask, t_state);
    return GPU_Move{1ULL << t_idx, 1ULL << to, 0ULL};
}

__device__ GPU_Move generate_random_move(curandState *t_state, const GPU_Board &t_board, const Colour t_colour)
{
    const size_t pawns            = t_board.pawns[t_colour];
    const size_t kings            = t_board.kings[t_colour];
    const size_t pieces           = pawns | kings;
    const size_t opponent_pawns   = t_board.pawns[1 - t_colour];
    const size_t opponent_kings   = t_board.kings[1 - t_colour];
    const size_t opponent_pieces  = opponent_pawns | opponent_kings;
    const size_t all_board_pieces = pieces | opponent_pieces;
    const size_t empty            = ~all_board_pieces;

    const size_t possible_pawns_attack = getPawnsAttackMask(pawns, opponent_pieces, empty);
    const size_t possible_kings_attack = kings_attack_mask_gpu(kings, all_board_pieces, opponent_pieces);

    // attacks
    if (const size_t possible_attack_mask = possible_pawns_attack | possible_kings_attack) {
        const int  found_idx = pick_random_bit(possible_attack_mask, t_state);
        const bool is_king   = 1ULL << found_idx & kings;
        return is_king ? make_king_attack(t_state, found_idx, all_board_pieces, opponent_pieces)
                       : make_pawn_attack(t_state, found_idx, opponent_pieces, empty);
    }

    // moves
    const size_t possible_pawns_quiet = get_pawns_quiet_gpu(pawns, empty, t_colour);
    const size_t possible_kings_quiet = kings_quiet_mask_gpu(kings, all_board_pieces);
    const size_t possible_quiet_mask  = possible_kings_quiet | possible_pawns_quiet;
    const int  found_idx = pick_random_bit(possible_quiet_mask, t_state);
    const bool is_king   = 1ULL << found_idx & kings;
    return is_king ? make_king_quiet(t_state, found_idx, all_board_pieces, empty)
                   : make_pawn_quiet(t_state, found_idx, t_colour, empty);
}
