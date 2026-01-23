//
// Created by chefxx on 11.01.2026.
//

#ifndef GPU_MOVE_GEN_CUH
#define GPU_MOVE_GEN_CUH

#include <cassert>

#include "bit_operations.cuh"
#include "common.cuh"
#include "gpu_board.cuh"
#include "gpu_move.cuh"

// -------------------------------------
// Constant memory variables definitions
// -------------------------------------
__constant__ inline Direction d_diffToDir[128];
__constant__ inline size_t    d_NeighbourTable[64][4];
__constant__ inline size_t    d_JumpTable[64][4];
__constant__ inline size_t    d_diagonalMaskEx[64];
__constant__ inline size_t    d_anitDiagonalMaskEx[64];
__constant__ inline int       d_canMove[2][4];

__device__ __forceinline__ size_t get_pawns_quiet_gpu(const size_t t_moversPawns,
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
__device__ __forceinline__ size_t diagonal_king_mask_gpu(const size_t t_boardState, const int t_kingSquare)
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
__device__ __forceinline__ size_t anti_diagonal_king_mask_gpu(const size_t t_boardState, const int t_kingSquare)
{
    size_t forward = t_boardState & d_anitDiagonalMaskEx[t_kingSquare];
    size_t reverse = __nv_bswap64(forward);
    forward -= 1ULL << t_kingSquare;
    reverse -= __nv_bswap64(1ULL << t_kingSquare);
    forward ^= __nv_bswap64(reverse);
    forward &= d_anitDiagonalMaskEx[t_kingSquare];
    return forward;
}

__device__ __forceinline__ size_t both_diagonals_king_mask_gpu(const size_t t_boardState, const int t_kingSquare)
{
    return diagonal_king_mask_gpu(t_boardState, t_kingSquare) | anti_diagonal_king_mask_gpu(t_boardState, t_kingSquare);
}

__device__ __forceinline__ size_t kings_attack_mask_gpu(size_t       t_kingsMask,
                                                        const size_t t_boardState,
                                                        const size_t t_opponentsPieces)
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

__device__ __forceinline__ size_t kings_quiet_mask_gpu(size_t t_kingsMask, const size_t t_boardState)
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

__device__ __forceinline__ int pick_random_bit(size_t t_mask, curandState *t_state)
{
    const int count = popCount(t_mask);
    assert(count != 0);

    const int target = curand(t_state) % count;
    for (int i = 0; i < target; ++i) {
        t_mask &= t_mask - 1;
    }

    return getLsb(t_mask);
}

__device__ __forceinline__ GPU_Move make_king_attack(curandState *t_state, const int t_idx)
{
    // TODO: implement this func
    return GPU_Move(1ULL, 1ULL, 1ULL);
}

__device__ __forceinline__ GPU_Move make_pawn_attack(curandState *t_state, const int t_idx, const size_t t_opponentPieces, const size_t t_empty)
{
    size_t current_victims = 0ULL;
    int curr_pos = t_idx;
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

        const int dest = pick_random_bit(attacks_destinations, t_state);
        const int diff = dest - curr_pos;
        const Direction dir = d_diffToDir[diff + 64];
        current_victims |= d_NeighbourTable[t_idx][dir];
        curr_pos = dest;
    }
    return GPU_Move(1ULL << t_idx, 1ULL << curr_pos, current_victims);
}

__device__ __forceinline__ GPU_Move make_king_quiet(curandState *t_state, const int t_idx, const size_t t_boardState, const size_t t_empty)
{
    const size_t move_mask  = both_diagonals_king_mask_gpu(t_boardState, t_idx) & t_empty; // & with empty
    assert(move_mask != 0);
    const int to = pick_random_bit(move_mask, t_state);
    return GPU_Move(1ULL << t_idx, 1ULL << to, 0ULL);
}

__device__ __forceinline__ GPU_Move make_pawn_quiet(curandState *t_state,
                                                    const int    t_idx,
                                                    const Colour t_colour,
                                                    const size_t t_empty)
{
    size_t move_mask = 0ULL;
#pragma unroll
    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        move_mask |= d_canMove[t_colour][dir] * d_NeighbourTable[t_idx][dir] & t_empty;
    }
    assert(move_mask != 0);
    const int to = pick_random_bit(move_mask, t_state);
    return GPU_Move(1ULL << t_idx, 1ULL << to, 0ULL);
}

__device__ __forceinline__ GPU_Move generate_random_move(curandState     *t_state,
                                                         const GPU_Board &t_board,
                                                         const Colour     t_colour)
{
    const size_t pawns            = t_board.pawns[t_colour];
    const size_t kings            = t_board.kings[t_colour];
    const size_t pieces           = pawns | kings;
    const size_t opponent_pawns   = t_board.pawns[1 - t_colour];
    const size_t opponent_kings   = t_board.kings[1 - t_colour];
    const size_t opponent_pieces  = opponent_pawns | opponent_kings;
    const size_t all_board_pieces = pieces | opponent_pieces;
    const size_t empty            = ~all_board_pieces;

    const size_t possible_pawns_mask = getPawnsAttackMask(pawns, opponent_pieces, empty);
    const size_t possible_kings_mask = kings_attack_mask_gpu(kings, all_board_pieces, opponent_pieces);

    // attacks
    if (const size_t possible_attack_mask = possible_pawns_mask | possible_kings_mask) {
        const int found_idx = pick_random_bit(possible_attack_mask, t_state);
        if (const bool is_king = 1ULL << found_idx & kings) {
            return make_king_attack(t_state, found_idx);
        }
        return make_pawn_attack(t_state, found_idx);
    }

    // moves
    const size_t possible_pawns_quiet = get_pawns_quiet_gpu(pawns, empty, t_colour);
    const size_t possible_kings_quiet = kings_quiet_mask_gpu(kings, all_board_pieces);
    const size_t possible_quiet_mask  = possible_kings_quiet | possible_pawns_quiet;

    const int found_idx = pick_random_bit(possible_quiet_mask, t_state);
    if (const bool is_king = 1ULL << found_idx & kings) {
        return make_king_quiet(t_state, found_idx, all_board_pieces, empty);
    }
    return make_pawn_quiet(t_state, found_idx, t_colour, empty);
}


#endif
