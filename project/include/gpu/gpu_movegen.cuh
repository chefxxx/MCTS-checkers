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
__constant__ inline size_t d_NeighbourTable[64][4];
__constant__ inline size_t d_diagonalMaskEx[64];
__constant__ inline size_t d_anitDiagonalMaskEx[64];

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

__device__ __forceinline__ size_t kings_attack_mask_gpu(size_t t_kingsMask, const size_t t_boardState, const size_t t_opponentsPieces)
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
            if (const Direction dir = d_diffToDir[diff + 64];
                d_NeighbourTable[v_idx][dir] & empty) {
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
        const int k_idx   = popLsb(t_kingsMask);
        if (size_t moves_mask = both_diagonals_king_mask_gpu(t_boardState, k_idx) & ~t_boardState) {
            result_kings |= 1ULL << k_idx;
        }
    }
    return result_kings;
}

__device__ __forceinline__ int pick_random_bit(size_t t_mask, curandState* t_state)
{
    const int count = popCount(t_mask);
    assert(count != 0);

    const int target = curand(t_state) % count;
    for (int i = 0; i < target; ++i) {
        t_mask &= t_mask - 1;
    }

    return getLsb(t_mask);
}

__device__ __forceinline__ GPU_Move make_king_attack(curandState* t_state, const size_t t_startKingMask)
{
    // TODO: implement this func
    return GPU_Move(1ULL, 1ULL, 1ULL);
}

__device__ __forceinline__ GPU_Move make_pawn_attack(curandState* t_state, const size_t t_startPawnMask)
{
    // TODO: implement this func
    return GPU_Move(1ULL, 1ULL, 1ULL);
}

__device__ __forceinline__ GPU_Move make_king_quiet(curandState* t_state, const size_t t_startKingMask)
{
    // TODO: implement this func
    return GPU_Move(1ULL, 1ULL, 1ULL);
}

__device__ __forceinline__ GPU_Move make_pawn_quiet(curandState* t_state, const size_t t_startPawnMask)
{
    // TODO: implement this func
    return GPU_Move(1ULL, 1ULL, 1ULL);
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
    if (const size_t possible_attack_mask = possible_pawns_mask | possible_kings_mask) {
        const int found_idx = pick_random_bit(possible_attack_mask, t_state);
        const size_t start_mask = 1ULL << found_idx;
        if (const bool is_king = start_mask & kings) {
            return make_king_attack(t_state, start_mask);
        }
        return make_pawn_attack(t_state, start_mask);
    }

    // moves
    const size_t possible_pawns_quiet = getPawnsQuietMovesMask(pawns, empty, t_colour);
    const size_t possible_kings_quiet = kings_quiet_mask_gpu(kings, all_board_pieces);
    const size_t possible_quiet_mask  = possible_kings_quiet | possible_pawns_quiet;

    const int found_idx = pick_random_bit(possible_quiet_mask, t_state);
    const size_t start_mask = 1ULL << found_idx;
    if (const bool is_king = start_mask & kings) {
        return make_king_quiet(t_state, start_mask);
    }
    return make_pawn_quiet(t_state, start_mask);
}




#endif
