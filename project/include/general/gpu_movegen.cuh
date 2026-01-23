//
// Created by chefxx on 11.01.2026.
//

#ifndef GPU_MOVE_GEN_CUH
#define GPU_MOVE_GEN_CUH

#include "gpu_board.cuh"
#include "common.cuh"

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

__device__ __forceinline__ size_t kings_attack_mask(size_t t_kingsMask, const size_t t_boardState, const size_t t_opponentsPieces)
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

__device__ __forceinline__

__device__ __forceinline__ size_t generate_random_move(const GPU_Board &t_board, const Colour t_colour)
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
    const size_t possible_kings_mask = kings_attack_mask(kings, all_board_pieces, opponent_pieces);

    return possible_pawns_mask | possible_kings_mask;
}




#endif
