//
// Created by chefxx on 23.01.2026.
//

#ifndef GPU_MOVE_GEN_CUH
#define GPU_MOVE_GEN_CUH

#include <curand_kernel.h>

#include "constants.h"
#include "gpu_board.cuh"
#include "gpu_move.cuh"


__host__ void prepare_gpu_const_mem();

__device__ size_t get_pawns_quiet_gpu(size_t t_moversPawns, size_t t_emptyFiles, Colour t_moversColour);

__device__ size_t diagonal_king_mask_gpu(size_t t_boardState, int t_kingSquare);

__device__ size_t anti_diagonal_king_mask_gpu(size_t t_boardState, int t_kingSquare);

__device__ size_t both_diagonals_king_mask_gpu(size_t t_boardState, int t_kingSquare);

__device__ size_t kings_attack_mask_gpu(size_t t_kingsMask, size_t t_boardState, size_t t_opponentsPieces);

__device__ size_t   kings_quiet_mask_gpu(size_t t_kingsMask, size_t t_boardState);
__device__ int      pick_random_bit(size_t t_mask, curandState *t_state);
__device__ GPU_Move make_king_attack(curandState *t_state, int t_idx, size_t t_boardState, size_t t_opponentPieces);
__device__ GPU_Move make_pawn_attack(curandState *t_state, int t_idx, size_t t_opponentPieces, size_t t_empty);
__device__ GPU_Move make_king_quiet(curandState *t_state, int t_idx, size_t t_boardState, size_t t_empty);
__device__ GPU_Move make_pawn_quiet(curandState *t_state, int t_idx, Colour t_colour, size_t t_empty);
__device__ GPU_Move generate_random_move(curandState *t_state, const GPU_Board &t_board, Colour t_colour);


#endif
