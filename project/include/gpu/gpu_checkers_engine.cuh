//
// Created by chefxx on 24.01.2026.
//

#ifndef GPU_CHECKERS_ENGINE_CUH
#define GPU_CHECKERS_ENGINE_CUH

#include "constants.h"
#include "gpu_board.cuh"
#include "gpu_move.cuh"

__device__ GPU_Board apply_move_gpu(const GPU_Board &t_board, const GPU_Move &t_move, Colour t_colour);
__device__ GameState check_end_of_game_conditions(const GPU_Board &t_board, Colour t_playerWhoJustMoved);
__host__ void        init_promotion_const_mem();

#endif