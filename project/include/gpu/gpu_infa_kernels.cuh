//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_INFRA_KERNELS_CUH
#define GPU_INFRA_KERNELS_CUH

#include <curand_kernel.h>

#include "constants.h"
#include "gpu_board.cuh"
#include "gpu_movegen.cuh"
#include "memory_cuda.cuh"

__global__ void rollout_kernel(curandState *t_stateBuff, Colour t_startingTurn, double *t_globalScore);
__global__ void setup_curand_kernel(curandState *state, unsigned long seed);
__global__ void test_kernel(const curandState *t_stateBuff, const GPU_Board *testBoard, Colour t_startingTurn, GPU_Move *t_resultMove);
__global__ void reset_score(double *t_globalScore);

__host__ void init_gpu_const_board(const Board &t_board);
__host__ mem_cuda::unique_ptr<curandState> init_random_states();
__device__ void summarize_warp(double t_score, double *t_globalScore);


#endif
