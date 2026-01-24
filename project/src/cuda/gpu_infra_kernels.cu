// -------------------------------------
// Constant memory variables definitions
// -------------------------------------

#include "gpu_checkers_engine.cuh"
#include "gpu_infa_kernels.cuh"
#include "helper_cuda.h"

__constant__ GPU_Board d_initBoard;

void init_gpu_const_board(const Board &t_board)
{
    const auto  d_board = GPU_Board(t_board);
    checkCudaErrors(cudaMemcpyToSymbol(d_initBoard, &d_board, sizeof(GPU_Board)));
}

mem_cuda::unique_ptr<curandState> init_random_states()
{
    // Allocate the buffer on the GPU
    auto d_states = mem_cuda::make_unique<curandState>(BLOCKS_PER_GRID * THREAD_PER_BLOCK);

    // Launch the setup once
    setup_curand_kernel<<<BLOCKS_PER_GRID, THREAD_PER_BLOCK>>>(d_states.get(), time(nullptr));
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    return d_states;
}

__device__ void summarize_warp(double t_score, double *t_globalScore)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        constexpr unsigned int mask = 0xFFFFFFFFU;
        t_score += __shfl_down_sync(mask, t_score, offset);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(t_globalScore, t_score);
    }
}

__global__ void rollout_kernel(curandState *t_stateBuff, const Colour t_startingTurn, double *t_globalScore)
{
    // Initialize the kernel's variables
    const size_t    tid       = threadIdx.x + blockIdx.x * blockDim.x;
    GPU_Board tmp_board       = d_initBoard;
    curandState     local     = t_stateBuff[tid];
    GameState state           = CONTINUES;
    const auto parent_colour  = static_cast<Colour>(1 - t_startingTurn);
    auto turn           = t_startingTurn;

    // perform a random game
    while (true) {
        const auto move = generate_random_move(&local, tmp_board, turn);
        apply_move_gpu(tmp_board, move, turn);
        state = check_end_of_game_conditions(tmp_board, turn);
        if (state != CONTINUES)
            break;
        turn = static_cast<Colour>(1 - turn);
    }

    double score = -1.0;
    if (state == DRAW)
        score = 0.5;
    else
        score = turn == parent_colour ? 1.0 : 0.0;

    summarize_warp(score, t_globalScore);

    // save results
    t_stateBuff[tid] = local;
}

__global__ void setup_curand_kernel(curandState *state, const unsigned long seed)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    /* curand_init parameters:
       1. seed: Can be the same for all threads.
       2. sequence number: MUST be unique (use thread ID).
       3. offset: Usually 0.
       4. state: Pointer to the specific state in your buffer.
    */
    curand_init(seed, tid, 0, &state[tid]);
}

__global__ void test_kernel(const curandState *t_stateBuff, const GPU_Board *testBoard, const Colour t_startingTurn, GPU_Move *t_resultMove)
{
    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto local_state = t_stateBuff[tid];
    const auto move = generate_random_move(&local_state, *testBoard, t_startingTurn);
    *t_resultMove = move;
}

__global__ void reset_score(double *t_globalScore)
{
    *t_globalScore = 0.0;
}
