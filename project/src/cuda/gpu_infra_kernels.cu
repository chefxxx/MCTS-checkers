// -------------------------------------
// Constant memory variables definitions
// -------------------------------------

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

__global__ void rollout_kernel(curandState *t_stateBuff, const Colour t_startingTurn)
{
    // Initialize the kernel's variables
    const size_t    tid       = threadIdx.x + blockIdx.x * blockDim.x;
    const GPU_Board tmp_board = d_initBoard;
    curandState     local     = t_stateBuff[tid];

    // perform a random game
    const auto move = generate_random_move(&local, tmp_board, t_startingTurn);

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
