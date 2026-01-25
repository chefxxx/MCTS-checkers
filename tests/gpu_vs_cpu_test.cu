//
// Created by chefxx on 25.01.2026.
//

#include <gtest/gtest.h>

#include "checkers_engine.h"
#include "game_engine.cuh"
#include "gpu_checkers_engine.cuh"
#include "gpu_infa_kernels.cuh"
#include "gpu_movegen.cuh"
#include "mcts_tree.h"
#include "memory_cuda.cuh"

class Gpu_vs_Cpu_Test : public ::testing::Test
{
public:
    mem_cuda::unique_ptr<curandState> d_states;
    mem_cuda::unique_ptr<double> d_globalScore;

    MctsTree                      cpu_mcts_tree;
    MctsTree                      gpu_mcts_tree;

    void SetUp() override
    {
        init_promotion_const_mem();
        init_gpu_movegen_const_mem();
        d_globalScore = mem_cuda::make_unique<double>();
        d_states      = init_random_states();
    }
};

#if false
TEST_F(Gpu_vs_Cpu_Test, cpu_vs_cpu_test)
{
    double cpu_score = 0.0;
    double gpu_score = 0.0;
    for (int i = 0; i < 10; i++) {
        Board b;
        b.initStartingBoard();

        const Colour cpu_col = drawStartingColour();

        cpu_mcts_tree.initTree(b, white);
        gpu_mcts_tree.initTree(b, white);

        Colour current_turn = white;
        GameState state = CONTINUES;
        while (state == CONTINUES) {
            constexpr double time_per_turn = 0.5;
            MctsTree& actor_tree = (current_turn == cpu_col) ? cpu_mcts_tree : gpu_mcts_tree;
            MctsTree& observer_tree = (current_turn == cpu_col) ? gpu_mcts_tree : cpu_mcts_tree;
            std::string mode = (current_turn == cpu_col) ? "cpu" : "gpu";
            std::string opposite_mode = (current_turn == cpu_col) ? "gpu" : "cpu";

            const MctsNode* best = runMctsSimulation(actor_tree, time_per_turn, mode, d_states, d_globalScore);
            b = best->current_board_state;
            const LightMovePath path = best->packed_positions_transition;
            actor_tree.updateRoot(best);
            observer_tree.updateTree();
            const auto syncNode = findPlayerMove(observer_tree.root.get(), b, path);
            assert(syncNode != nullptr);
            observer_tree.updateRoot(syncNode);

            state = checkEndOfGameConditions(b, current_turn);
            if (state != CONTINUES) break;

            current_turn = static_cast<Colour>(1 - current_turn);
        }

        if (state == DRAW) {
            cpu_score += 0.5; gpu_score += 0.5;
        } else {
            if (current_turn == cpu_col) cpu_score += 1.0;
            else gpu_score += 1.0;
        }
        std::cout << "gpu_score: " << gpu_score << std::endl;
        std::cout << "cpu_score: " << cpu_score << std::endl;
    }
}
#endif