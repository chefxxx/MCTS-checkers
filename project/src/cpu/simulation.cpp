//
// Created by chefxx on 17.01.2026.
//

#include "simulation.h"

#include <cassert>
#include <chrono>

MctsNode *runCPU_MCTS(MctsTree &t_tree, const double t_timeLimit)
{
    const auto                start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed;
    const auto                limit_micro_sec =
        std::chrono::microseconds(static_cast<long long>(t_timeLimit * 1e6 * TURN_TIME_MULTIPLICATOR));
    int iters = 0;
    while (elapsed < limit_micro_sec) {
        mctsIteration(t_tree);
        iters++;
        if (iters % ITERATION_CHECK == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            elapsed  = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
            if (elapsed > limit_micro_sec) {
                break;
            }
        }
    }
    return chooseBestMove(t_tree);
}

// Board run_DEBUG_MCTS(MctsTree &t_tree)
// {
//     for (int i = 0; i < 1000; ++i) {
//         mctsIteration(t_tree);
//     }
//     return chooseBestMove(t_tree);
// }

void mctsIteration(const MctsTree &t_tree)
{
    // 1. selection
    auto selectedNode = selectNode(t_tree.root.get());

    // 2. expansion
    if (!selectedNode->is_terminal()) {
        selectedNode = expandNode(selectedNode);

        // 3. rollout
        const auto score = rollout();

        // 4. backpropagate
        backpropagate(selectedNode, score, t_tree.colour_of_ai);
    }
}