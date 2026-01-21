//
// Created by chefxx on 17.01.2026.
//

#include "simulation.h"

#include <cassert>
#include <chrono>

MctsNode *runCPU_MCTS(const MctsTree &t_tree, const double t_timeLimit)
{
    assert(static_cast<long long>(t_timeLimit * 1e6 * TURN_TIME_MULTIPLICATOR) > 0);
    const auto limit_micro_sec = std::chrono::microseconds(static_cast<long long>(t_timeLimit * 1e6 * TURN_TIME_MULTIPLICATOR));
    int iters = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        mctsIteration(t_tree);
        iters++;
        if (iters % ITERATION_CHECK == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            if (const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
                elapsed > limit_micro_sec) {
                break;
            }
        }
    }
    return chooseBestMove(t_tree);
}

MctsNode *run_DEBUG_MCTS(const MctsTree &t_tree)
{
    for (int i = 0; i < 1000; ++i) {
        mctsIteration(t_tree);
    }
    return chooseBestMove(t_tree);
}

void mctsIteration(const MctsTree &t_tree)
{
    // 1. selection
    auto selectedNode = selectNode(t_tree.root.get());

    // 2. expansion
    if (!selectedNode->is_fully_expanded()) {
        selectedNode = expandNode(selectedNode);

        // 3. rollout
        const auto score = rollout(selectedNode);

        // 4. backpropagate
        backpropagate(selectedNode, score);
    }
    else {
        assert(1 == 0);
    }
}