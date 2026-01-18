//
// Created by chefxx on 17.01.2026.
//

#include "simulation.h"

#include <chrono>

Board runCPU_MCTS(const MctsTree &t_tree, const double t_timeLimit)
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
    return chooseBestMove(t_tree.root.get());
}

void mctsIteration(const MctsTree &t_tree)
{
    // 1. selection
    auto selectedNode = selectNode(t_tree.root.get());
    // 2. If node has been visited - expansion
    if (selectedNode->number_of_visits) {
        // TODO: expand node must handle the no move cases
        expandNode(selectedNode);
        selectedNode = selectedNode->children[0].get();
    }
    // 3. rollout
    const auto t_score = rollout();
    // 4. backpropagation
    backpropagate(selectedNode, t_score, t_tree.colour_of_ai);
}