//
// Created by chefxx on 17.01.2026.
//

#include "simulation.h"
#include <chrono>

Board runCPU_MCTS(const std::unique_ptr<MctsNode> &t_root, const double t_timeLimit)
{

    const auto start = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds elapsed;
    const auto limit_micro_sec = std::chrono::microseconds(static_cast<long long>(t_timeLimit * 1e6 * TURN_TIME_MULTIPLICATOR));
    int iters = 0;
    while (elapsed < limit_micro_sec) {
        mctsIteration(t_root);
        iters++;
        if (iters % ITERATION_CHECK == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
            if (elapsed > limit_micro_sec) {
                break;
            }
        }
    }
    return MctsTree::chooseBestMove(t_root.get());
}

void mctsIteration(const std::unique_ptr<MctsNode> &t_root)
{
    // 1. selection
    auto selectedNode = MctsTree::selectNode(t_root.get());
    // 2. If node has been visited - expansion
    if (selectedNode->number_of_visits) {
        // TODO: expand node must handle the no move cases
        MctsTree::expandNode(selectedNode);
        selectedNode = selectedNode->children[0].get();
    }
    // 3. rollout
    const auto t_score = MctsTree::rollout();
    // 4. backpropagation
    MctsTree::backpropagate(selectedNode, t_score);
}