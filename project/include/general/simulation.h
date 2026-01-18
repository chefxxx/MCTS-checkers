//
// Created by chefxx on 17.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_SIMULATION_H
#define MCTS_CHECKERS_CPU_SIMULATION_H


#include <memory>

#include "board_infra.h"
#include "mcts_tree.h"

Board runCPU_MCTS(const std::unique_ptr<MctsNode> &t_root, double t_timeLimit);
void  mctsIteration(const std::unique_ptr<MctsNode> &t_root);

#endif // MCTS_CHECKERS_CPU_SIMULATION_H
