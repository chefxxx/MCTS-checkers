//
// Created by chefxx on 17.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_SIMULATION_H
#define MCTS_CHECKERS_CPU_SIMULATION_H

#include "board_infra.h"
#include "mcts_tree.h"

Board runCPU_MCTS(MctsTree &t_tree, double t_timeLimit);
Board run_DEBUG_MCTS(MctsTree &t_tree);
void  mctsIteration(const MctsTree &t_tree);

#endif // MCTS_CHECKERS_CPU_SIMULATION_H
