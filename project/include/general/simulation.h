//
// Created by chefxx on 17.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_SIMULATION_H
#define MCTS_CHECKERS_CPU_SIMULATION_H

#include "board_infra.h"
#include "mcts_tree.h"

MctsNode *runCPU_MCTS(const MctsTree &t_tree, double t_timeLimit);
MctsNode *run_DEBUG_MCTS(const MctsTree &t_tree);
void  mctsIteration(const MctsTree &t_tree);

#endif // MCTS_CHECKERS_CPU_SIMULATION_H
