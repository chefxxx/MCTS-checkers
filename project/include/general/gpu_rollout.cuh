//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_ROLLOUT_CUH
#define GPU_ROLLOUT_CUH

#include "mcts_tree.h"

double rollout_gpu(const MctsNode *t_node);

#endif