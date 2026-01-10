//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include "board_infra.cuh"

StartingColour drawStartingColour();
void gameLoop(StartingColour t_player);
void genMoves_cpu();

#endif // MCTS_CHECKERS_GAME_ENGINE_H
