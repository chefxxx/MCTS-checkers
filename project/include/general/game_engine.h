//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include "board_infra.cuh"

Colour drawStartingColour();
void playPlayer(Board& t_currentBoard, Colour t_myColour);
void playAI_cpu();

#endif // MCTS_CHECKERS_GAME_ENGINE_H
