//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include <optional>
#include <string>

#include "board_infra.cuh"
#include "move.h"

Colour              drawStartingColour();
void                playPlayer(Board &t_currentBoard, Colour t_myColour);
std::optional<move> parseMove(const std::string &t_move);
void                playAI_cpu();

#endif // MCTS_CHECKERS_GAME_ENGINE_H
