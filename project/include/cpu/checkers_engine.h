//
// Created by chefxx on 16.01.2026.
//

#ifndef MCTS_CHECKERS_CPU_CHECKERS_ENGINE_H
#define MCTS_CHECKERS_CPU_CHECKERS_ENGINE_H

#include <optional>

#include "board_infra.h"
#include "constants.h"
#include "move.h"

std::optional<Board> applyMove(const Board &t_board, const Move &t_move, Colour t_colour);
GameState            checkEndOfGameConditions(const Board &t_board, Colour t_playerWhoJustMadeAMove);

#endif // MCTS_CHECKERS_CPU_CHECKERS_ENGINE_H
