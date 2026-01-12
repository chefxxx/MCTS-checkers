//
// Created by chefxx on 10.01.2026.
//

#include "game_engine.h"

#include <random>

#include "logger.h"

Colour drawStartingColour()
{
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen) == 0 ? Colour::black : Colour::white;
}

std::optional<PlayerMove> parseMove(const std::string &t_move)
{
    if (t_move.size() < 5) {
        logger::warn("Not valid move format, try again!");
        return std::nullopt;
    }
    PlayerMove move;
    move.kind = t_move[2] == '-' ? MoveKind::normal : MoveKind::attack;
    for (size_t i = 0; i < t_move.size(); i += 3) {
        if (std::isalpha(t_move[i])) {
            // we have found character
            if (i + 1 < t_move.size() && std::isdigit(t_move[i + 1])) {
                std::string curr = t_move.substr(i, 2);
                move.addPosition(curr);
            }
        }
    }
    return move;
}