//
// Created by chefxx on 10.01.2026.
//

#include <random>
#include "game_engine.h"

StartingColour drawStartingColour() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen) == 0 ? StartingColour::black : StartingColour::white;
}

void gameLoop(const StartingColour t_player)
{

}