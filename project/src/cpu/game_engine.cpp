//
// Created by chefxx on 10.01.2026.
//

#include <random>
#include "game_engine.h"

Colour drawStartingColour() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen) == 0 ? Colour::black : Colour::white;
}

void playPlayer(Board& t_currentBoard, Colour t_myColour)
{
    if (t_myColour == Colour::black)
        std::cout << "Player moves...\n";
    t_currentBoard.printBoard();
}