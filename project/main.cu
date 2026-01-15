#include "game_engine.h"

int main(int argc, const char **argv)
{
    std::cout << "Welcome to checkers.mcts!\n";
    std::cout << "Press any key to draw a colour and start the game...\n";
    std::cin.get();
    const auto playerColour = drawStartingColour();
    std::cout << "You are " << playerColour << "!\n";

    const Colour aiColour     = playerColour == Colour::black ? Colour::white : Colour::black;
    auto         current_turn = Colour::white;
    bool         gameRunning  = true;

    // initialize board from player's colour perspective
    const GameManager manager{playerColour};
    manager.printBoard();

    return 0;
}
