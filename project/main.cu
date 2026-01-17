#include "game_engine.h"

int main(int argc, const char **argv)
{
    std::cout << "Welcome to checkers.mcts!\n";
    std::cout << "Press any key to draw a colour and start the game...\n";
    std::cin.get();
    const auto playerColour = drawStartingColour();
    std::cout << "You are " << playerColour << "!\n";

    // initialize board from player's colour perspective
    GameManager manager{playerColour};
    manager.printBoard();
    manager.playTheGame();
    return 0;
}
