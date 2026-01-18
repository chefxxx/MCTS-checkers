#include "game_engine.h"

void usage()
{
    logger::err("Usage: ./MCTS-checkers [mode=<cpu/gpu>] [time-for-ai-turn=<1> (in seconds)]");
    return exit(EXIT_FAILURE);
}

int main(const int argc, const char **argv)
{
    if (argc != 3) {
        usage();
    }
    const std::string mode = argv[1];
    const double time = std::stoi(argv[2]);

    logger::info("Welcome to checkers.mcts!\n");
    logger::info("Press any key to draw a colour and start the game...");
    std::cin.get();
    const auto playerColour = drawStartingColour();
    logger::info("You are {}!\n\n", playerColour);

    // initialize board from player's colour perspective
    GameManager manager{playerColour, time, mode};
    manager.playTheGame();
    return 0;
}
