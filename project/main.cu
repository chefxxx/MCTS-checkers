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

    std::cout << "******************************************** RULESET ******************************************\n";
    std::cout << "* 0. Welcome to checkers.mcts!                                                                *\n";
    std::cout << "* 1. Quiet moves are marked as (from)-(to).                                                   *\n";
    std::cout << "* 2. Attack moves are marked as (from):(mid):(to).                                            *\n";
    std::cout << "* 3. Pawns move in one direction, diagonally, one square. Pawns can attack in any direction.  *\n";
    std::cout << "* 4. If pawn ends its move on the opposite last rank, it becomes a king.                      *\n";
    std::cout << "* 5. Kings move ina any direction, diagonally, any number of squares.                         *\n";
    std::cout << "* 6. There is no limit of taking opponents pieces in one attack move.                         *\n";
    std::cout << "* 7. At any point of the game you can resign, by writing 'resign'.                            *\n";
    std::cout << "* 8. Have fun!                                                                                *\n";
    std::cout << "******************************************** RULESET ******************************************\n\n";
    logger::info("Press enter to draw a colour and start the game...");
    std::cin.get();
    const auto playerColour = drawStartingColour();
    logger::info("You are {}!\n\n", playerColour);

    // initialize board from player's colour perspective
    GameManager manager{playerColour, time, mode};
    manager.playTheGame();
    return 0;
}
