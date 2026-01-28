#include "game_engine.cuh"


int main(const int argc, const char **argv)
{
    std::cout << "Welcome to checkers.mcts!\n";
    std::cout << "Choose one of the following modes:\n\n";
    std::cout << "************ MODE SELECTION ************\n";
    std::cout << "*                                      *\n";
    std::cout << "*  1. Human vs computer (press 1)      *\n";
    std::cout << "*  2. Computer vs computer (press 2)   *\n";
    std::cout << "*                                      *\n";
    std::cout << "************ MODE SELECTION ************\n";

    std::string mode;
    std::getline(std::cin, mode);

    if (mode == "1") {
        const double time = setTime("computer");
        const auto engine = selectArchitecture("computer");
        printRuleset();

        logger::info("Press enter to draw a colour and start the game...");
        std::cin.get();
        const auto playerColour = drawStartingColour();
        logger::info("You are {}!\n\n", playerColour);

        // initialize board from player's colour perspective
        GameManager manager{playerColour, time,engine};
        manager.playTheGame();
    }
    else if (mode == "2") {

    }
    else {
        logger::err("Unknown game mode, try again!\n");
    }

    return 0;
}
