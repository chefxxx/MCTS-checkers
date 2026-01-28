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
        const double timeP1 = setTime("player 1");
        const double timeP2 = setTime("player 2");
        const auto engineP1 = selectArchitecture("player 1");
        const auto engineP2 = selectArchitecture("player 2");
        const auto colP1 = selectColour("player 1");
        const auto colP2 = static_cast<Colour>(1 - colP1);
        AI_Player p1{timeP1, engineP1, colP1};
        AI_Player p2{timeP2, engineP2, colP2};
        ComputerGame game_manager{p1, p2};
        game_manager.playTheGame();
    }
    else {
        logger::err("Unknown game mode, try again!\n");
    }

    return EXIT_SUCCESS;
}
