#include "game_engine.h"
#include "mcts_engine.cuh"

int main(int argc, const char ** argv) {

    std::cout << "Welcome to checkers.mcts!\n";
    std::cout << "Press any key to draw a colour and start the game...\n";
    std::cin.get();
    const auto playerColour = drawStartingColour();
    std::cout << "You are " << playerColour << "!\n";

    const Colour aiColour = playerColour == Colour::black ? Colour::white : Colour::black;
    auto current_turn = Colour::white;
    bool gameRunning = true;

    // initialize board from player's colour perspective
    Board board{playerColour};
    board.printBoard();

    while (gameRunning) {
        if (current_turn == playerColour) {
            playPlayer(board, playerColour);
        }
        else {
            playAI_gpu(board, aiColour);
        }

        if (board.isGameOver()) {
            // game over
            gameRunning = false;
        }
        else {
            current_turn = current_turn == Colour::white ? Colour::black : Colour::white;
        }
    }
    return 0;
}
