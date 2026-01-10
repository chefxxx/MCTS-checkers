#include "board_infra.cuh"

int main(int argc, const char ** argv) {
    const Board board{StartingColour::white};
    board.printBoard();
    return 0;
}
