//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include <optional>
#include <string>

#include "bit_operations.h"
#include "board_infra.cuh"
#include "constants.h"
#include "move.h"

struct GameManager
{
    explicit GameManager(const Colour t_perspective)
        : m_perspective(t_perspective)
    {
        initPerspective();
        board.initStartingBoard();
    }
    Board board{};

    void printBoard() const
    {
        const std::string middleRow = " +---+---+---+---+---+---+---+---+\n";

        // Remember that rocks may overlap each other, this func just prints
        std::cout << columnsNamesRow;
        for (int visualRow = 0; visualRow < 8; ++visualRow) {
            std::cout << middleRow;
            std::cout << rowsNames[visualRow];
            for (int visualCol = 0; visualCol < 8; ++visualCol) {
                int actualRow, actualCol;
                if (m_perspective == black) {
                    actualRow = visualRow;
                    actualCol = 7 - visualCol;
                }
                else {
                    actualRow = 7 - visualRow;
                    actualCol = visualCol;
                }
                if (const int bitIdx = actualRow * 8 + actualCol; checkBitAtIdx(board.pawns[white], bitIdx)) {
                    std::cout << "| w ";
                }
                else if (checkBitAtIdx(board.pawns[black], bitIdx)) {
                    std::cout << "| b ";
                }
                else {
                    std::cout << "|   ";
                }
            }
            std::cout << '|' << rowsNames[visualRow] << '\n';
        }

        std::cout << middleRow;
        std::cout << columnsNamesRow;
    }

private:
    void initPerspective()
    {
        // perspective only affects printing
        if (m_perspective == black) {
            columnsNamesRow = "    h   g   f   e   d   c   b   a \n";
            rowsNames       = {'1', '2', '3', '4', '5', '6', '7', '8'};
        }
        else {
            columnsNamesRow = "    a   b   c   d   e   f   g   h \n";
            rowsNames       = {'8', '7', '6', '5', '4', '3', '2', '1'};
        }
    }

    // perspective and printing utils
    Colour              m_perspective;
    std::array<char, 8> rowsNames{};
    std::string         columnsNamesRow;
};

Colour                    drawStartingColour();
std::optional<PlayerMove> parseMove(const std::string &t_move);

#endif // MCTS_CHECKERS_GAME_ENGINE_H
