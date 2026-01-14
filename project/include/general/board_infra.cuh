//
// Created by chefxx on 8.01.2026.
//

#ifndef BOARD_INFRA_H
#define BOARD_INFRA_H

#include <array>
#include <iostream>
#include <string>

#include "bit_operations.h"
#include "constants.h"


struct Board
{
    explicit Board(const Colour t_perspective)
        : m_perspective(t_perspective)
    {
        initBoard();
    }
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
                if (const int bitIdx = actualRow * 8 + actualCol; checkBitAtIdx(pawns[white], bitIdx)) {
                    std::cout << "| w ";
                }
                else if (checkBitAtIdx(pawns[black], bitIdx)) {
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

    // TODO: dummy function
    [[nodiscard]] bool isGameOver() const { return popCount(pawns[white]) == popCount(pawns[black]); }
    // this is the state of the board
    // 0 represents black pieces
    // 1 represents white pieces
    std::array<size_t, 2> pawns{};
    std::array<size_t, 2> kings{};

private:
    void initBoard()
    {
        constexpr size_t firstRowMask  = 85;
        constexpr size_t secondRowMask = 170;

        pawns[white] = 0ull;
        pawns[black] = 0ull;

        // initialize white player rocks
        pawns[white] |= firstRowMask;
        pawns[white] |= secondRowMask << 8;
        pawns[white] |= firstRowMask << 16;

        // initialize black player rocks
        pawns[black] |= secondRowMask << 40;
        pawns[black] |= firstRowMask << 48;
        pawns[black] |= secondRowMask << 56;

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

#endif