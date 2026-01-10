//
// Created by chefxx on 8.01.2026.
//

#ifndef BOARD_INFRA_H
#define BOARD_INFRA_H

#include <array>
#include <iostream>
#include <string>

#include "bit_operations.h"

enum class Colour
{
    black = 0,
    white = 1,
};

inline std::ostream& operator<<(std::ostream& os, const Colour& t_colour)
{
    const std::string colStr = t_colour == Colour::black ? "black" : "white";
    return os << colStr;
}

struct Board
{
    explicit Board(const Colour t_perspective) : m_whiteBoard(0ull), m_blackBoard(0ull), m_perspective(t_perspective)
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
                if (m_perspective == Colour::black) {
                    actualRow = visualRow;
                    actualCol = 7 - visualCol;
                }
                else {
                    actualRow = 7 - visualRow;
                    actualCol = visualCol;
                }
                if (const int bitIdx = actualRow * 8 + actualCol; checkBitAtIdx(m_whiteBoard, bitIdx)) {
                    std::cout << "| w ";
                }
                else if (checkBitAtIdx(m_blackBoard, bitIdx)) {
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
    [[nodiscard]] bool isGameOver() const
    {
        return popCount(m_whiteBoard) == popCount(m_blackBoard);
    }
private:
    void initBoard()
    {
        constexpr size_t firstRowMask  = 85;
        constexpr size_t secondRowMask = 170;

        // initialize us player rocks
        m_whiteBoard |= firstRowMask;
        m_whiteBoard |= secondRowMask << 8;
        m_whiteBoard |= firstRowMask  << 16;

        // initialize opponent player rocks
        m_blackBoard |= secondRowMask << 40;
        m_blackBoard |= firstRowMask  << 48;
        m_blackBoard |= secondRowMask << 56;

        // perspective only affects printing
        if (m_perspective == Colour::black) {
            columnsNamesRow = "    h   g   f   e   d   c   b   a \n";
            rowsNames = {'1', '2', '3', '4', '5', '6', '7', '8'};
        }
        else {
            columnsNamesRow = "    a   b   c   d   e   f   g   h \n";
            rowsNames = {'8', '7', '6', '5', '4', '3', '2', '1'};
        }
    }
    size_t m_whiteBoard;
    size_t m_blackBoard;

    // perspective and printing utils
    Colour m_perspective;
    std::array<char, 8> rowsNames{};
    std::string columnsNamesRow;
};

#endif