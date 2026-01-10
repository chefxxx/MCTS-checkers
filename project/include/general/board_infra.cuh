//
// Created by chefxx on 8.01.2026.
//

#ifndef BOARD_INFRA_H
#define BOARD_INFRA_H

#include <array>
#include <iostream>
#include <string>

#include "bit_operations.cuh"

enum class StartingColour
{
    black = 0,
    white = 1,
};

struct Board
{
    __host__ explicit Board(const StartingColour t_perspective) : m_whiteBoard(0ull), m_blackBoard(0ull), m_perspective(t_perspective)
    {
        initBoard();
    }
    __host__ void printBoard() const
    {
        const std::string middleRow = " +---+---+---+---+---+---+---+---+\n";

        // Remember that rocks may overlap each other, this func just prints
        std::cout << columnsNamesRow;
        int row = 0;
        for (int i = 56; i >= 0; i -= 8) {
            std::cout << middleRow;
            std::cout << rowsNames[row];
            for (int j = i; j < i + 8; ++j) {
                if (checkBitAtIdx(m_whiteBoard, j)) {
                    std::cout << "| w ";
                }
                else if (checkBitAtIdx(m_blackBoard, j)) {
                    std::cout << "| b ";
                }
                else {
                    std::cout << "|   ";
                }
            }
            std::cout << '|' << rowsNames[row] << '\n';
            ++row;
        }
        std::cout << middleRow;
        std::cout << columnsNamesRow;
    }
private:
    void initBoard()
    {
        constexpr size_t firstRowMask  = 85;
        constexpr size_t secondRowMask = 170;

        size_t us = 0ull;
        size_t opponent = 0ull;

        // initialize us player rocks
        us |= firstRowMask;
        us |= secondRowMask << 8;
        us |= firstRowMask  << 16;

        // initialize opponent player rocks
        opponent |= secondRowMask << 40;
        opponent |= firstRowMask  << 48;
        opponent |= secondRowMask << 56;

        if (m_perspective == StartingColour::black) {
            columnsNamesRow = "    h   g   f   e   d   c   b   a \n";
            rowsNames = {'1', '2', '3', '4', '5', '6', '7', '8'};
            m_blackBoard = us;
            m_whiteBoard = opponent;
        }
        else {
            columnsNamesRow = "    a   b   c   d   e   f   g   h \n";
            rowsNames = {'8', '7', '6', '5', '4', '3', '2', '1'};
            m_blackBoard = opponent;
            m_whiteBoard = us;
        }
    }
    // all moves will be performed from player perspective, so
    // always the player perspective start on the 0..32 positions
    size_t m_whiteBoard;
    size_t m_blackBoard;

    // perspective and printing utils
    StartingColour m_perspective;
    std::array<char, 8> rowsNames{};
    std::string columnsNamesRow;
};

#endif