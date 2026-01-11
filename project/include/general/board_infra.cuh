//
// Created by chefxx on 8.01.2026.
//

#ifndef BOARD_INFRA_H
#define BOARD_INFRA_H

#include <array>
#include <iostream>
#include <string>

#include "bit_operations.h"

constexpr size_t  NOT_FILE_A = 0xFEFEFEFEFEFEFEFEULL;
constexpr size_t  NOT_FILE_H = 0x7F7F7F7F7F7F7F7FULL;
constexpr uint64_t NOT_FILE_B = 0xFDFDFDFDFDFDFDFDULL;
constexpr uint64_t NOT_FILE_G = 0xBFBFBFBFBFBFBFBFULL;

constexpr uint8_t UP_RIGHT = 0;
constexpr uint8_t UP_LEFT = 1;
constexpr uint8_t DOWN_RIGHT = 2;
constexpr uint8_t DOWN_LEFT = 3;

enum Colour
{
    black = 0,
    white = 1,
};

inline std::ostream& operator<<(std::ostream& os, const Colour& t_colour)
{
    const std::string colStr = t_colour == black ? "black" : "white";
    return os << colStr;
}

struct Board
{
    explicit Board(const Colour t_perspective) : m_perspective(t_perspective)
    {
        initBoard();
        initLUTs();
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
                if (const int bitIdx = actualRow * 8 + actualCol; checkBitAtIdx(m_pawns[white], bitIdx)) {
                    std::cout << "| w ";
                }
                else if (checkBitAtIdx(m_pawns[black], bitIdx)) {
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
        return popCount(m_pawns[white]) == popCount(m_pawns[black]);
    }
    // this is the state of the board
    // 0 represents black pieces
    // 1 represents white pieces
    std::array<size_t, 2> m_pawns;
    std::array<size_t, 2> m_kings;

    // lookup tables
    size_t JumpTable[64][4];
    size_t NeighbourTable[64][4];

    // TODO: initialization of king table
    //const size_t KingTable[64][4];

private:
    void initBoard()
    {
        constexpr size_t firstRowMask  = 85;
        constexpr size_t secondRowMask = 170;

        m_pawns[white] = 0ull;
        m_pawns[black] = 0ull;

        // initialize white player rocks
        m_pawns[white] |= firstRowMask;
        m_pawns[white] |= secondRowMask << 8;
        m_pawns[white] |= firstRowMask  << 16;

        // initialize black player rocks
        m_pawns[black] |= secondRowMask << 40;
        m_pawns[black] |= firstRowMask  << 48;
        m_pawns[black] |= secondRowMask << 56;

        // perspective only affects printing
        if (m_perspective == black) {
            columnsNamesRow = "    h   g   f   e   d   c   b   a \n";
            rowsNames = {'1', '2', '3', '4', '5', '6', '7', '8'};
        }
        else {
            columnsNamesRow = "    a   b   c   d   e   f   g   h \n";
            rowsNames = {'8', '7', '6', '5', '4', '3', '2', '1'};
        }
    }

    void initLUTs()
    {
        for (int i = 0; i < 64; ++i) {
            const size_t index = MIN_LSB << i;
            NeighbourTable[i][UP_RIGHT]   = (index & NOT_FILE_H) << 9;
            NeighbourTable[i][UP_LEFT]    = (index & NOT_FILE_A) << 7;
            NeighbourTable[i][DOWN_RIGHT] = (index & NOT_FILE_H) >> 7;
            NeighbourTable[i][DOWN_LEFT]  = (index & NOT_FILE_A) >> 9;

            JumpTable[i][UP_RIGHT]   = (index & NOT_FILE_G & NOT_FILE_H) << 18;
            JumpTable[i][UP_LEFT]    = (index & NOT_FILE_A & NOT_FILE_B) << 14;
            JumpTable[i][DOWN_RIGHT] = (index & NOT_FILE_G & NOT_FILE_H) >> 14;
            JumpTable[i][DOWN_LEFT]  = (index & NOT_FILE_A & NOT_FILE_B) >> 18;
        }
    }

    // perspective and printing utils
    Colour m_perspective;
    std::array<char, 8> rowsNames{};
    std::string columnsNamesRow;
};

#endif