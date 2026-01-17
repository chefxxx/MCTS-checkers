//
// Created by chefxx on 14.01.2026.
//

#ifndef MCTS_CHECKERS_CONSTANTS_H
#define MCTS_CHECKERS_CONSTANTS_H

#include <iostream>

constexpr int DRAW_LIMIT = 30;

constexpr size_t NOT_FILE_A = 0xFEFEFEFEFEFEFEFEULL;
constexpr size_t NOT_FILE_H = 0x7F7F7F7F7F7F7F7FULL;
constexpr size_t NOT_FILE_B = 0xFDFDFDFDFDFDFDFDULL;
constexpr size_t NOT_FILE_G = 0xBFBFBFBFBFBFBFBFULL;

constexpr size_t PROMOTION_BLACK = 0x00000000000000FFULL;
constexpr size_t PROMOTION_WHITE = 0xFF00000000000000ULL;

constexpr size_t promotion[2]{PROMOTION_BLACK, PROMOTION_WHITE};

enum Direction { UP_RIGHT = 0, UP_LEFT = 1, DOWN_RIGHT = 2, DOWN_LEFT = 3 };

enum GameState { LOST = 0, DRAW = 1, WON = 2, CONTINUES = 3 };

enum Colour {
    black = 0,
    white = 1,
};

inline std::ostream &operator<<(std::ostream &os, const Colour &t_colour)
{
    const std::string colStr = t_colour == black ? "black" : "white";
    return os << colStr;
}

inline std::ostream &operator<<(std::ostream &os, const GameState &t_game_state)
{
    const std::string colStr = t_game_state == WON ? "won" : "lost";
    return os << colStr;
}

// An array to perform branchless
// sliding move generation.
//
// ------------------------------
constexpr int canMove[2][4] = {
    {0, 0, 1, 1}, // this represents black
    {1, 1, 0, 0}  // this represents white
};


#endif // MCTS_CHECKERS_CONSTANTS_H
