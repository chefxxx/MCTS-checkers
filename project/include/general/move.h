//
// Created by chefxx on 11.01.2026.
//

#ifndef MCTS_CHECKERS_MOVE_H
#define MCTS_CHECKERS_MOVE_H

#include <array>

#include "logger.h"

// Program "sees" the board from this perspective
// thus macros for cols are set here accordingly.
//
//    a   b   c   d   e   f   g   h
//  +---+---+---+---+---+---+---+---+
// 8|   | b |   | b |   | b |   | b |8
//  +---+---+---+---+---+---+---+---+
// 7| b |   | b |   | b |   | b |   |7
//  +---+---+---+---+---+---+---+---+
// 6|   | b |   | b |   | b |   | b |6
//  +---+---+---+---+---+---+---+---+
// 5|   |   |   |   |   |   |   |   |5
//  +---+---+---+---+---+---+---+---+
// 4|   |   |   |   |   |   |   |   |4
//  +---+---+---+---+---+---+---+---+
// 3| w |   | w |   | w |   | w |   |3
//  +---+---+---+---+---+---+---+---+
// 2|   | w |   | w |   | w |   | w |2
//  +---+---+---+---+---+---+---+---+
// 1| w |   | w |   | w |   | w |   |1
//  +---+---+---+---+---+---+---+---+
//     a   b   c   d   e   f   g   h

inline int Col(const char t_col)
{
    return t_col - 97;
}

inline int Row(const char t_col)
{
    return t_col - 49;
}

inline uint8_t strToPos(const std::string& t_pos)
{
    // a1
    const auto col = Col(t_pos[0]);
    const auto row = Row(t_pos[1]);
    return row * 8 + col;
}

constexpr int MAX_MOVE_SEQUENCE = 12;

enum class MoveKind
{
    null = -1,
    normal = 0,
    take = 1
};

struct Move
{
    void addPosition(const std::string& t_pos)
    {
        const auto idx = strToPos(t_pos);
        if (count < MAX_MOVE_SEQUENCE) {
            positions[count] = idx;
            count++;
        }
        else {
            logger::warn("Maximum moves sequence exceeded!");
        }
    }
    std::array<uint8_t, MAX_MOVE_SEQUENCE> positions{};
    uint8_t count = 0;
    MoveKind kind = MoveKind::null;
};

#endif // MCTS_CHECKERS_MOVE_H
