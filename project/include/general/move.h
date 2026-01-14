//
// Created by chefxx on 11.01.2026.
//

#ifndef MCTS_CHECKERS_MOVE_H
#define MCTS_CHECKERS_MOVE_H

#include <array>
#include <vector>

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

inline int Col(const char t_col) { return t_col - 97; }

inline int Row(const char t_col) { return t_col - 49; }

inline uint8_t strToPos(const std::string &t_pos)
{
    // a1
    const auto col = Col(t_pos[0]);
    const auto row = Row(t_pos[1]);
    return row * 8 + col;
}

inline std::string posToStr(const int t_pos)
{
    const int  row     = t_pos / 8;
    const int  col     = t_pos % 8;
    const char colChar = static_cast<char>('a' + col);
    const char rowChar = static_cast<char>('1' + row);
    return {colChar, rowChar};
}

constexpr int MAX_MOVE_SEQUENCE = 8;

enum class MoveKind { null = -1, normal = 0, attack = 1 };

// TODO: change PlayerMove underlying into Move
struct PlayerMove
{
    void addPosition(const std::string &t_pos)
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
    uint8_t                                count = 0;
    MoveKind                               kind  = MoveKind::null;
};

// Note: move struct handles promotion conditions, so all player logic should be handled
// by this struct.
struct Move
{
    explicit Move(const size_t t_from,
                  const size_t t_to,
                  const bool   t_promotion,
                  const int    t_fromIdx,
                  const int    t_toIdx)
        : from_mask(t_from)
        , to_mask(t_to)
        , captures_mask(0ull)
        , is_promotion(t_promotion)
        , positions({t_fromIdx, t_toIdx})
    {
    }
    explicit Move(const size_t     t_from,
                  const size_t     t_to,
                  const size_t     t_captures,
                  const bool       t_promotion,
                  std::vector<int> t_path)
        : from_mask(t_from)
        , to_mask(t_to)
        , captures_mask(t_captures)
        , is_promotion(t_promotion)
        , positions(std::move(t_path))
    {
    }
    size_t           from_mask;
    size_t           to_mask;
    size_t           captures_mask;
    bool             is_promotion;
    std::vector<int> positions;
};

inline std::ostream &operator<<(std::ostream &os, const Move &t_move)
{
    const char delim = t_move.captures_mask ? ':' : '-';
    os << posToStr(t_move.positions[0]);
    for (size_t i = 1; i < t_move.positions.size(); ++i) {
        os << delim << posToStr(t_move.positions[i]);
    }
    return os;
}

inline std::string stringMove(const Move &t_move)
{
    std::stringstream ss;
    ss << t_move;
    return ss.str();
}

#endif // MCTS_CHECKERS_MOVE_H
