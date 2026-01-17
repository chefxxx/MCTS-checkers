//
// Created by chefxx on 11.01.2026.
//

#ifndef MCTS_CHECKERS_MOVE_H
#define MCTS_CHECKERS_MOVE_H

#include <array>
#include <vector>

#include "bit_operations.h"
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

enum class MoveKind { null = -1, quiet = 0, attack = 1 };

// Note: move struct handles promotion conditions, so all player logic should be handled
// by this struct.
struct Move
{
    explicit Move(const int    t_fromIdx,
                  const int    t_toIdx)
        : from_mask(1ULL << t_fromIdx)
        , to_mask(1ULL << t_toIdx)
        , captures_mask(0ull)
        , positions({t_fromIdx, t_toIdx})
        , kind(MoveKind::quiet)
    {
    }
    explicit Move(const size_t     t_captures,
                  std::vector<int> t_path)
        : from_mask(1ULL << t_path[0])
        , to_mask(1ULL << t_path[t_path.size() - 1])
        , captures_mask(t_captures)
        , positions(std::move(t_path))
        , kind(MoveKind::attack)
    {
    }

    /** @brief This constructor is strictly used in
     * the plyer input move scenario.
     *
     * @param t_path
     */
    explicit Move(std::vector<int> t_path)
        : from_mask(1ULL << t_path[0])
        , to_mask(1ULL << t_path[t_path.size() - 1])
        , positions(std::move(t_path))
    {
        size_t captures = 0ULL;
        for (size_t i = 1; i < t_path.size() - 1; i++) {
            captures |= (1ULL << t_path[i]);
        }
        captures_mask = captures;
    }
    size_t from_mask;
    size_t to_mask;
    size_t captures_mask;
    // TODO: add limitation to this vector's size to 9
    // TODO: this is due to the bitpacking in LightMovePath
    std::vector<int> positions;
    MoveKind kind = MoveKind::null;
};

struct LightMovePath
{
    constexpr static int64_t None = -1;

    explicit LightMovePath(const std::vector<int> &t_positions, const bool t_capture)
    {
        pack_path(t_positions, t_capture);
    }

    explicit LightMovePath()
        : packed_path(None)
    {
    }

    void pack_path(const std::vector<int> &positions, const bool t_capture)
    {
        // Store the number of squares in the first 4 bits (max 15)
        packed_path |= positions.size() & 0xF;
        packed_path |= (t_capture & 1ULL) << 63;
        for (size_t i = 0; i < positions.size() && i < 10; ++i) {
            const size_t square = positions[i] & 0x3F; // 6 bits for 0-63
            packed_path |= (square << (4 + (i * 6)));
        }
    }

    int64_t packed_path = 0ULL;
};

struct PrintingMovePath
{
    explicit PrintingMovePath(const size_t t_packedPath) { unpack_path(t_packedPath); }

    void unpack_path(const size_t t_packed)
    {
        const int count = t_packed & 0xF;
        capture         = checkBitAtIdx(t_packed, 63);
        for (int i = 0; i < count; ++i) {
            positions.push_back((t_packed >> (4 + (i * 6))) & 0x3F);
        }
    }

    std::vector<int> positions;
    bool             capture;
};

inline std::ostream &operator<<(std::ostream &os, const PrintingMovePath &t_move)
{
    const char delim = t_move.capture ? ':' : '-';
    os << posToStr(t_move.positions[0]);
    for (size_t i = 1; i < t_move.positions.size(); ++i) {
        os << delim << posToStr(t_move.positions[i]);
    }
    return os;
}

inline std::string stringMove(const PrintingMovePath &t_move)
{
    std::stringstream ss;
    ss << t_move;
    return ss.str();
}

#endif // MCTS_CHECKERS_MOVE_H
