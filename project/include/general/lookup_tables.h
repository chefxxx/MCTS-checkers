//
// Created by chefxx on 14.01.2026.
//

#ifndef MCTS_CHECKERS_LOOKUP_TABLES_H
#define MCTS_CHECKERS_LOOKUP_TABLES_H

#include "constants.h"

struct LookupTables
{
    constexpr LookupTables()
    {
        initLUTs();
        initDiagonals();
        initDirectionMapping();
    }
    constexpr void initLUTs()
    {
        for (int i = 0; i < 64; ++i) {
            const size_t index            = 1ULL << i;
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

    // https://github.com/maksimKorzh/chess_programming/blob/master/src/bitboards/slider_attacks/bitboard.c
    constexpr void initDiagonals()
    {
        for (int i = 0; i < 64; ++i) {
            size_t    upRightRay   = 0ULL;
            size_t    upLeftRay    = 0ULL;
            size_t    downRightRay = 0ULL;
            size_t    downLeftRay  = 0ULL;
            const int tr           = i / 8;
            const int tf           = i % 8;

            for (int r = tr + 1, f = tf + 1; r <= 7 && f <= 7; r++, f++)
                upRightRay |= (1ULL << (r * 8 + f));
            for (int r = tr - 1, f = tf - 1; r >= 0 && f >= 0; r--, f--)
                downLeftRay |= (1ULL << (r * 8 + f));

            for (int r = tr + 1, f = tf - 1; r <= 7 && f >= 0; r++, f--)
                upLeftRay |= (1ULL << (r * 8 + f));
            for (int r = tr - 1, f = tf + 1; r >= 0 && f <= 7; r--, f++)
                downRightRay |= (1ULL << (r * 8 + f));

            diagonalMaskEx[i]       = upRightRay | downLeftRay;
            anitDiagonalMaskEx[i]   = upLeftRay | downRightRay;
            rayMasks[i][UP_LEFT]    = upLeftRay;
            rayMasks[i][UP_RIGHT]   = upRightRay;
            rayMasks[i][DOWN_LEFT]  = downLeftRay;
            rayMasks[i][DOWN_RIGHT] = downRightRay;
        }
    }

    constexpr void initDirectionMapping()
    {
        for (int i = 1; i < 8; ++i) {
            diffToDir[64 + (i * 9)] = UP_RIGHT;   // +9, +18...
            diffToDir[64 + (i * 7)] = UP_LEFT;    // +7, +14...
            diffToDir[64 - (i * 7)] = DOWN_RIGHT; // -7, -14...
            diffToDir[64 - (i * 9)] = DOWN_LEFT;  // -9, -18...
        }
    }

    // lookup tables
    size_t JumpTable[64][4]{};
    size_t NeighbourTable[64][4]{};

    size_t diagonalMaskEx[64]{};
    size_t anitDiagonalMaskEx[64]{};
    size_t rayMasks[64][4]{};

    Direction diffToDir[128]{};
};

constexpr LookupTables globalTables;

#endif // MCTS_CHECKERS_LOOKUP_TABLES_H
