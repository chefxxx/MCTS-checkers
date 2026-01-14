//
// Created by chefxx on 11.01.2026.
//

#include <cassert>

#include "cpu_movegen.h"
#include "lookup_tables.h"


std::vector<Move> generateAllPossibleMoves(const Board &t_board, const Colour t_color)
{
    std::vector<Move> result;
    // t_colour represents a player that is currently
    // on the move, I do not want to think of this func
    // just form one perspective

    // remember to correctly initialize all variables!
    const size_t pawns = t_board.pawns[t_color];
    const size_t kings = t_board.kings[t_color];
    const size_t pieces = pawns | kings;
    const size_t opponent_pawns = t_board.pawns[1 - t_color];
    const size_t opponent_kings = t_board.kings[1 - t_color];
    const size_t opponent_pieces = opponent_pawns | opponent_kings;
    const size_t all_board_pieces = pieces | opponent_pieces;
    const size_t empty = ~all_board_pieces;

    const size_t pawns_attackers = getPawnsAttackMask(pawns, opponent_pieces, empty);
    const size_t kings_attackers = getKingsAttackMask(kings, all_board_pieces, opponent_pieces);
    if (pawns_attackers || kings_attackers) {
        // if there is a move that is a jump/attack we have to do this move
        createAllPawnsAttacks(result, pawns_attackers, opponent_pieces, empty, t_color);
    }
    else {
        // if there is no attack possibility we can think of sliding moves
        const size_t pawns_movers = getPawnsQuietMovesMask(pawns, empty, t_color);
        createAllPawnsQuietMoves(result, pawns_movers, empty, t_color);
    }
    return result;
}

size_t getPawnsAttackMask(const size_t t_attackerPawns, const size_t t_opponentPieces, const size_t t_emptyFiles)
{
    const size_t attacks_upRight   = (((t_attackerPawns & NOT_FILE_G & NOT_FILE_H) << 9 & t_opponentPieces) << 9 & t_emptyFiles) >> 18;
    const size_t attacks_downRight = (((t_attackerPawns & NOT_FILE_G & NOT_FILE_H) >> 7 & t_opponentPieces) >> 7 & t_emptyFiles) << 14;
    const size_t attacks_upLeft    = (((t_attackerPawns & NOT_FILE_A & NOT_FILE_B) << 7 & t_opponentPieces) << 7 & t_emptyFiles) >> 14;
    const size_t attacks_downLeft  = (((t_attackerPawns & NOT_FILE_A & NOT_FILE_B) >> 9 & t_opponentPieces) >> 9 & t_emptyFiles) << 18;
    return attacks_upRight | attacks_downRight | attacks_upLeft | attacks_downLeft;
}

size_t getPawnsQuietMovesMask(const size_t t_moversPawns, const size_t t_emptyFiles, const Colour t_moversColour)
{
    // It is assumed that program "sees" the board from white perspective
    // so for white pawns only "up" slides are permitted, meaning
    // that for black only "down" slides are permitted.

    const size_t move_upRight   = canMove[t_moversColour][UP_RIGHT] * ((t_moversPawns & NOT_FILE_H) << 9 & t_emptyFiles) >> 9;
    const size_t move_upLeft    = canMove[t_moversColour][UP_LEFT] * ((t_moversPawns & NOT_FILE_A) << 7 & t_emptyFiles) >> 7;
    const size_t move_downRight = canMove[t_moversColour][DOWN_RIGHT] * ((t_moversPawns & NOT_FILE_H) >> 7 & t_emptyFiles) << 7;
    const size_t move_downLeft  = canMove[t_moversColour][DOWN_LEFT] * ((t_moversPawns & NOT_FILE_A) >> 9 & t_emptyFiles) << 9;
    return move_upRight | move_upLeft | move_downRight | move_downLeft;
}

// https://www.chessprogramming.org/Hyperbola_Quintessence
size_t diagonalKingMask(const size_t t_boardState, const int t_kingSquare)
{
    size_t forward = t_boardState & globalTables.diagonalMaskEx[t_kingSquare];
    size_t reverse = byte_swap64(forward);
    forward -= 1ULL << t_kingSquare;
    reverse -= byte_swap64(1ULL << t_kingSquare);
    forward ^= byte_swap64(reverse);
    forward &= globalTables.diagonalMaskEx[t_kingSquare];
    return forward;
}

// https://www.chessprogramming.org/Hyperbola_Quintessence
size_t antiDiagonalKingMask(const size_t t_boardState, const int t_kingSquare)
{
    size_t forward = t_boardState & globalTables.anitDiagonalMaskEx[t_kingSquare];
    size_t reverse = byte_swap64(forward);
    forward -= 1ULL << t_kingSquare;
    reverse -= byte_swap64(1ULL << t_kingSquare);
    forward ^= byte_swap64(reverse);
    forward &= globalTables.anitDiagonalMaskEx[t_kingSquare];
    return forward;
}

size_t bothDiagonalsKingMask(const size_t t_boardState, const int t_kingSquare)
{
    return diagonalKingMask(t_boardState, t_kingSquare) | antiDiagonalKingMask(t_boardState, t_kingSquare);
}

size_t getKingsAttackMask(size_t t_kingsMask, const size_t t_boardState, const size_t t_opponentsPieces)
{
    const size_t empty = ~t_boardState;
    size_t result_kings = 0ULL;
    while (t_kingsMask) {
        const int k_idx = popLsb(t_kingsMask);
        const size_t mask = bothDiagonalsKingMask(t_boardState, k_idx);
        size_t possible_victims = mask & t_opponentsPieces;
        while (possible_victims) {
            const int v_idx = popLsb(possible_victims);
            const int diff = v_idx - k_idx;
            if (const Direction dir = globalTables.diffToDir[diff + 64]; globalTables.NeighbourTable[v_idx][dir] & empty) {
                result_kings |= 1ULL << k_idx;
                break;
            }
        }
    }
    return result_kings;
}

void createAllPawnsAttacks(std::vector<Move> &t_allMoves,
                           const size_t       t_attackersMask,
                           const size_t       t_opponentPieces,
                           const size_t       t_emptyFiles,
                           const Colour       t_attackersColour)
{
    size_t maskCopy = t_attackersMask;
    while (maskCopy) {
        const int attacker_idx = popLsb(maskCopy);
        std::vector attackPath{attacker_idx};
        recursiveCreatePawnsAttacks(t_allMoves,
                                    attackPath,
                                    attacker_idx,
                                    t_opponentPieces,
                                    t_emptyFiles,
                                    0ULL,
                                    1ULL << attacker_idx,
                                    t_attackersColour);
    }
}

void recursiveCreatePawnsAttacks(std::vector<Move>& t_allMoves,
                                 std::vector<int>&  t_currentPath,
                                 const int          t_idx,
                                 const size_t       t_opponentPieces,
                                 const size_t       t_emptyFiles,
                                 const size_t       t_currentVictimsMask,
                                 const size_t       t_originalStartingPositionMask,
                                 const Colour       t_attackerColour)
{
    bool foundJump = false;

    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        const size_t victim_mask = globalTables.NeighbourTable[t_idx][dir];
        size_t jump_mask   = globalTables.JumpTable[t_idx][dir];

        // Note:
        // 1. Check if in the dir there is an opponent piece
        // 2. Check if it was taken earlier on move path
        // 3. Check if we can jump over it on empty file
        if (victim_mask & t_opponentPieces &&
            !(victim_mask & t_currentVictimsMask) &&
            jump_mask & t_emptyFiles) {
            // update masks
            foundJump = true;

            const auto new_empty = (t_emptyFiles | 1ULL << t_idx) & ~jump_mask;
            const auto new_currentVictims = t_currentVictimsMask | victim_mask;

            const auto new_idx = popLsb(jump_mask);
            t_currentPath.push_back(new_idx);
            recursiveCreatePawnsAttacks(t_allMoves,
                                        t_currentPath,
                                        new_idx,
                                        t_opponentPieces,
                                        new_empty,
                                        new_currentVictims,
                                        t_originalStartingPositionMask,
                                        t_attackerColour);
            t_currentPath.pop_back();
        }
    }
    // add new move to vector
    if (!foundJump) {
        // Note:
        // If we are in this branch, it means that "for" loop
        // did not find any moves. But this could've happened
        // only at the second level of recursion tree, because
        // of the move generation construction. So the t_idx here
        // is for sure some index after at least one jump was made.
        assert(t_currentVictimsMask);
        const size_t to_mask = 1ULL << t_idx;
        t_allMoves.emplace_back(t_originalStartingPositionMask, to_mask, t_currentVictimsMask, to_mask & promotion[t_attackerColour], t_currentPath);
    }
}

void createAllPawnsQuietMoves(std::vector<Move> &t_allMoves,
                        const size_t       t_moversMask,
                        const size_t       t_emptyFiles,
                        const Colour       t_moversColour)
{
    size_t maskCopy = t_moversMask;
    while (maskCopy) {
        const int mover_idx = popLsb(maskCopy);
        createOnePawnQuietMoves(t_allMoves, mover_idx, t_emptyFiles, t_moversColour);
    }
}

void createOnePawnQuietMoves(std::vector<Move> &t_allMoves,
                        const int          t_idx,
                        const size_t       t_emptyFiles,
                        const Colour       t_moversColour)
{
    // check directions of moves and create them
    for (const int dir : {UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT}) {
        // ReSharper disable once CppTooWideScope
        const size_t move_mask = canMove[t_moversColour][dir] * globalTables.NeighbourTable[t_idx][dir] & t_emptyFiles;
        if (move_mask) {
            size_t copy = move_mask;
            const auto move = Move(1ULL << t_idx, move_mask, move_mask & promotion[t_moversColour], t_idx, popLsb(copy));
            t_allMoves.push_back(move);
        }
    }
}