//
// Created by chefxx on 10.01.2026.
//

#include "game_engine.h"

#include <cassert>
#include <random>

#include "checkers_engine.h"
#include "logger.h"
#include "simulation.h"

void GameManager::playTheGame()
{

}

MctsNode *GameManager::aiTurn()
{
    MctsNode* bestNode = nullptr;
    if (m_mode == "cpu") {
        bestNode = runCPU_MCTS(mcts_tree, m_ai_time_per_turn);
    }
    else {
        throw "Not implemented!\n";
    }
    assert(bestNode != nullptr);
    return bestNode;
}


std::optional<Move> parsePlayerMove(const Board &t_board, const Colour t_colour)
{
    logger::info("Please enter your move...\n");
    std::string moveStr;
    std::getline(std::cin, moveStr);
    if (moveStr.size() < 5) {
        logger::warn("Wrong move format, try again!\n");
        return std::nullopt;
    }

    return processMoveString(moveStr, t_board, t_colour);
}

std::tuple<LightMovePath, Board> GameManager::playerTurn() const
{
    while (true) {
        const auto move = parsePlayerMove(board, m_player_colour);
        if (move == std::nullopt) {
            continue;
        }
        const auto n_board = applyMove(board, move.value(), m_player_colour);
        if (n_board == std::nullopt) {
            continue;
        }

        return std::forward_as_tuple(move.value().path, n_board.value());
    }
}

Colour drawStartingColour()
{
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen) == 0 ? black : white;
}

std::optional<Move> processMoveString(const std::string &t_moveStr, const Board &t_currBoard, const Colour t_colour)
{
    const size_t opponent =  t_currBoard.pawns[1 - t_colour] | t_currBoard.kings[1 - t_colour];
    const size_t empty = ~(t_currBoard.pawns[t_colour] | t_currBoard.kings[t_colour] | opponent);
    if (t_moveStr[2] == '-') {
        // normal move case
        const std::string from    = t_moveStr.substr(0, 2);
        const std::string to      = t_moveStr.substr(3, 2);
        const int         fromIdx = strToPos(from);
        const int         toIdx   = strToPos(to);
        if (!checkBitAtIdx(t_currBoard.pawns[t_colour], fromIdx) && !checkBitAtIdx(t_currBoard.kings[t_colour], fromIdx)) {
            logger::warn("Wrong move quiet 'from' position, try again..\n");
            return std::nullopt;
        }
        if (!checkBitAtIdx(empty, toIdx)) {
            logger::warn("Wrong move quiet 'to' position, try again..\n");
            return std::nullopt;
        }
        return Move(fromIdx, toIdx);
    }

    std::vector<int> positions;
    size_t captures = 0ULL;
    for (size_t i = 0; i < t_moveStr.size(); i += 3) {
        if (i + 1 < t_moveStr.size() && std::isalpha(t_moveStr[i]) && std::isdigit(t_moveStr[i + 1])) {
            if (i + 2 < t_moveStr.size() && t_moveStr[i + 2] != ':') {
                logger::warn("Wrong move format, try again\n");
                return std::nullopt;
            }
            const std::string strPos = t_moveStr.substr(i, i + 2);
            const int         idx  = strToPos(strPos);
            positions.push_back(idx);
        }
        else {
            logger::warn("Wrong move format, try again\n");
            return std::nullopt;
        }
    }

    if (checkBitAtIdx(t_currBoard.pawns[t_colour],positions[0])) {
        // pawn takes
        for (size_t i = 1; i < positions.size(); ++i) {
            const int diff = positions[i] - positions[i - 1];
            const Direction dir = globalTables.diffToDir[64 + diff];
            if (const size_t mask = globalTables.NeighbourTable[positions[i - 1]][dir]; mask & opponent) {
                captures |= mask;
            }
            else {
                logger::warn("Wrong piece to attack position, try again..\n");
                return std::nullopt;
            }
        }
    }
    else if (checkBitAtIdx(t_currBoard.kings[t_colour],positions[0])) {
        // king takes
        for (size_t i = 1; i < positions.size(); ++i) {
            const int diff = positions[i] - positions[i - 1];
            const Direction dir = globalTables.diffToDir[64 + diff];
            if (const size_t mask = bothDiagonalsKingMask(~empty, positions[i - 1])
                                  & globalTables.rayMasks[positions[i - 1]][dir] & opponent) {
                captures |= mask;
            }
            else {
                logger::warn("Something wrong..\n");
                return std::nullopt;
            }
        }
    }
    else {
        logger::warn("Wrong attack move 'from' position, try again..\n");
        return std::nullopt;
    }
    return Move(captures, positions);
}
