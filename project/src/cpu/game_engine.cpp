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
        bestNode = runCPU_MCTS(m_tree, m_ai_time_per_turn);
    }
    else {
        throw "Not implemented!\n";
    }
    assert(bestNode != nullptr);
    return bestNode;
}


std::optional<Move> parsePlayerMove()
{
    logger::info("Please enter your move...\n");
    std::string moveStr;
    std::getline(std::cin, moveStr);
    if (moveStr.size() < 5) {
        logger::warn("Wrong move format, try again!\n");
        return std::nullopt;
    }

    return processMoveString(moveStr);
}

std::tuple<LightMovePath, Board> GameManager::playerTurn() const
{
    while (true) {
        const auto move = parsePlayerMove();
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

std::optional<Move> processMoveString(const std::string &t_moveStr)
{
    if (t_moveStr[2] == '-') {
        // normal move case
        const std::string from    = t_moveStr.substr(0, 2);
        const std::string to      = t_moveStr.substr(3, 2);
        const int         fromIdx = strToPos(from);
        const int         toIdx   = strToPos(to);
        return Move(fromIdx, toIdx);
    }

    std::vector<int> positions;
    int visited = 0;
    size_t captures = 0ULL;
    for (size_t i = 0; i < t_moveStr.size(); i += 3) {
        if (i + 1 < t_moveStr.size() && std::isalpha(t_moveStr[i]) && std::isdigit(t_moveStr[i + 1])) {
            if (i + 2 < t_moveStr.size() && t_moveStr[i + 2] != ':') {
                logger::warn("Wrong move format, try again\n");
                return std::nullopt;
            }
            const std::string strPos = t_moveStr.substr(i, i + 2);
            const int         toIdx  = strToPos(strPos);
            positions.push_back(toIdx);
            if (i > 0) {
                const int diff = positions[visited + 1] - positions[visited];
                const Direction dir = globalTables.diffToDir[diff];
                const size_t captured = globalTables.NeighbourTable[positions[visited]][dir];
                captures |= captured;
                visited++;
            }
        }
        else {
            logger::warn("Wrong move format, try again\n");
            return std::nullopt;
        }
    }
    return Move(captures, positions);
}
