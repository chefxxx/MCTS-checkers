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
    GameState state = CONTINUES;
    Colour    turn  = white;

    // Note:
    // game state is perceived from the players perspective,
    // meaning that is state == LOST, then player lost etc.
    LightMovePath lastPlayerMove;
    while (state == CONTINUES) {
        printBoard();
        if (turn == m_player_colour) {
            lastPlayerMove = playerTurn();
        }
        else {
            aiTurn(lastPlayerMove);
        }
        state = checkEndOfGameConditions(board, turn);
        turn  = static_cast<Colour>(1 - turn);
    }

    std::cout << "You " << state << "!\n";
}

void GameManager::aiTurn(const LightMovePath t_move)
{
    if (board != m_tree.root->board) {
        const auto new_root = findPlayerMove(m_tree.root.get(), board, t_move);
        assert(new_root != nullptr);
        m_tree.updateRoot(new_root);
    }
    if (m_mode == "cpu") {
        board = runCPU_MCTS(m_tree, m_ai_time_per_turn);
    }
    else {
        throw "Not implemented!\n";
    }
}

std::optional<Move> GameManager::parsePlayerMove()
{
    logger::info("Please enter your move...\n");
    std::string moveStr;
    std::getline(std::cin, moveStr);
    if (moveStr.size() < 5) {
        logger::warn("Wrong move format, try again!\n");
        return std::nullopt;
    }
    if (moveStr[2] == '-') {
        // normal move case
        const std::string from    = moveStr.substr(0, 2);
        const std::string to      = moveStr.substr(3, 2);
        const int         fromIdx = strToPos(from);
        const int         toIdx   = strToPos(to);
        return Move(fromIdx, toIdx);
    }
    std::vector<int> positions;
    for (size_t i = 0; i < moveStr.size(); i += 3) {
        if (i + 1 < moveStr.size() && std::isalpha(moveStr[i]) && std::isdigit(moveStr[i + 1])) {
            if (i + 2 < moveStr.size() && moveStr[i + 2] != ':') {
                logger::warn("Wrong move format, try again\n");
                return std::nullopt;
            }
            const std::string strPos = moveStr.substr(i, i + 2);
            const int         toIdx  = strToPos(strPos);
            positions.push_back(toIdx);
        }
        else {
            logger::warn("Wrong move format, try again\n");
            return std::nullopt;
        }
    }
    return Move(positions);
}

LightMovePath GameManager::playerTurn()
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
        board = n_board.value();
        return LightMovePath(move.value().positions, move.value().captures_mask > 0);
    }
}

Colour drawStartingColour()
{
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen) == 0 ? black : white;
}
