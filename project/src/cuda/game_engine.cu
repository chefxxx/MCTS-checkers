//
// Created by chefxx on 10.01.2026.
//

#include <cassert>
#include <random>
#include <sys/stat.h>

#include "checkers_engine.h"
#include "game_engine.cuh"
#include "gpu_rollout.cuh"
#include "logger.h"

void GameManager::playTheGame()
{
    Colour    turn  = white;
    GameState state = CONTINUES;

    if (m_player_colour == white) {
        printBoard();
        playerTurn(false);
        turn = black;
    }

    mcts_tree.initTree(board, m_ai_colour);

    while (state == CONTINUES) {
        printGameHist();
        printBoard();

        if (turn == m_player_colour) {
            // player
            playerTurn();
        }
        else {
            // mcts
            aiTurn();
        }

        state = checkEndOfGameConditions(board, turn);
        turn  = static_cast<Colour>(1 - turn);
    }
    printBoard();
    std::string mess = "Maybe next time, you lost!\n";
    if (state == DRAW)
        mess = "Game ended in a draw!\n";
    if (turn == m_ai_colour) {
        mess = "Congratulations, You won!\n";
    }
    std::cout << mess;

    auto        now      = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    std::ostringstream filename;
    filename << "../../games/game_history_" << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S") << ".txt";
    if (std::ofstream outFile(filename.str()); outFile.is_open()) {
        printGameHist(outFile);
        outFile.close();
        std::cout << "History saved to: " << filename.str() << std::endl;
    }
}

void GameManager::aiTurn()
{
    MctsNode *bestNode = nullptr;
    bestNode           = runMctsSimulation(mcts_tree, m_timePerTurn, m_mode, d_states, d_globalScore);
    assert(bestNode != nullptr);
    board = bestNode->current_board_state;
    mcts_tree.updateRoot(bestNode);
    game_hist.emplace_back(bestNode->packed_positions_transition.packed_data);
}

std::optional<Move> parsePlayerMove(const Board &t_board, const Colour t_colour)
{
    logger::info("Please enter your move...\n");
    std::string moveStr;
    std::getline(std::cin, moveStr);
    if (moveStr == "resign") {
        std::cout << "I guess the bot is too strong, but you can try again!\n";
        exit(EXIT_SUCCESS);
    }
    if (moveStr.size() < 5) {
        logger::warn("Wrong move format, try again!\n");
        return std::nullopt;
    }

    return processMoveString(moveStr, t_board, t_colour);
}

void GameManager::playerTurn(const bool t_midGame)
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

        board = n_board.value();
        game_hist.emplace_back(move->packed_positions.packed_data);
        if (t_midGame) {
            mcts_tree.updateTree(); // ensure that this move exists in the tree
            const auto node = findPlayerMove(mcts_tree.root.get(), board, move->packed_positions);
            assert(node != nullptr);
            mcts_tree.updateRoot(node);
        }
        return;
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
    const size_t opponent    = t_currBoard.pawns[1 - t_colour] | t_currBoard.kings[1 - t_colour];
    const size_t empty       = ~(t_currBoard.pawns[t_colour] | t_currBoard.kings[t_colour] | opponent);
    const auto   check_moves = generateAllPossibleMoves(t_currBoard, t_colour);
    Move         result;

    if (t_moveStr[2] == '-') {
        // normal move case
        const std::string from    = t_moveStr.substr(0, 2);
        const std::string to      = t_moveStr.substr(3, 2);
        const int         fromIdx = strToPos(from);
        const int         toIdx   = strToPos(to);
        if (!checkBitAtIdx(t_currBoard.pawns[t_colour], fromIdx)
            && !checkBitAtIdx(t_currBoard.kings[t_colour], fromIdx)) {
            logger::warn("Wrong move quiet 'from' position, try again..\n");
            return std::nullopt;
        }
        if (!checkBitAtIdx(empty, toIdx)) {
            logger::warn("Wrong move quiet 'to' position, try again..\n");
            return std::nullopt;
        }
        result = Move(fromIdx, toIdx);
    }
    else {
        std::vector<int> positions;
        size_t           captures = 0ULL;
        for (size_t i = 0; i < t_moveStr.size(); i += 3) {
            if (i + 1 < t_moveStr.size() && std::isalpha(t_moveStr[i]) && std::isdigit(t_moveStr[i + 1])) {
                if (i + 2 < t_moveStr.size() && t_moveStr[i + 2] != ':') {
                    logger::warn("Wrong move format, try again\n");
                    return std::nullopt;
                }
                const std::string strPos = t_moveStr.substr(i, i + 2);
                const int         idx    = strToPos(strPos);
                positions.push_back(idx);
            }
            else {
                logger::warn("Wrong move format, try again\n");
                return std::nullopt;
            }
        }

        if (checkBitAtIdx(t_currBoard.pawns[t_colour], positions[0])) {
            // pawn takes
            for (size_t i = 1; i < positions.size(); ++i) {
                const int       diff = positions[i] - positions[i - 1];
                const Direction dir  = globalTables.diffToDir[64 + diff];
                if (const size_t mask = globalTables.NeighbourTable[positions[i - 1]][dir]; mask & opponent) {
                    captures |= mask;
                }
                else {
                    logger::warn("Wrong piece to attack position, try again..\n");
                    return std::nullopt;
                }
            }
        }
        else if (checkBitAtIdx(t_currBoard.kings[t_colour], positions[0])) {
            // king takes
            for (size_t i = 1; i < positions.size(); ++i) {
                const int       diff = positions[i] - positions[i - 1];
                const Direction dir  = globalTables.diffToDir[64 + diff];
                const size_t    mask = bothDiagonalsKingMask(~empty, positions[i - 1])
                                  & globalTables.rayMasks[positions[i - 1]][dir] & opponent;
                if (mask > 0 && (1ULL << positions[i] & empty) > 0) {
                    captures |= mask;
                }
                else {
                    logger::warn("Something wrong for king move..\n");
                    return std::nullopt;
                }
            }
        }
        else {
            logger::warn("Wrong attack move 'from' position, try again..\n");
            return std::nullopt;
        }
        result = Move(captures, positions);
    }
    const auto it = std::ranges::find_if(check_moves, [result](const Move &mv) { return mv == result; });
    if (it == check_moves.end()) {
        logger::warn("Not allowed move, try again..\n");
        return std::nullopt;
    }
    return result;
}

MctsNode *runMctsSimulation(const MctsTree                          &t_tree,
                            const double                             t_timeLimit,
                            const std::string                       &t_mode,
                            const mem_cuda::unique_ptr<curandState> &t_states,
                            const mem_cuda::unique_ptr<double>      &t_globalScore)
{
    if (t_tree.root->is_solved()) {
        return chooseBestMove(t_tree);
    }
    int CHECK = -1;
    if (t_mode == "cpu") {
        CHECK = CPU_ITERATION_CHECK;
    }
    else {
        CHECK = GPU_ITERATION_CHECK;
    }
    assert(static_cast<long long>(t_timeLimit * 1e6 * TURN_TIME_MULTIPLICATOR) > 0);
    const auto limit_micro_sec =
        std::chrono::microseconds(static_cast<long long>(t_timeLimit * 1e6 * TURN_TIME_MULTIPLICATOR));
    int        iters = 0;
    const auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        mctsIteration(t_tree, t_mode, t_states, t_globalScore);
        iters++;
        if (iters % CHECK == 0) {
            if (t_tree.root->is_solved()) {
                return chooseBestMove(t_tree);
            }
            auto now = std::chrono::high_resolution_clock::now();
            if (const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start);
                elapsed > limit_micro_sec) {
                break;
            }
        }
    }
    return chooseBestMove(t_tree);
}

void mctsIteration(const MctsTree                                           &t_tree,
                   const std::string                                        &t_mode,
                   [[maybe_unused]] const mem_cuda::unique_ptr<curandState> &t_states,
                   [[maybe_unused]] const mem_cuda::unique_ptr<double>      &t_globalScore)
{
    // 1. selection
    auto selectedNode = selectNode(t_tree.root.get());

    // 2. expansion
    if (!selectedNode->is_solved() && !selectedNode->is_fully_expanded())
        selectedNode = expandNode(selectedNode);

    // 3. rollout or score
    double score;
    if (selectedNode->is_solved()) {
        if (selectedNode->status == NodeStatus::WIN)
            score = 0.0;
        else if (selectedNode->status == NodeStatus::LOSS)
            score = 1.0;
        else
            score = 0.5;
    }
    else {
        if (t_mode == "cpu")
            score = rollout(selectedNode);
        else
            score = rollout_gpu(selectedNode, t_states, t_globalScore);
    }

    // 4. backpropagate
    backpropagate(selectedNode, score);
}