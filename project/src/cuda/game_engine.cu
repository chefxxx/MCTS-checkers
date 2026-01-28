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
        printBoard(board, m_player_colour);
        playerTurn(false);
        turn = black;
    }

    mcts_tree.initTree(board, m_ai_colour);

    while (state == CONTINUES) {
        printGameHist(game_hist);
        printBoard(board, m_player_colour);

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
    printBoard(board, m_player_colour);
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
        printGameHist(game_hist, outFile);
        outFile.close();
        std::cout << "Game history saved to: " << filename.str() << std::endl;
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

void   ComputerGame::playTheGame()
{
    Colour turn = white;
    GameState state = CONTINUES;

    while (state == CONTINUES) {
        // print always from the white player perspective
        printGameHist(game_hist);
        if (!game_hist.empty()) {
            auto just_played = static_cast<Colour>(1 - turn);
            std::cout << "The " << just_played << " player played the move " << game_hist.back() << '\n';
        }
        printBoard(board, white);

        MctsTree& actor_tree = (turn == ai_player1.colour) ? P1_mcts_tree : P2_mcts_tree;
        MctsTree& observer_tree = (turn == ai_player1.colour) ? P2_mcts_tree : P1_mcts_tree;
        AI_Player& actor = (turn == ai_player1.colour) ? ai_player1 : ai_player2;

        const MctsNode* bestActorNode = runMctsSimulation(actor_tree, actor.time_per_round, actor.engine, d_states, d_globalScore);

        board = bestActorNode->current_board_state;
        const LightMovePath move = bestActorNode->packed_positions_transition;
        actor_tree.updateRoot(bestActorNode);
        observer_tree.updateTree();
        const auto syncNode = findPlayerMove(observer_tree.root.get(), board, move);
        assert(syncNode != nullptr);
        observer_tree.updateRoot(syncNode);

        game_hist.emplace_back(move.packed_data);

        state = checkEndOfGameConditions(board, turn);
        if (state != CONTINUES) break;

        turn = static_cast<Colour>(1 - turn);
    }
    printBoard(board, white);
    std::cout << "The " << turn << " won the game!\n";

    auto        now      = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream filename;
    filename << "../../games/game_history_" << std::put_time(std::localtime(&now_time), "%Y-%m-%d_%H-%M-%S") << ".txt";
    if (std::ofstream outFile(filename.str()); outFile.is_open()) {
        printGameHist(game_hist, outFile);
        outFile.close();
        std::cout << "Game history saved to: " << filename.str() << std::endl;
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

void printGameHist(const std::vector<PrintingMovePath> &t_hist, std::ostream &os)
{
    const std::string mess_box = "**** GAME HISTORY ****";

    os << mess_box << '\n';

    for (const auto &entry : t_hist) {
        os << '*';

        // Logic for calculating lengths and padding
        const size_t len = entry.positions.size() * 2 + entry.positions.size() - 1;
        const int    pad = (mess_box.size() - len - 1) / 2;

        os << std::string(pad, ' ') << entry << std::string(mess_box.size() - pad - len - 2, ' ');

        os << "*\n";
    }

    os << mess_box << '\n';
}


void printBoard(const Board &t_board, const Colour t_perspective)
{
    // perspective and printing utils
    std::array<char, 8> rowsNames{};
    std::string         columnsNamesRow;

    if (t_perspective == black) {
        columnsNamesRow = "    h   g   f   e   d   c   b   a \n";
        rowsNames       = {'1', '2', '3', '4', '5', '6', '7', '8'};
    }
    else {
        columnsNamesRow = "    a   b   c   d   e   f   g   h \n";
        rowsNames       = {'8', '7', '6', '5', '4', '3', '2', '1'};
    }

    const std::string middleRow = " +---+---+---+---+---+---+---+---+\n";

    // Remember that rocks may overlap each other, this func just prints
    std::cout << columnsNamesRow;
    for (int visualRow = 0; visualRow < 8; ++visualRow) {
        std::cout << middleRow;
        std::cout << rowsNames[visualRow];
        for (int visualCol = 0; visualCol < 8; ++visualCol) {
            int actualRow, actualCol;
            if (t_perspective == black) {
                actualRow = visualRow;
                actualCol = 7 - visualCol;
            }
            else {
                actualRow = 7 - visualRow;
                actualCol = visualCol;
            }
            if (const int bitIdx = actualRow * 8 + actualCol; checkBitAtIdx(t_board.pawns[white], bitIdx)) {
                std::cout << "| w ";
            }
            else if (checkBitAtIdx(t_board.pawns[black], bitIdx)) {
                std::cout << "| b ";
            }
            else if (checkBitAtIdx(t_board.kings[white], bitIdx)) {
                std::cout << "| W ";
            }
            else if (checkBitAtIdx(t_board.kings[black], bitIdx)) {
                std::cout << "| B ";
            }
            else {
                std::cout << "|   ";
            }
        }
        std::cout << '|' << rowsNames[visualRow] << '\n';
    }

    std::cout << middleRow;
    std::cout << columnsNamesRow;
}

void printRuleset()
{
    std::cout << "******************************************* RULESET *******************************************\n";
    std::cout << "* 0. Welcome to checkers.mcts!                                                                *\n";
    std::cout << "* 1. Quiet moves are marked as (from)-(to).                                                   *\n";
    std::cout << "* 2. Attack moves are marked as (from):(mid):(to).                                            *\n";
    std::cout << "* 3. Pawns move in one direction, diagonally, one square. Pawns can attack in any direction.  *\n";
    std::cout << "* 4. If pawn ends its move on the opposite last rank, it becomes a king.                      *\n";
    std::cout << "* 5. Kings move ina any direction, diagonally, any number of squares.                         *\n";
    std::cout << "* 6. There is no limit of taking opponents pieces in one attack move.                         *\n";
    std::cout << "* 7. At any point of the game you can resign, by writing 'resign'.                            *\n";
    std::cout << "* 8. Have fun!                                                                                *\n";
    std::cout << "******************************************* RULESET *******************************************\n";
}

double setTime(const std::string &t_mess)
{
    std::string resTimeStr;
    const std::string mess_box = "************************* TIME SETTINGS *************************";
    const std::string mid_box  = "*                                                               *\n";
    const std::string full_prompt = "Please enter the time for the " + t_mess + " in seconds.";

    const int total_width = mess_box.size();
    const int inner_width = total_width - 2;
    const int pad_left = (inner_width - full_prompt.size()) / 2;
    const int pad_right = inner_width - full_prompt.size() - pad_left;

    std::cout << mess_box << '\n';
    std::cout << mid_box;
    std::cout << '*'
       << std::string(pad_left, ' ')
       << full_prompt
       << std::string(pad_right, ' ')
       << "*\n";
    std::cout << mid_box;
    std::cout << mess_box << "\n";
    std::cout << "ENTER THE TIME: ";
    std::getline(std::cin, resTimeStr);
    const auto resTime = std::stod(resTimeStr);
    return resTime;
}

std::string selectArchitecture(const std::string &t_mess)
{
    std::string arch;

    const std::string mess_box = "****************************** ENGINE SETTINGS ******************************";
    const std::string mid_box  = "*                                                                           *\n";
    const std::string full_prompt = "Please enter the engine for the " + t_mess + " - write 'gpu' or 'cpu'.";

    const int total_width = mess_box.size();
    const int inner_width = total_width - 2;
    const int pad_left = (inner_width - full_prompt.size()) / 2;
    const int pad_right = inner_width - full_prompt.size() - pad_left;

    std::cout << mess_box << '\n';
    std::cout << mid_box;
    std::cout << '*'
       << std::string(pad_left, ' ')
       << full_prompt
       << std::string(pad_right, ' ')
       << "*\n";
    std::cout << mid_box;
    std::cout << mess_box << "\n";

    std::cout << "ENTER THE ENGINE: ";
    std::getline(std::cin, arch);
    if (arch != "cpu" && arch != "gpu") {
        logger::err("Wrong engine type entered, try again!\n");
        exit(EXIT_FAILURE);
    }
    return arch;
}

Colour selectColour(const std::string &t_mess)
{
    std::string colour;
    const std::string mess_box = "****************************** COLOUR SETTINGS ******************************";
    const std::string mid_box  = "*                                                                           *\n";
    const std::string full_prompt = "Please enter the colour for the " + t_mess + " - write 'b' or 'w'.";

    const int total_width = mess_box.size();
    const int inner_width = total_width - 2;
    const int pad_left = (inner_width - full_prompt.size()) / 2;
    const int pad_right = inner_width - full_prompt.size() - pad_left;

    std::cout << mess_box << '\n';
    std::cout << mid_box;
    std::cout << '*'
       << std::string(pad_left, ' ')
       << full_prompt
       << std::string(pad_right, ' ')
       << "*\n";
    std::cout << mid_box;
    std::cout << mess_box << "\n";
    std::cout << "ENTER THE COLOUR: ";
    std::getline(std::cin, colour);
    if (colour != "w" && colour != "b" ) {
        logger::err("Wrong colour entered, try again!\n");
        exit(EXIT_FAILURE);
    }
    if (colour == "w") {
        return white;
    }
    return black;
}