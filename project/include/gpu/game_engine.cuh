//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include <optional>
#include <string>

#include "../cpu/board_infra.h"
#include "../cpu/constants.h"
#include "../cpu/mcts_tree.h"
#include "../cpu/move.h"
#include "bit_operations.cuh"
#include "gpu_checkers_engine.cuh"
#include "gpu_infa_kernels.cuh"
#include "gpu_movegen.cuh"
#include "memory_cuda.cuh"

struct GameManager
{
    explicit GameManager(const Colour t_perspective, const double t_ai_time, const std::string &t_mode)
        : m_player_colour(t_perspective)
        , m_ai_colour(static_cast<Colour>(1 - t_perspective))
        , m_mode(t_mode)
        , m_timePerTurn(t_ai_time)
    {
        initPerspective();
        board.initStartingBoard();
        if (m_mode == "gpu") {
            init_promotion_const_mem();
            init_gpu_movegen_const_mem();
            d_globalScore = mem_cuda::make_unique<double>();
            d_states      = init_random_states();
        }
    }

    MctsTree                      mcts_tree;
    Board                         board;
    std::vector<PrintingMovePath> game_hist;

    void printGameHist(std::ostream &os = std::cout) const
    {
        const std::string mess_box = "**** GAME HISTORY ****";

        os << mess_box << '\n';

        for (const auto &entry : game_hist) {
            os << '*';

            // Logic for calculating lengths and padding
            const size_t len = entry.positions.size() * 2 + entry.positions.size() - 1;
            const int    pad = (mess_box.size() - len - 1) / 2;

            os << std::string(pad, ' ') << entry << std::string(mess_box.size() - pad - len - 2, ' ');

            os << "*\n";
        }

        os << mess_box << '\n';
    }

    void printBoard() const
    {
        const std::string middleRow = " +---+---+---+---+---+---+---+---+\n";

        // Remember that rocks may overlap each other, this func just prints
        std::cout << columnsNamesRow;
        for (int visualRow = 0; visualRow < 8; ++visualRow) {
            std::cout << middleRow;
            std::cout << rowsNames[visualRow];
            for (int visualCol = 0; visualCol < 8; ++visualCol) {
                int actualRow, actualCol;
                if (m_player_colour == black) {
                    actualRow = visualRow;
                    actualCol = 7 - visualCol;
                }
                else {
                    actualRow = 7 - visualRow;
                    actualCol = visualCol;
                }
                if (const int bitIdx = actualRow * 8 + actualCol; checkBitAtIdx(board.pawns[white], bitIdx)) {
                    std::cout << "| w ";
                }
                else if (checkBitAtIdx(board.pawns[black], bitIdx)) {
                    std::cout << "| b ";
                }
                else if (checkBitAtIdx(board.kings[white], bitIdx)) {
                    std::cout << "| W ";
                }
                else if (checkBitAtIdx(board.kings[black], bitIdx)) {
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

    void playTheGame();
    void aiTurn();
    void playerTurn(bool t_midGame = true);

private:
    void initPerspective()
    {
        // perspective only affects printing
        if (m_player_colour == black) {
            columnsNamesRow = "    h   g   f   e   d   c   b   a \n";
            rowsNames       = {'1', '2', '3', '4', '5', '6', '7', '8'};
        }
        else {
            columnsNamesRow = "    a   b   c   d   e   f   g   h \n";
            rowsNames       = {'8', '7', '6', '5', '4', '3', '2', '1'};
        }
    }

    // perspective and printing utils
    std::array<char, 8> rowsNames{};
    std::string         columnsNamesRow;

    // -------------
    // game settings
    // -------------
    Colour      m_player_colour;
    Colour      m_ai_colour;
    std::string m_mode;
    double      m_timePerTurn;

    //
    mem_cuda::unique_ptr<curandState> d_states;
    mem_cuda::unique_ptr<double>      d_globalScore;
};

Colour              drawStartingColour();
std::optional<Move> parsePlayerMove(const Board &t_board, Colour t_colour);
std::optional<Move> processMoveString(const std::string &t_moveStr, const Board &t_currBoard, Colour t_colour);

MctsNode *runMctsSimulation(const MctsTree                          &t_tree,
                            double                                   t_timeLimit,
                            const std::string                       &t_mode,
                            const mem_cuda::unique_ptr<curandState> &t_states,
                            const mem_cuda::unique_ptr<double>      &t_globalScore);
void      mctsIteration(const MctsTree                          &t_tree,
                        const std::string                       &t_mode,
                        const mem_cuda::unique_ptr<curandState> &t_states,
                        const mem_cuda::unique_ptr<double>      &t_globalScore);

#endif // MCTS_CHECKERS_GAME_ENGINE_H
