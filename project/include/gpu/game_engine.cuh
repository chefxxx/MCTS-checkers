//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include <optional>
#include <string>

#include "board_infra.h"
#include "constants.h"
#include "mcts_tree.h"
#include "move.h"
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

    void playTheGame();
    void aiTurn();
    void playerTurn(bool t_midGame = true);

private:
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

struct AI_Player
{
    AI_Player(const double t_time, const std::string &t_engine, const Colour t_player_colour) :
    time_per_round(t_time)
        , colour(t_player_colour)
        , engine(t_engine) {}


    double time_per_round;
    Colour colour;
    std::string engine;
};

struct ComputerGame
{
    ComputerGame(const AI_Player &t_p1, const AI_Player &t_p2) : ai_player1(t_p1), ai_player2(t_p2)
    {
        if (t_p1.engine == "gpu" || t_p2.engine == "gpu") {
            init_promotion_const_mem();
            init_gpu_movegen_const_mem();
            d_globalScore = mem_cuda::make_unique<double>();
            d_states      = init_random_states();
        }
        board.initStartingBoard();
        P1_mcts_tree.initTree(board, white);
        P2_mcts_tree.initTree(board, white);
    }

    Board board;
    AI_Player ai_player1;
    AI_Player ai_player2;

    MctsTree P1_mcts_tree;
    MctsTree P2_mcts_tree;
    std::vector<PrintingMovePath>     game_hist;
    mem_cuda::unique_ptr<curandState> d_states;
    mem_cuda::unique_ptr<double>      d_globalScore;

    void playTheGame();
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

void printGameHist(const std::vector<PrintingMovePath> &t_hist, std::ostream &os = std::cout);
void printBoard(const Board &t_board, Colour t_perspective);
void printRuleset();
double setTime(const std::string &t_mess);
std::string selectArchitecture(const std::string &t_mess);
Colour      selectColour(const std::string &t_mess);

#endif // MCTS_CHECKERS_GAME_ENGINE_H
