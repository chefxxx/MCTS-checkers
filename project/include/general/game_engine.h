//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_GAME_ENGINE_H
#define MCTS_CHECKERS_GAME_ENGINE_H

#include <optional>
#include <string>

#include "bit_operations.h"
#include "board_infra.h"
#include "constants.h"
#include "mcts_tree.h"
#include "move.h"

struct GameManager
{
    explicit GameManager(const Colour t_perspective, const double t_ai_time, std::string t_mode)
        : m_player_colour(t_perspective)
        , m_ai_colour(static_cast<Colour>(1 - t_perspective))
        , m_mode(t_mode)
        , m_ai_time_per_turn(t_ai_time)
    {
        initPerspective();
        board.initStartingBoard();
        m_tree.initTree(board, m_ai_colour);
    }
    Board board{};

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
                else {
                    std::cout << "|   ";
                }
            }
            std::cout << '|' << rowsNames[visualRow] << '\n';
        }

        std::cout << middleRow;
        std::cout << columnsNamesRow;
    }

    void                        playTheGame();
    void                        aiTurn(LightMovePath t_move);
    static std::optional<Move>  parsePlayerMove();
    [[nodiscard]] LightMovePath playerTurn();

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
    double      m_ai_time_per_turn;

    //
    MctsTree m_tree;
};

Colour drawStartingColour();

#endif // MCTS_CHECKERS_GAME_ENGINE_H
