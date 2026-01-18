//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_MCTS_H
#define MCTS_CHECKERS_MCTS_H

#include <algorithm>
#include <cmath>
#include <memory>

#include "board_infra.h"
#include "move.h"

// class to manage mcts tree on the cpu

constexpr double C = 2;

struct MctsNode
{
    explicit MctsNode(MctsNode *t_parent, const Board &t_board, const Colour t_colour)
        : colour_of_player_to_move(t_colour)
        , parent(t_parent)
        , current_score(0)
        , number_of_visits(0)
    {
        board = t_board;
    }

    explicit MctsNode(MctsNode *t_parent, const Board &t_board, const LightMovePath &t_movePath, const Colour t_colour)
        : move_packed_path(t_movePath)
        , colour_of_player_to_move(t_colour)
        , parent(t_parent)
        , current_score(0)
        , number_of_visits(0)
    {
        board = t_board;
    }

    ~MctsNode() = default;

    // -------------------
    // Game related fields
    // -------------------

    // Move applied in this node
    LightMovePath move_packed_path{};

    // This board represents game state in this
    // node after applying "move"
    Board  board{};
    Colour colour_of_player_to_move;

    // ---------------
    // Tree management
    // ---------------
    std::vector<std::unique_ptr<MctsNode>> children{};
    MctsNode                              *parent;

    // ---------------
    // MCTS management
    // ---------------

    // ------------------------------------
    // Wins - 1.0, Loses - 0.0, Draws - 0.5
    //
    // Remember to correctly alternating
    // between nodes
    // ------------------------------------
    double current_score;
    double number_of_visits;

    [[nodiscard]] double parent_visits() const { return parent ? parent->number_of_visits : number_of_visits; }
    [[nodiscard]] double calculate_UCB() const
    {
        if (number_of_visits == 0) {
            return std::numeric_limits<double>::infinity();
        }
        return current_score / number_of_visits + C * sqrt(log(parent_visits()) / number_of_visits);
    }
};


struct MctsTree
{
    // ---------------------------------
    // Creation of empty tree,
    // used at the beginning of the game
    // ---------------------------------
    MctsTree() {}
    explicit MctsTree(const Board &t_board, const Colour t_colour)
        : colour_of_ai(t_colour)
    {
        // Note:
        // I assume that the root node is always the node
        // which is the first bot move in the game
        //
        // so if its is black it will be the node after first player move
        // if white, the starting board
        root = std::make_unique<MctsNode>(nullptr, t_board, t_colour);
    }

    std::unique_ptr<MctsNode> root = nullptr;
    Colour                    colour_of_ai = none;

    ~MctsTree() = default;

    void initTree(const Board &t_board, const Colour t_colour)
    {
        root = std::make_unique<MctsNode>(nullptr, t_board, t_colour);
    }
};

[[nodiscard]] MctsNode *selectNode(MctsNode *t_root);
[[nodiscard]] Board     chooseBestMove(const MctsNode *t_root);
[[nodiscard]] int       rollout();
void                    expandNode(MctsNode *t_node);
void                    backpropagate(MctsNode *t_leaf, double t_score, Colour t_aiColour);


#endif // MCTS_CHECKERS_MCTS_H
