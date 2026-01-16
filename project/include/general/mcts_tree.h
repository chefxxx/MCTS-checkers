//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_MCTS_H
#define MCTS_CHECKERS_MCTS_H

#include <cmath>

#include "board_infra.cuh"
#include "move.h"

// class to manage mcts tree on the cpu

constexpr double C = 2;

struct MctsNode
{
    explicit MctsNode(MctsNode* t_parent, const Move& t_move)
        : move(t_move)
        ,parent(t_parent)
        ,current_score(0)
        ,number_of_visits(0) {}

    // -------------------
    // Game related fields
    // -------------------

    // Move applied in this node
    size_t move_packed_path;

    // draw moves counter
    uint16_t kings_quiet_moves_counter;

    // This board represents game state in this
    // node after applying "move"
    Board board;

    // ---------------
    // Tree management
    // ---------------
    std::vector<MctsNode*> children;
    MctsNode* parent;

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
    [[nodiscard]] double calculate_UCB() const { return current_score / number_of_visits + C * sqrt(log(parent_visits()) / number_of_visits); }
};

struct MctsTree
{
    explicit MctsTree()
    {
    }

    MctsNode* root;
};



#endif // MCTS_CHECKERS_MCTS_H
