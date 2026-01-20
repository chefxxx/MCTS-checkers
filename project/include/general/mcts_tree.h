//
// Created by chefxx on 10.01.2026.
//

#ifndef MCTS_CHECKERS_MCTS_H
#define MCTS_CHECKERS_MCTS_H

#include <algorithm>
#include <cmath>
#include <memory>

#include "board_infra.h"
#include "cpu_movegen.h"
#include "move.h"

enum class NodeStatus { SEARCHING, WIN, LOSS, DRAW };

// class to manage mcts tree on the cpu

struct MctsNode
{
    explicit MctsNode(MctsNode *t_parent, const Board &t_board, const Colour t_colour)
        : turn_colour(t_colour)
        , parent(t_parent)
        , current_score(0)
        , number_of_visits(0)
    {
        current_board_state = t_board;
        possible_moves = generateAllPossibleMoves(t_board, t_colour);
    }

    explicit MctsNode(MctsNode *t_parent, const Board &t_board, const LightMovePath &t_movePath, const Colour t_colour)
        : move_that_led_to_this_position(t_movePath)
        , turn_colour(t_colour)
        , parent(t_parent)
        , current_score(0)
        , number_of_visits(0)
    {
        current_board_state = t_board;
        possible_moves = generateAllPossibleMoves(t_board, t_colour);
    }

    ~MctsNode() = default;

    // -------------------
    // Game related fields
    // -------------------

    // Move applied in this node
    Board current_board_state;
    LightMovePath move_that_led_to_this_position;
    std::vector<Move> possible_moves;

    // from which player perspective
    // moves from this node's position are played
    Colour turn_colour;
    NodeStatus status = NodeStatus::SEARCHING;

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

    [[nodiscard]] double parent_visits() const { return parent->number_of_visits; }
    [[nodiscard]] double ucb_score() const
    {
        if (number_of_visits == 0) {
            return std::numeric_limits<double>::infinity();
        }
        return current_score / number_of_visits + C * std::sqrt(std::log(parent_visits()) / number_of_visits);
    }
    [[nodiscard]] bool is_fully_expanded() const { return possible_moves.empty(); }
    [[nodiscard]] size_t possible_count() const { return possible_moves.size(); }
    [[nodiscard]] bool is_terminal() const { return possible_moves.empty() && children.empty(); }
};


struct MctsTree
{
    // ---------------------------------
    // Creation of empty tree,
    // used at the beginning of the game
    // ---------------------------------
    MctsTree() = default;
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

    std::unique_ptr<MctsNode> root         = nullptr;
    Colour                    colour_of_ai = none;

    ~MctsTree() = default;

    void initTree(const Board &t_board, const Colour t_colour)
    {
        root = std::make_unique<MctsNode>(nullptr, t_board, t_colour);
        colour_of_ai = t_colour;
    }
    void updateRoot(const MctsNode *t_new_root);
};

[[nodiscard]] MctsNode *findPlayerMove(const MctsNode *t_root, const Board &t_board, LightMovePath t_move);
[[nodiscard]] MctsNode *selectNode(MctsNode *t_node);
[[nodiscard]] MctsNode *chooseBestMove(const MctsTree& t_tree);
[[nodiscard]] int       rollout();
[[nodiscard]] MctsNode *expandNode(MctsNode *t_node);
void                    backpropagate(MctsNode *t_leaf, double t_score, Colour t_aiColour);


#endif // MCTS_CHECKERS_MCTS_H
