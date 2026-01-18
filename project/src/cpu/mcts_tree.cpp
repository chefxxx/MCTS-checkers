//
// Created by chefxx on 10.01.2026.
//

#include "mcts_tree.h"

#include <cassert>
#include <random>

#include "checkers_engine.h"
#include "cpu_movegen.h"

MctsNode *selectNode(MctsNode *t_root)
{
    if (t_root != nullptr && t_root->children.empty()) {
        return t_root;
    }

    double    max       = std::numeric_limits<double>::min();
    MctsNode *best_node = nullptr;
    for (const auto &child : t_root->children) {
        if (const auto curr = child->calculate_UCB(); curr > max) {
            max       = curr;
            best_node = child.get();
        }
    }

    return selectNode(best_node);
}

// Note:
//
// Currently a robust child is chosen.
Board chooseBestMove(const MctsNode *t_root)
{
    if (t_root->children.empty()) return t_root->board;

    const MctsNode *robust_child = t_root->children[0].get();
    int             max_n        = -1;
    for (auto &child : t_root->children) {
        if (child->number_of_visits > max_n) {
            max_n        = child->number_of_visits;
            robust_child = child.get();
        }
    }
    return robust_child->board;
}

int rollout()
{
    // TODO: now i use dummy func
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen);
}

// TODO: somewhere here you have to check game state conditions etc.
void expandNode(MctsNode *t_node)
{
    for (const auto  possible_moves = generateAllPossibleMoves(t_node->board, t_node->colour_of_player_to_move);
         const auto &mv : possible_moves) {
        const auto b = applyMove(t_node->board, mv, t_node->colour_of_player_to_move);
        assert(b.has_value());
        const auto lp        = LightMovePath(mv.positions, mv.captures_mask > 0);
        auto       new_child = std::make_unique<MctsNode>(
            t_node, b.value(), lp, static_cast<Colour>(1 - t_node->colour_of_player_to_move));
        t_node->children.push_back(std::move(new_child));
    }
}

void backpropagate(MctsNode *t_leaf, const double t_score, const Colour t_aiColour)
{
    MctsNode *tmp = t_leaf;
    while (tmp != nullptr) {
        const double node_score = tmp->colour_of_player_to_move == t_aiColour ? t_score : 1.0 - t_score;
        tmp->current_score+= node_score;
        tmp->number_of_visits++;
        tmp = tmp->parent;
    }
}
