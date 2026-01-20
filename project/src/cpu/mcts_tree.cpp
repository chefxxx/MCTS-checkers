//
// Created by chefxx on 10.01.2026.
//

#include "mcts_tree.h"

#include <cassert>
#include <random>

#include "checkers_engine.h"
#include "cpu_movegen.h"


void MctsTree::updateRoot(const MctsNode *t_new_root)
{
    assert(t_new_root != root.get());
    assert(t_new_root != nullptr);
    assert(t_new_root->parent != nullptr);
    MctsNode* parent = t_new_root->parent;

    const auto it = std::ranges::find_if(
        parent->children,
        [t_new_root](const std::unique_ptr<MctsNode>& ptr) {
            return ptr.get() == t_new_root;
        });

    assert(it != parent->children.end());
    MctsNode *new_raw = it->release();
    assert(new_raw == t_new_root);
    parent->children.erase(it);
    root.reset(new_raw);
    root->parent = nullptr;
}

MctsNode *findPlayerMove(const MctsNode *t_root, const Board &t_board, const LightMovePath t_move)
{
    if (!t_root) return nullptr;
    for (const auto& child : t_root->children) {
        if (child->current_board_state == t_board && child->move_that_led_to_this_position == t_move) {
            return child.get();
        }
    }
    return nullptr;
}

MctsNode *selectNode(MctsNode *t_node)
{
    assert(t_node != nullptr);

    if (t_node->is_terminal() || !t_node->is_fully_expanded())
        return t_node;

    auto curr_max = std::numeric_limits<double>::min();
    MctsNode *best_node = t_node;
    for (const auto &child : t_node->children) {
        if (child->ucb_score() > curr_max) {
            curr_max = child->ucb_score();
            best_node = child.get();
        }
    }
    return selectNode(best_node);
}

// Note:
//
// Currently a robust child is chosen.
Board chooseBestMove(MctsTree& t_tree)
{
    const auto root = t_tree.root.get();
    if (root->children.empty())
        return root->current_board_state;

    const MctsNode *robust_child = root->children[0].get();
    int             max_n        = -1;
    for (auto &child : root->children) {
        if (child->number_of_visits > max_n) {
            max_n        = child->number_of_visits;
            robust_child = child.get();
        }
    }
    // update the tree state
    t_tree.updateRoot(robust_child);
    return robust_child->current_board_state;
}

int rollout()
{
    // TODO: now i use dummy func
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, 1);
    return distrib(gen);
}

int randomChild(const int t_size)
{
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, t_size - 1);
    return distrib(gen);
}

MctsNode* expandNode(MctsNode *t_node)
{
    assert(!t_node->is_fully_expanded());

    // access random child and erase it
    const auto idx = static_cast<size_t>(randomChild(t_node->possible_count()));
    const auto mv = t_node->possible_moves[idx];
    std::swap(t_node->possible_moves[idx], t_node->possible_moves.back());
    t_node->possible_moves.pop_back();

    // create board
    const auto next_board_state = applyMove(t_node->current_board_state, mv, t_node->turn_colour);
    assert(next_board_state.has_value());

    // create path encoding
    const auto lp = mv.path;

    // add new child
    auto new_child = std::make_unique<MctsNode>(t_node, next_board_state.value(), lp, static_cast<Colour>(1 - t_node->turn_colour));
    t_node->children.push_back(std::move(new_child));
    return new_child.get();
}

void backpropagate(MctsNode *t_leaf, const double t_score, const Colour t_aiColour)
{
    MctsNode *tmp = t_leaf;
    while (tmp != nullptr) {
        // If the turn_colour in the current node is the same
        // as the aiColour it means that this position
        // is created from opponent's move.
        // E.g. if a node is black and ai_col is white it means that ai
        // move led to this position - meaning that if we have a win form
        // this position we score it as 1.0
        const double node_score = tmp->turn_colour == t_aiColour ? 1.0 - t_score : t_score;
        tmp->current_score += node_score;
        tmp->number_of_visits++;
        tmp = tmp->parent;
    }
}
