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
        if (child->board == t_board && child->move_packed_path == t_move) {
            return child.get();
        }
    }
    return nullptr;
}

MctsNode *selectNode(MctsNode *t_node)
{
    assert(t_node != nullptr);

    double    max       = std::numeric_limits<double>::min();
    MctsNode *best_node = nullptr;
    for (const auto &child : t_node->children) {
        if (const auto curr = child->calculate_UCB(); curr > max) {
            max       = curr;
            best_node = child.get();
        }
    }

    return best_node ? selectNode(best_node) : t_node;
}

// Note:
//
// Currently a robust child is chosen.
Board chooseBestMove(MctsTree& t_tree)
{
    const auto root = t_tree.root.get();
    if (root->children.empty())
        return root->board;

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
        tmp->current_score += node_score;
        tmp->number_of_visits++;
        tmp = tmp->parent;
    }
}
