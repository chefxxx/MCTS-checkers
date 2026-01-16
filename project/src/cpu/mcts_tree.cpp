//
// Created by chefxx on 10.01.2026.
//

#include "mcts_tree.h"

#include "checkers_engine.h"
#include "cpu_movegen.h"

MctsNode *MctsTree::selectNode()
{
    throw "Not implemented!";
}

void MctsTree::expandNode(MctsNode *t_node)
{
    for (const auto  possible_moves = generateAllPossibleMoves(t_node->board, t_node->colour_of_player_to_move);
         const auto &mv : possible_moves) {
        const Board b = applyMove(t_node->board, mv, t_node->colour_of_player_to_move);
        const auto lp = LightMovePath(mv.positions, mv.captures_mask > 0);
        auto new_child = std::make_unique<MctsNode>(t_node, b, lp, static_cast<Colour>(1 - t_node->colour_of_player_to_move));
        t_node->children.push_back(std::move(new_child));
    }
}
