//
// Created by chefxx on 10.01.2026.
//

#include "mcts_tree.h"

#include <cassert>
#include <random>

#include "checkers_engine.h"


void MctsTree::updateRoot(const MctsNode *t_new_root)
{
    assert(t_new_root != root.get());
    assert(t_new_root != nullptr);
    assert(t_new_root->parent != nullptr);
    MctsNode *parent = t_new_root->parent;

    const auto it = std::ranges::find_if(
        parent->children, [t_new_root](const std::unique_ptr<MctsNode> &ptr) { return ptr.get() == t_new_root; });

    assert(it != parent->children.end());
    MctsNode *new_raw = it->release();
    assert(new_raw == t_new_root);
    parent->children.erase(it);
    root.reset(new_raw);
    root->parent = nullptr;
}

void MctsTree::updateTree() const
{
    while (!root->is_fully_expanded()) {
        const auto selectedNode = expandNode(root.get());
        double score;
        if (selectedNode->is_solved()) {
            if (selectedNode->status == NodeStatus::WIN)
                score = 0.0;
            else if (selectedNode->status == NodeStatus::LOSS)
                score = 1.0;
            else
                score = 0.5;
        }
        else {
            score = rollout(selectedNode);
        }
        backpropagate(selectedNode, score);
    }
}

MctsNode *findPlayerMove(const MctsNode *t_root, const Board &t_board, const LightMovePath t_move)
{
    assert(t_root != nullptr);
    for (const auto &child : t_root->children) {
        if (child->current_board_state == t_board && child->packed_positions_transition == t_move) {
            return child.get();
        }
    }
    return nullptr;
}

MctsNode *selectNode(MctsNode *t_node)
{
    assert(t_node != nullptr);

    if (t_node->is_solved() || !t_node->is_fully_expanded())
        return t_node;

    auto      curr_max  = std::numeric_limits<double>::min();
    MctsNode *best_node = nullptr;
    for (const auto &child : t_node->children) {
        // if child is a win, avoid it
        if (child->status == NodeStatus::WIN)
            continue;

        if (child->ucb_score() > curr_max) {
            curr_max  = child->ucb_score();
            best_node = child.get();
        }
    }

    // if all children are forced losses
    if (best_node == nullptr) {
        curr_max = -std::numeric_limits<double>::infinity();
        for (const auto &child : t_node->children) {
            if (child->ucb_score() > curr_max) {
                curr_max  = child->ucb_score();
                best_node = child.get();
            }
        }
    }

    assert(best_node != nullptr);
    return selectNode(best_node);
}

// Note:
//
// Currently a robust child is chosen.
MctsNode *chooseBestMove(const MctsTree &t_tree)
{
    const auto root = t_tree.root.get();
    assert(!root->children.empty());
    if (root->children.empty())
        return root;

    // If found a winning path, choose it
    for (const auto &child : root->children) {
        if (child->status == NodeStatus::LOSS)
            return child.get();
    }

    // if a win cannot be found look for a draw
    if (root->status == NodeStatus::DRAW) {
        for (const auto &child : root->children) {
            if (child->status == NodeStatus::DRAW)
                return child.get();
        }
    }

    MctsNode *robust_child = root->children[0].get();
    double    max_n        = -1;
    for (auto const &child : root->children) {
        if (child->number_of_visits > max_n) {
            max_n        = child->number_of_visits;
            robust_child = child.get();
        }
    }

    assert(robust_child != nullptr);
    return robust_child;
}

int randomIdx(const int t_size)
{
    std::random_device            rd;
    std::mt19937                  gen(rd());
    std::uniform_int_distribution distrib(0, t_size - 1);
    return distrib(gen);
}

void updateNodeStatus(MctsNode *t_node)
{
    if (t_node->is_solved())
        return;

    // Here we know that the node is fully expanded
    // and that it is not solved.

    // If one of the children is a loss - this is a win
    auto it = std::ranges::find_if(
        t_node->children, [](const std::unique_ptr<MctsNode> &child) { return child->status == NodeStatus::LOSS; });
    if (it != t_node->children.end()) {
        t_node->status = NodeStatus::WIN;
        return;
    }

    // To prove a win we only need to find one winning path
    if (!t_node->is_fully_expanded())
        return;

    // If we are at this point we know that there is no child marked as a LOSS
    // only options are WIN, DRAW, SEARCHING.

    // Check if all children are solved nodes
    const bool all_solved = std::ranges::all_of(t_node->children, [](const auto &child) { return child->is_solved(); });
    if (!all_solved)
        return;

    // Here we know that the children nodes are only DRAW or WIN,
    // if we find any DRAW node, this node is forced draw
    it = std::ranges::find_if(t_node->children,
                              [](const std::unique_ptr<MctsNode> &child) { return child->status == NodeStatus::DRAW; });

    if (it != t_node->children.end()) {
        t_node->status = NodeStatus::DRAW;
    }
    else {
        // Every single child is a WIN for the opponent.
        t_node->status = NodeStatus::LOSS;
    }
}

void MctsNode::checkNodeStatus()
{
    if (current_board_state.kings_quiet_moves == DRAW_LIMIT) {
        status = NodeStatus::DRAW;
        return;
    }
    if (!(current_board_state.pawns[turn_colour] | current_board_state.kings[turn_colour])) {
        status = NodeStatus::LOSS;
        return;
    }
    if (possible_moves.empty()) {
        status = NodeStatus::LOSS;
        return;
    }
    status = NodeStatus::SEARCHING;
}

double rollout(const MctsNode *t_node)
{
    // Note:
    //
    // This rollout function returns the result
    // in the perspective of the node that is the
    // parent of the t_node.
    //
    auto       tmp_board     = t_node->current_board_state;
    Colour     turn          = t_node->turn_colour;
    const auto parent_colour = static_cast<Colour>(1 - turn);
    GameState  nodeResult;
    assert(!t_node->possible_moves.empty());

    while (true) {
        const auto  moves  = generateAllPossibleMoves(tmp_board, turn);
        const auto &rand_mv = moves[randomIdx(static_cast<int>(moves.size()))];
        const auto  opt  = applyMove(tmp_board, rand_mv, turn);
        assert(opt.has_value());
        tmp_board  = opt.value();
        nodeResult = checkEndOfGameConditions(tmp_board, turn);
        if (nodeResult != CONTINUES) {
            break;
        }
        turn = static_cast<Colour>(1 - turn);
    }
    if (nodeResult == DRAW) {
        return 0.5;
    }
    return turn == parent_colour ? 1.0 : 0.0;
}

MctsNode *expandNode(MctsNode *t_node)
{
    assert(!t_node->is_fully_expanded());

    // access random child and erase it
    const auto idx = static_cast<size_t>(randomIdx(static_cast<int>(t_node->possible_moves.size())));
    const auto mv  = t_node->possible_moves[idx];
    std::swap(t_node->possible_moves[idx], t_node->possible_moves.back());
    t_node->possible_moves.pop_back();

    // create board
    const auto next_board_state = applyMove(t_node->current_board_state, mv, t_node->turn_colour);
    assert(next_board_state.has_value());

    // create path encoding
    const auto lp = mv.packed_positions;

    // add new child
    auto new_child =
        std::make_unique<MctsNode>(t_node, next_board_state.value(), lp, static_cast<Colour>(1 - t_node->turn_colour));
    const auto retPtr = new_child.get();
    t_node->children.push_back(std::move(new_child));
    assert(retPtr != nullptr);
    return retPtr;
}

void backpropagate(MctsNode *t_leaf, const double t_score)
{
    MctsNode *tmp           = t_leaf;
    double    running_score = t_score;
    while (tmp != nullptr) {
        tmp->current_score += running_score;
        running_score = 1.0 - running_score;
        updateNodeStatus(tmp);
        tmp->number_of_visits++;
        tmp = tmp->parent;
    }
}
