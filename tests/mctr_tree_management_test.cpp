//
// Created by chefxx on 16.01.2026.
//

#include <gtest/gtest.h>

#include "checkers_engine.h"
#include "game_engine.h"
#include "mcts_tree.h"

TEST(MctsTreeBasicTests, creationTest)
{
    Board board;
    board.initStartingBoard();
    const MctsTree mcts_tree{board, white};
    ASSERT_FALSE(mcts_tree.root == nullptr);
}

TEST(MctsTreeBasicTests, expandNodeTests)
{
    Board board;
    board.initStartingBoard();
    const MctsTree mcts_tree{board, white};

    while (!mcts_tree.root->is_fully_expanded()) {
        [[maybe_unused]] const auto node = expandNode(mcts_tree.root.get());
    }
    const auto mv0 = processMoveString("a3-b4", board, white);
    const auto mv1 = processMoveString("c3-b4", board, white);
    const auto mv2 = processMoveString("c3-d4", board, white);
    const auto mv3 = processMoveString("e3-d4", board, white);
    const auto mv4 = processMoveString("e3-f4", board, white);
    const auto mv5 = processMoveString("g3-f4", board, white);
    const auto mv6 = processMoveString("g3-h4", board, white);

    for (const std::vector moves = {mv0, mv1, mv2, mv3, mv4, mv5, mv6}; const auto &move : moves) {
        const auto newBoard  = applyMove(board, move.value(), white);
        ASSERT_TRUE(newBoard.has_value());
        const auto foundNode = findPlayerMove(mcts_tree.root.get(), newBoard.value(), move->path);
        ASSERT_TRUE(foundNode != nullptr);
    }
}

TEST(MctsTreeBasicTests, aiStartsAsBlackInitialization)
{
    Board board;
    board.initStartingBoard();
    const auto first_move = processMoveString("c3-d4", board, white);
    board = applyMove(board, first_move.value(), white).value();
    MctsTree mcts_tree;
    mcts_tree.initTree(board, black);

    while (!mcts_tree.root->is_fully_expanded()) {
        [[maybe_unused]] const auto node = expandNode(mcts_tree.root.get());
    }

    const auto mv0 = processMoveString("b6-a5", board, black);
    const auto mv1 = processMoveString("b6-c5", board, black);
    const auto mv2 = processMoveString("d6-c5", board, black);
    const auto mv3 = processMoveString("d6-e5", board, black);
    const auto mv4 = processMoveString("f6-e5", board, black);
    const auto mv5 = processMoveString("f6-g5", board, black);
    const auto mv6 = processMoveString("h6-g5", board, black);

    int i = 0;
    for (const std::vector moves = {mv0, mv1, mv2, mv3, mv4, mv5, mv6}; const auto &move : moves) {
        const auto newBoard  = applyMove(board, move.value(), black);
        ASSERT_TRUE(newBoard.has_value());
        const auto foundNode = findPlayerMove(mcts_tree.root.get(), newBoard.value(), move->path);
        ASSERT_TRUE(foundNode != nullptr) << "At iteration" << i << '\n';
        i++;
    }
}

TEST(MctsTreeBasicTests, updateRootTestWhiteStarts)
{
    Board board;
    board.initStartingBoard();
    MctsTree mcts_tree{board, white};

    while (!mcts_tree.root->is_fully_expanded()) {
        [[maybe_unused]] const auto node = expandNode(mcts_tree.root.get());
    }
    const auto move = processMoveString("e3-d4", board, white);
    const auto newBoard  = applyMove(board, move.value(), white).value();
    const auto node      = findPlayerMove(mcts_tree.root.get(), newBoard, move->path);
    ASSERT_TRUE(node != nullptr);
    mcts_tree.updateRoot(node);
    ASSERT_TRUE(mcts_tree.root.get() != nullptr);
    ASSERT_TRUE(mcts_tree.root.get() == node);
}

TEST(MctsTreeBasicTests, gameSimulationTest)
{
    GameManager manager{white, 1, "cpu"};
    manager.mcts_tree.initTree(manager.board, white);

    const std::vector<std::string> whiteMoves = {
     "c3-d4",
       "a3-b4",
        "g3-h4",
        "e3:g5",
        "b2-a3",
        "h4:f6",
        "a1-b2",
        "b2-c3",
        "c1-b2",
        "f2-g3",
        "e1:g3:e5:g7",
        "c3-b4",
        "b4:d6",
        "d2-c3",
        "g1-f2",
        "c3:e5:g3",
        "g3-h4",
        "h4-g5",
        "g5-h6",
        "h6-g7",
        "g7-h8",
        "h8-c3",
        "c3:a5:d8"
    };

    const std::vector<std::string> blackMoves = {
        "b6-a5",
        "a5:c3:e5",
        "e5-f4",
        "h6:f4",
        "f6-g5",
        "e7:g5",
        "d6-c5",
        "g5-h4",
        "g7-f6",
        "h4:f2",
        "h8:f6",
        "f6-g5",
        "c7:e5",
        "g5-f4",
        "e5-d4",
        "f8-e7",
        "d8-c7",
        "e7-d6",
        "a7-b6",
        "b6-a5",
        "b8-a7",
        "a5-b4"
    };

    const std::array plays = {blackMoves, whiteMoves};
    std::array counters = {0, 0};
    const size_t each_moves = whiteMoves.size() + blackMoves.size();
    Colour turn = white;
    for (size_t i = 0; i < each_moves; ++i) {
        manager.printBoard();
        // expand the tree
        while (!manager.mcts_tree.root->is_fully_expanded()) {
            [[maybe_unused]] const auto node = expandNode(manager.mcts_tree.root.get());
        }
        const auto idx = counters[turn];
        const std::string message = "At iteration " + std::to_string(i) + ", turn " + std::to_string(turn);
        counters[turn]++;
        const auto moveStr  = plays[turn][idx];
        std::cout << std::to_string(turn) << " plays " << moveStr <<'\n';
        const auto move = processMoveString(moveStr, manager.board, turn);
        manager.board             = applyMove(manager.board, move.value(), turn).value();
        const auto node           = findPlayerMove(manager.mcts_tree.root.get(), manager.board, move->path);
        ASSERT_TRUE(node != nullptr) << message;
        manager.mcts_tree.updateRoot(node);
        ASSERT_TRUE(manager.mcts_tree.root.get() != nullptr);
        ASSERT_TRUE(manager.mcts_tree.root.get() == node);
        turn = static_cast<Colour>(1- turn);
    }
    manager.printBoard();
}