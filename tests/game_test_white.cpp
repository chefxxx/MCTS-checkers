//
// Created by chefxx on 18.01.2026.
//

#include <gtest/gtest.h>

#include "game_engine.h"

TEST(GameTest, whiteTest) {
    GameManager gm{black, 1, "debug"};
    gm.playTheGame();
}