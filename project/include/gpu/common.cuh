//
// Created by chefxx on 22.01.2026.
//

#ifndef GPU_COMMON_CUH
#define GPU_COMMON_CUH

#include <cuda_runtime_api.h>

#include "constants.h"

// Pawns move and attack mask creation functions

__device__ __host__ __forceinline__ size_t getPawnsAttackMask(const size_t t_attackerPawns,
                                                              const size_t t_opponentPieces,
                                                              const size_t t_emptyFiles)
{
    const size_t attacks_upRight =
        (((t_attackerPawns & NOT_FILE_G & NOT_FILE_H) << 9 & t_opponentPieces) << 9 & t_emptyFiles) >> 18;
    const size_t attacks_downRight =
        (((t_attackerPawns & NOT_FILE_G & NOT_FILE_H) >> 7 & t_opponentPieces) >> 7 & t_emptyFiles) << 14;
    const size_t attacks_upLeft =
        (((t_attackerPawns & NOT_FILE_A & NOT_FILE_B) << 7 & t_opponentPieces) << 7 & t_emptyFiles) >> 14;
    const size_t attacks_downLeft =
        (((t_attackerPawns & NOT_FILE_A & NOT_FILE_B) >> 9 & t_opponentPieces) >> 9 & t_emptyFiles) << 18;
    return attacks_upRight | attacks_downRight | attacks_upLeft | attacks_downLeft;
}


#endif
