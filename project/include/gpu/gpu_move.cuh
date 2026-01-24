//
// Created by chefxx on 23.01.2026.
//

#ifndef GPU_MOVE_CUH
#define GPU_MOVE_CUH

struct GPU_Move
{
    __host__ __device__ GPU_Move() = default;
    __device__ GPU_Move(const size_t t_from, const size_t t_to, const size_t t_captures)
        : from_mask(t_from)
        , to_mask(t_to)
        , captures_mask(t_captures)
    {
    }
    size_t from_mask;
    size_t to_mask;
    size_t captures_mask;
};

#endif
