//
// Created by chefxxx on 20.10.2025.
//

#ifndef CUDA_TEMPLATE_MEMORY_CUDA
#define CUDA_TEMPLATE_MEMORY_CUDA

#include <atomic>
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <type_traits>

#include "helper_cuda.h"
#include "logger_util.h"

namespace cuda {
template <class T> concept cuda_pointerable_type = std::is_array_v<T> == !std::is_array_v<T> && !std::is_pointer_v<T>;

template <cuda_pointerable_type U> struct cuda_deleter
{
    void operator()(U *d_ptr) const noexcept { checkCudaErrors(cudaFree(d_ptr)); }
};

template <cuda_pointerable_type T, class D = cuda_deleter<T>> class unique_ptr
{
public:
    using pointer      = T *;
    using element_type = T;
    using deleter_type = D;

    // --------------
    // Class creation
    // --------------

    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr unique_ptr(std::nullptr_t) noexcept
        : mDevPtr(nullptr)
        , mDeleter{}
    {
    }

    constexpr unique_ptr() noexcept
        : mDevPtr(nullptr)
        , mDeleter{}
    {
    }

    /**
     * Use with caution - passing host pointer or pointer
     * not yet allocated on GPU causes undefined behaviour.
     *
     * Note: It is encouraged to use factory functions cuda::make_unique<>().
     *
     * @params devPtr already allocated (with CudaMalloc or cudaMallocPitch) pointer do device memory.
     */
    explicit unique_ptr(const pointer devPtr) noexcept
        : mDevPtr(devPtr)
        , mDeleter{}
    {
    }

    unique_ptr(unique_ptr &&other) noexcept
        : mDevPtr(other.release())
        , mDeleter(std::move(other.get_deleter()))
    {
    }

    // ----------
    // Destructor
    // ----------

    constexpr ~unique_ptr() noexcept
    {
        if (mDevPtr) {
            logger::info("Destroying mem_cuda::unique_ptr and releasing memory...");
            get_deleter()(get());
        }
    }

    // -------------------
    // Disable lvalue copy
    // -------------------

    unique_ptr &operator=(const unique_ptr &) = delete;
    unique_ptr(const unique_ptr &)            = delete;

    // ----------
    // Assignment
    // ----------

    unique_ptr &operator=(unique_ptr &&r) noexcept
    {
        reset(r.release());
        mDeleter = std::move(r.get_deleter());
        return *this;
    }

    // ---------
    // Observers
    // ---------

    pointer get() const noexcept { return mDevPtr; }

    const deleter_type &get_deleter() const noexcept { return mDeleter; }

    deleter_type &get_deleter() noexcept { return mDeleter; }

    explicit operator bool() const noexcept { return get() != nullptr; }

    // ---------
    // Modifiers
    // ---------

    /**
     * Implementation assumes that passed pointer is already allocated on GPU.
     * Otherwise, behaviour is undefined.
     *
     * @param nDevPtr pointer to already allocated (using cudaMalloc or cudaMallocPitch) memory on gpu.
     */
    void reset(pointer nDevPtr = pointer()) noexcept
    {
        auto oldDevPtr = get();
        mDevPtr        = nDevPtr;
        if (oldDevPtr)
            get_deleter()(oldDevPtr);
    }

    pointer release() noexcept
    {
        const pointer tmp = get();
        mDevPtr           = nullptr;
        return tmp;
    }

    // -------------------
    // Operators ->, *, []
    // -------------------

    /**
     * Those operators are deleted to forbid using device memory on host.
     */

    pointer                                 operator->() const noexcept      = delete;
    std::add_lvalue_reference<element_type> operator*() const noexcept       = delete;
    std::add_lvalue_reference<element_type> operator[](int i) const noexcept = delete;

private:
    pointer      mDevPtr;
    deleter_type mDeleter;
};

/**
 * Constructs @param count objects of type T and allocates them on GPU
 * using cudaMalloc() function. Then wraps them into cuda::unique_ptr.
 *
 * @param count number of objects to allocate on GPU.
 * @return cuda::unique_ptr that owns pointer to those object(s).
 */
template <cuda_pointerable_type T> unique_ptr<T> make_unique(const size_t count = 1)
{
    T *devPtr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devPtr), sizeof(T) * count));
    return unique_ptr<T>(devPtr);
}

template <cuda_pointerable_type T, class D = cuda_deleter<T>> struct control_block
{
    explicit control_block(T *devPtr)
        : mDeleter{}
        , mDevPtr(devPtr)
        , mRefCount{1}
        , mWeakRefCount{1}
    {
    }
    D                 mDeleter;
    T                *mDevPtr;
    std::atomic<long> mRefCount;
    std::atomic<long> mWeakRefCount;

    void add_ref() noexcept { mRefCount.fetch_add(1, std::memory_order_relaxed); }

    void release_ref() noexcept
    {
        if (mRefCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            mDeleter(mDevPtr);
            logger::info("Releasing mem_cuda::shared_ptr memory...");
            release_weak_ref();
        }
    }

    void add_weak_ref() noexcept { mWeakRefCount.fetch_add(1, std::memory_order_relaxed); }

    void release_weak_ref() noexcept
    {
        if (mWeakRefCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            std::atomic_thread_fence(std::memory_order_acquire);
            logger::info("Deleting control block...");
            delete this;
        }
    }
};

template <cuda_pointerable_type T> class shared_ptr
{
public:
    using element_type = std::remove_extent_t<T>;
    using weak_type    = std::weak_ptr<T>;

    // --------------
    // Class creation
    // --------------

    // NOLINTNEXTLINE(google-explicit-constructor)
    constexpr shared_ptr(std::nullptr_t) noexcept
        : mControlBlock(nullptr)
        , mDevPtr(nullptr)
    {
    }

    constexpr shared_ptr()
        : mControlBlock(nullptr)
        , mDevPtr(nullptr)
    {
    }

    /**
     * Use with caution - passing host pointer or pointer
     * not yet allocated on GPU causes undefined behaviour.
     *
     * Note: It is encouraged to use factory functions cuda::make_shared<>().
     *
     * @params devPtr already allocated (with CudaMalloc or cudaMallocPitch) pointer do device memory.
     */
    explicit constexpr shared_ptr(element_type *devPtr) noexcept
        : mDevPtr(devPtr)
    {
        mControlBlock = new control_block<element_type>(devPtr);
    }

    shared_ptr(const shared_ptr &other) noexcept
    {
        this->mControlBlock = other.mControlBlock;
        this->mDevPtr       = other.mDevPtr;
        if (mControlBlock)
            mControlBlock->add_ref();
    }

    shared_ptr(shared_ptr &&other) noexcept
    {
        this->mControlBlock = other.mControlBlock;
        this->mDevPtr       = other.mDevPtr;
        other.mControlBlock = other.mDevPtr = nullptr;
    }

    // ----------
    // Destructor
    // ----------

    ~shared_ptr()
    {
        logger::info("Destroying cuda::shared_ptr...");
        _cleanup();
    }

    // ----------
    // Assignment
    // ----------

    shared_ptr &operator=(const shared_ptr &r) noexcept
    {
        _cleanup();
        this->mControlBlock = r.mControlBlock;
        this->mDevPtr       = r.mDevPtr;
        if (mControlBlock)
            mControlBlock->add_ref();
        return *this;
    }

    shared_ptr &operator=(shared_ptr &&r) noexcept
    {
        _cleanup();
        this->mControlBlock = r.mControlBlock;
        this->mDevPtr       = r.mDevPtr;
        r.mControlBlock = r.mDevPtr = nullptr;
        return *this;
    }

    // ---------
    // Modifiers
    // ---------

    void reset() noexcept { _cleanup(); }

    /**
     * Implementation assumes that passed pointer is already allocated on gpu.
     * Otherwise, behaviour is undefined.
     *
     * @param nDevPtr pointer to already allocated (using cudaMalloc or cudaMallocPitch) memory on gpu.
     */
    void reset(element_type *nDevPtr)
    {
        _cleanup();
        if (nDevPtr) {
            mControlBlock = new control_block<element_type>(nDevPtr);
            mDevPtr       = nDevPtr;
        }
    }

    // ---------
    // Observers
    // ---------

    element_type *get() const noexcept { return mDevPtr; }

    [[nodiscard]] long use_count() const noexcept
    {
        return mControlBlock ? mControlBlock->mRefCount.load(std::memory_order_relaxed) : 0;
    }

    explicit operator bool() const noexcept { return get() != nullptr; }

    // -------------------
    // Operators ->, *, []
    // -------------------

    /**
     * Those operators are deleted to forbid using device memory on host.
     */

    element_type                           *operator->() const noexcept      = delete;
    std::add_lvalue_reference<element_type> operator*() const noexcept       = delete;
    std::add_lvalue_reference<element_type> operator[](int i) const noexcept = delete;

private:
    control_block<element_type> *mControlBlock;
    element_type                *mDevPtr;

    void _cleanup() noexcept
    {
        if (mControlBlock) {
            mControlBlock->release_ref();
            mControlBlock = nullptr;
            mDevPtr       = nullptr;
        }
    }
};

/**
 * Constructs @param count objects of type T and allocates them on GPU
 * using cudaMalloc() function. Then wraps them into cuda::shared_ptr.
 *
 * @param count number of objects to allocate on GPU.
 * @return cuda::shared_ptr that owns pointer to those object(s).
 */
template <cuda_pointerable_type T> shared_ptr<T> make_shared(const size_t count = 1)
{
    T *devPtr;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&devPtr), sizeof(T) * count));
    return shared_ptr<T>(devPtr);
}
} // namespace cuda

#endif