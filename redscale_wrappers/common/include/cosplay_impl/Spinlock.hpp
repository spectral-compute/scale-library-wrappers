#pragma once

#include <atomic>

namespace CudaRocmWrapper
{

/**
 * A simple spinlock.
 *
 * This spinlock is not reentrant. It meets the requirements of C++ BasicLockable.
 */
class Spinlock final
{
public:
    void lock() noexcept
    {
        while (flag.test_and_set(std::memory_order_acquire)) {}
    }

    void unlock() noexcept
    {
        flag.clear(std::memory_order_release);
    }

private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
};

} // namespace CudaRocmWrapper
