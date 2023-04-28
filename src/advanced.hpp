#pragma once
#include <thread>
#include <vector>

#include <fp16/fp16.h>

namespace unum {
namespace usearch {

class f16_converted_t {
    uint16_t uint16_{};

  public:
    inline f16_converted_t() noexcept : uint16_(0) {}
    inline f16_converted_t(f16_converted_t&&) = default;
    inline f16_converted_t& operator=(f16_converted_t&&) = default;
    inline f16_converted_t(f16_converted_t const&) = default;
    inline f16_converted_t& operator=(f16_converted_t const&) = default;

    inline f16_converted_t(float v) noexcept : uint16_(fp16_ieee_from_fp32_value(v)) {}
    inline operator float() const noexcept { return fp16_ieee_to_fp32_value(uint16_); }

    inline f16_converted_t& operator+=(float v) noexcept {
        uint16_ = fp16_ieee_from_fp32_value(v + fp16_ieee_to_fp32_value(uint16_));
        return *this;
    }

    inline f16_converted_t& operator-=(float v) noexcept {
        uint16_ = fp16_ieee_from_fp32_value(v - fp16_ieee_to_fp32_value(uint16_));
        return *this;
    }

    inline f16_converted_t& operator*=(float v) noexcept {
        uint16_ = fp16_ieee_from_fp32_value(v * fp16_ieee_to_fp32_value(uint16_));
        return *this;
    }

    inline f16_converted_t& operator/=(float v) noexcept {
        uint16_ = fp16_ieee_from_fp32_value(v / fp16_ieee_to_fp32_value(uint16_));
        return *this;
    }
};

template <typename callback_at> //
void multithreaded(std::size_t threads, std::size_t tasks, callback_at&& callback) {

    if (threads == 0)
        threads = std::thread::hardware_concurrency();
    if (threads == 1) {
        for (std::size_t task_idx = 0; task_idx < tasks; ++task_idx)
            callback(0, task_idx);
        return;
    }

    std::vector<std::thread> threads_pool;
    std::size_t tasks_per_thread = threads / tasks + (threads % tasks) != 0;
    for (std::size_t thread_idx = 0; thread_idx != threads; ++thread_idx) {
        threads_pool.emplace_back([=]() {
            for (std::size_t task_idx = thread_idx * tasks_per_thread;
                 task_idx < std::min(tasks, thread_idx * tasks_per_thread + tasks_per_thread); ++task_idx)
                callback(thread_idx, task_idx);
        });
    }

    for (std::size_t thread_idx = 0; thread_idx != threads; ++thread_idx)
        threads_pool[thread_idx].join();
}

} // namespace usearch
} // namespace unum