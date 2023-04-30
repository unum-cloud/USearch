#pragma once
#include <functional>
#include <thread>
#include <vector>

#if __linux__
#include <sys/auxv.h>
#endif

#include <fp16/fp16.h>
#include <simsimd/simsimd.h>

#include <usearch/usearch.hpp>

namespace unum {
namespace usearch {

using byte_t = char;
using punned_distance_t = float;
using punned_metric_t = punned_distance_t (*)(byte_t const*, byte_t const*, std::size_t, std::size_t);

template <typename metric_at>
static punned_distance_t punned_metric( //
    byte_t const* a, byte_t const* b, std::size_t a_bytes, std::size_t b_bytes) noexcept {
    using scalar_t = typename metric_at::scalar_t;
    return metric_at{}((scalar_t const*)a, (scalar_t const*)b, a_bytes / sizeof(scalar_t), b_bytes / sizeof(scalar_t));
}

using punned_stateful_metric_t =
    std::function<punned_distance_t(byte_t const*, byte_t const*, std::size_t, std::size_t)>;

template <typename at, typename compare_at> inline at clamp(at v, at lo, at hi, compare_at comp) noexcept {
    return comp(v, lo) ? lo : comp(hi, v) ? hi : v;
}
template <typename at> inline at clamp(at v, at lo, at hi) noexcept { return clamp(v, lo, hi, std::less<at>{}); }

class i8q100_converted_t;
class f16_converted_t;

class f16_converted_t {
    std::uint16_t uint16_{};

  public:
    inline f16_converted_t() noexcept : uint16_(0) {}
    inline f16_converted_t(f16_converted_t&&) = default;
    inline f16_converted_t& operator=(f16_converted_t&&) = default;
    inline f16_converted_t(f16_converted_t const&) = default;
    inline f16_converted_t& operator=(f16_converted_t const&) = default;

    inline operator float() const noexcept { return fp16_ieee_to_fp32_value(uint16_); }

    inline f16_converted_t(i8q100_converted_t) noexcept;
    inline f16_converted_t(float v) noexcept : uint16_(fp16_ieee_from_fp32_value(v)) {}
    inline f16_converted_t(double v) noexcept : uint16_(fp16_ieee_from_fp32_value(v)) {}

    inline f16_converted_t operator+(f16_converted_t other) const noexcept { return {float(*this) + float(other)}; }
    inline f16_converted_t operator-(f16_converted_t other) const noexcept { return {float(*this) - float(other)}; }
    inline f16_converted_t operator*(f16_converted_t other) const noexcept { return {float(*this) * float(other)}; }
    inline f16_converted_t operator/(f16_converted_t other) const noexcept { return {float(*this) / float(other)}; }

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

class i8q100_converted_t {
    std::int8_t int8_{};

  public:
    inline i8q100_converted_t() noexcept : int8_(0) {}
    inline i8q100_converted_t(i8q100_converted_t&&) = default;
    inline i8q100_converted_t& operator=(i8q100_converted_t&&) = default;
    inline i8q100_converted_t(i8q100_converted_t const&) = default;
    inline i8q100_converted_t& operator=(i8q100_converted_t const&) = default;

    inline operator float() const noexcept { return float(int8_) / 100.f; }
    inline operator f16_converted_t() const noexcept { return float(int8_) / 100.f; }
    inline operator double() const noexcept { return double(int8_) / 100.0; }

    inline i8q100_converted_t(float v) noexcept
        : int8_(clamp<std::int8_t>(static_cast<std::int8_t>(v * 100.f), -100, 100)) {}
    inline i8q100_converted_t(double v) noexcept
        : int8_(clamp<std::int8_t>(static_cast<std::int8_t>(v * 100.0), -100, 100)) {}
};

inline f16_converted_t::f16_converted_t(i8q100_converted_t v) noexcept : f16_converted_t(float(v)) {}

struct uuid_t {
    std::uint8_t octets[16];
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

/**
 *  @brief Relies on `posix_memalign` for aligned memory allocations.
 */
template <typename element_at = char, std::size_t alignment_ak = 64> //
class aligned_allocator_gt {
  public:
    using value_type = element_at;
    using size_type = std::size_t;
    using pointer = element_at*;
    using const_pointer = element_at const*;
    template <typename other_element_at> struct rebind {
        using other = aligned_allocator_gt<other_element_at>;
    };

    pointer allocate(size_type length) const noexcept {
        void* result = nullptr;
        int status = posix_memalign(&result, alignment_ak, ceil2(length * sizeof(value_type)));
        return status == 0 ? (pointer)result : nullptr;
    }

    void deallocate(pointer begin, size_type) const noexcept { std::free(begin); }
};

using aligned_allocator_t = aligned_allocator_gt<>;

enum class accuracy_t {
    f32_k,
    f16_k,
    f64_k,
    i8q100_k,
};

enum class isa_t {
    auto_k,
    neon_k,
    sve_k,
    avx2_k,
    avx512_k,
};

inline char const* isa(isa_t isa) noexcept {
    switch (isa) {
    case isa_t::auto_k: return "auto";
    case isa_t::neon_k: return "neon";
    case isa_t::sve_k: return "sve";
    case isa_t::avx2_k: return "avx2";
    case isa_t::avx512_k: return "avx512";
    }
}

inline std::size_t bytes_per_scalar(accuracy_t accuracy) noexcept {
    switch (accuracy) {
    case accuracy_t::f32_k: return 4;
    case accuracy_t::f16_k: return 2;
    case accuracy_t::f64_k: return 8;
    case accuracy_t::i8q100_k: return 1;
    }
}

inline bool supports_arm_sve() {
#if __linux__
    unsigned long hwcaps = getauxval(AT_HWCAP);
    if (hwcaps & HWCAP_SVE)
        return true;
#endif
    return false;
}

template <typename from_scalar_at, typename to_scalar_at> struct cast_gt {
    bool operator()(byte_t const* input, std::size_t bytes_in_input, byte_t* output) noexcept {
        from_scalar_at const* typed_input = reinterpret_cast<from_scalar_at const*>(input);
        to_scalar_at* typed_output = reinterpret_cast<to_scalar_at*>(output);
        std::transform( //
            typed_input, typed_input + bytes_in_input / sizeof(from_scalar_at), typed_output,
            [](from_scalar_at from) { return to_scalar_at(from); });
        return true;
    }
};

template <> struct cast_gt<f32_t, f32_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) noexcept { return false; }
};

template <> struct cast_gt<f64_t, f64_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) noexcept { return false; }
};

template <> struct cast_gt<f16_converted_t, f16_converted_t> {
    bool operator()(byte_t const*, std::size_t, byte_t*) noexcept { return false; }
};

/**
 *  @brief  Oversimplified type-punned index for equidimensional floating-point
 *          vectors with automatic down-casting and isa acceleration.
 */
template <typename label_at = std::int64_t, typename id_at = std::uint32_t> //
class auto_index_gt {
  public:
    using label_t = label_at;
    using id_t = id_at;
    using distance_t = punned_distance_t;

    using f32_t = float;
    using f64_t = double;

  private:
    /// @brief Schema: input buffer, bytes in input buffer, output buffer.
    using cast_t = std::function<bool(byte_t const*, std::size_t, byte_t*)>;
    /// @brief Punned index.
    using index_t = index_gt<punned_stateful_metric_t, label_t, id_t, byte_t>;
    /// @brief A type-punned metric and metadata about present isa support.
    struct metric_and_meta_t {
        punned_stateful_metric_t metric;
        isa_t acceleration;
    };

    std::size_t dimensions_ = 0;
    accuracy_t accuracy_ = accuracy_t::f32_k;
    std::unique_ptr<index_t> index_;

    isa_t acceleration_ = isa_t::auto_k;
    std::size_t casted_vector_bytes_ = 0;
    std::vector<byte_t> cast_buffer_;
    struct casts_t {
        cast_t from_i8q100{};
        cast_t from_f16{};
        cast_t from_f32{};
        cast_t from_f64{};
    } casts_;

    punned_stateful_metric_t root_metric_;
    config_t root_config_;

  public:
    std::size_t dimensions() const noexcept { return dimensions_; }
    std::size_t connectivity() const noexcept { return index_->connectivity(); }
    std::size_t size() const noexcept { return index_->size(); }
    std::size_t capacity() const noexcept { return index_->capacity(); }
    isa_t acceleration() const noexcept { return acceleration_; }

    void save(char const* path) { index_->save(path); }
    void load(char const* path) { index_->load(path); }
    void view(char const* path) { index_->view(path); }

    void add(label_t label, f32_t const* vector, std::size_t thread_idx = 0, bool copy = true) {
        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(f32_t);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * thread_idx;
        bool casted = casts_.from_f32(vector_data, casted_vector_bytes_, casted_data);
        if (casted) {
            vector_data = casted_data;
            vector_bytes = casted_vector_bytes_;
            copy = true;
        }

        index_->add(label, {vector_data, vector_bytes}, thread_idx, copy);
    }

    std::size_t search(                          //
        f32_t const* vector, std::size_t wanted, //
        label_t* matches, f32_t* distances, std::size_t thread_idx = 0) {

        byte_t const* vector_data = reinterpret_cast<byte_t const*>(vector);
        std::size_t vector_bytes = dimensions_ * sizeof(f32_t);

        byte_t* casted_data = cast_buffer_.data() + casted_vector_bytes_ * thread_idx;
        bool casted = casts_.from_f32(vector_data, casted_vector_bytes_, casted_data);
        if (casted) {
            vector_data = casted_data;
            vector_bytes = casted_vector_bytes_;
        }

        return index_->search({vector_data, vector_bytes}, wanted, matches, distances, thread_idx);
    }

    static auto_index_gt ip(std::size_t dimensions, accuracy_t accuracy = accuracy_t::f16_k, config_t config = {}) {
        return make(dimensions, accuracy, ip_metric(dimensions, accuracy), make_casts(accuracy), config);
    }

    static auto_index_gt l2(std::size_t dimensions, accuracy_t accuracy = accuracy_t::f16_k, config_t config = {}) {
        return make(dimensions, accuracy, l2_metric(dimensions, accuracy), make_casts(accuracy), config);
    }

    static auto_index_gt cos(std::size_t dimensions, accuracy_t accuracy = accuracy_t::f32_k, config_t config = {}) {
        return {};
    }
    static auto_index_gt haversine(accuracy_t accuracy = accuracy_t::f32_k, config_t config = {}) { return {}; }

    auto_index_gt fork() const {
        auto_index_gt result;

        result.dimensions_ = dimensions_;
        result.accuracy_ = accuracy_;
        result.acceleration_ = acceleration_;
        result.casted_vector_bytes_ = casted_vector_bytes_;
        result.cast_buffer_ = cast_buffer_;
        result.casts_ = casts_;

        result.root_metric_ = root_metric_;
        result.root_config_ = root_config_;
        result.index_.reset(new index_t(result.root_config_, result.root_metric_));

        return result;
    }

  private:
    static auto_index_gt make(                            //
        std::size_t dimensions, accuracy_t accuracy,      //
        metric_and_meta_t metric_and_meta, casts_t casts, //
        config_t config) {

        std::size_t max_threads = std::max(config.max_threads_add, config.max_threads_search);
        auto_index_gt result;
        result.dimensions_ = dimensions;
        result.accuracy_ = accuracy;
        result.casted_vector_bytes_ = bytes_per_scalar(accuracy) * dimensions;
        result.cast_buffer_.resize(max_threads * result.casted_vector_bytes_);
        result.casts_ = casts;
        result.index_.reset(new index_t(config, metric_and_meta.metric));
        result.acceleration_ = metric_and_meta.acceleration;
        result.root_metric_ = metric_and_meta.metric;
        result.root_config_ = config;
        return result;
    }

    template <typename to_scalar_at> static casts_t make_casts() {
        casts_t result;
        result.from_i8q100 = cast_gt<i8q100_converted_t, to_scalar_at>{};
        result.from_f16 = cast_gt<f16_converted_t, to_scalar_at>{};
        result.from_f32 = cast_gt<f32_t, to_scalar_at>{};
        result.from_f64 = cast_gt<f64_t, to_scalar_at>{};
        return result;
    }

    static casts_t make_casts(accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::f64_k: return make_casts<f64_t>();
        case accuracy_t::f32_k: return make_casts<f32_t>();
        case accuracy_t::f16_k: return make_casts<f16_converted_t>();
        case accuracy_t::i8q100_k: return make_casts<i8q100_converted_t>();
        }
    }

    template <typename scalar_at, typename typed_at> //
    static punned_stateful_metric_t pun_metric(typed_at metric) {
        return [=](byte_t const* a, byte_t const* b, std::size_t bytes, std::size_t) noexcept -> float {
            return metric((scalar_at const*)a, (scalar_at const*)b, bytes / sizeof(scalar_at));
        };
    }

    static metric_and_meta_t ip_metric_f32(std::size_t dimensions) {
#if 0
#if defined(__x86_64__)
        if (dimensions % 4 == 0)
            return {
                pun_metric<simsimd_f32_t>([](simsimd_f32_t const* a, simsimd_f32_t const* b, size_t d) noexcept {
                    return 1 - simsimd_dot_f32x4avx2(a, b, d);
                }),
                isa_t::avx2_k,
            };
#elif defined(__aarch64__)
        if (supports_arm_sve())
            return {
                pun_metric<simsimd_f32_t>([](simsimd_f32_t const* a, simsimd_f32_t const* b, size_t d) noexcept {
                    return 1 - simsimd_dot_f32sve(a, b, d);
                }),
                isa_t::sve_k,
            };
        if (dimensions % 4 == 0)
            return {
                pun_metric<simsimd_f32_t>([](simsimd_f32_t const* a, simsimd_f32_t const* b, size_t d) noexcept {
                    return 1 - simsimd_dot_f32x4neon(a, b, d);
                }),
                isa_t::neon_k,
            };
#endif
#endif
        return {pun_metric<f32_t>(ip_gt<f32_t>{}), isa_t::auto_k};
    }

    static metric_and_meta_t ip_metric_f16(std::size_t dimensions) {
#if 0
#if defined(__x86_64__)
#elif defined(__aarch64__)
        if (supports_arm_sve())
            return {
                pun_metric<simsimd_f16_t>([](simsimd_f16_t const* a, simsimd_f16_t const* b, size_t d) noexcept {
                    return 1 - simsimd_dot_f16sve(a, b, d);
                }),
                isa_t::sve_k,
            };
        if (dimensions % 8 == 0)
            return {
                pun_metric<simsimd_f16_t>([](simsimd_f16_t const* a, simsimd_f16_t const* b, size_t d) noexcept {
                    return 1 - simsimd_dot_f16x8neon(a, b, d);
                }),
                isa_t::neon_k,
            };
#endif
#endif
        return {pun_metric<f16_converted_t>(ip_gt<f16_converted_t>{}), isa_t::auto_k};
    }

    static metric_and_meta_t ip_metric(std::size_t dimensions, accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::i8q100_k: return {};
        case accuracy_t::f16_k: return ip_metric_f16(dimensions);
        case accuracy_t::f32_k: return ip_metric_f32(dimensions);
        case accuracy_t::f64_k: return {pun_metric<f64_t>(ip_gt<f64_t>{}), isa_t::auto_k};
        }
    }

    static metric_and_meta_t l2_metric(std::size_t dimensions, accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::i8q100_k: return {};
        case accuracy_t::f16_k: return {pun_metric<f16_converted_t>(l2_squared_gt<f16_converted_t>{}), isa_t::auto_k};
        case accuracy_t::f32_k: return {pun_metric<f32_t>(l2_squared_gt<f32_t>{}), isa_t::auto_k};
        case accuracy_t::f64_k: return {pun_metric<f64_t>(l2_squared_gt<f64_t>{}), isa_t::auto_k};
        }
    }

    static metric_and_meta_t cos_metric(std::size_t dimensions, accuracy_t accuracy) {
        switch (accuracy) {
        case accuracy_t::i8q100_k: return {};
        case accuracy_t::f16_k: return {pun_metric<f16_converted_t>(cos_gt<f16_converted_t>{}), isa_t::auto_k};
        case accuracy_t::f32_k: return {pun_metric<f32_t>(cos_gt<f32_t>{}), isa_t::auto_k};
        case accuracy_t::f64_k: return {pun_metric<f64_t>(cos_gt<f64_t>{}), isa_t::auto_k};
        }
    }
};

using auto_index_t = auto_index_gt<std::int64_t, std::uint32_t>;
using auto_index_big_t = auto_index_gt<uuid_t, uint40_t>;

} // namespace usearch
} // namespace unum
