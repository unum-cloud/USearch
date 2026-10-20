/**
 *  @file include/usearch/geospatial/ram_metric.hpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief Batched squared chords between Earth-fixed positions, and the terrain kernels lines of
 *      sight run on, over NumKong's many-to-many geospatial kernels.
 *
 *  The geospatial index scores one round at a time: @c reset opens it for a query position, @c push
 *  stages the candidates' positions, and @c drain scores the whole round in one 1×M call to
 *  @c nk_sqeuclideans3d_i32, exact in 64-bit integers. Staging and scoring apart is what lets the
 *  kernel vectorize: a score computed at push time is one candidate at a time, which no compiler
 *  widens.
 *
 *  @section geospatial_ram_metric_kits Kits
 *
 *  A kit binds one ISA's 3D squares with the terrain kernels, which are serial on every ISA for
 *  now. @c geospatial_ladder picks the widest kit this build compiled in under the requested
 *  capability mask, at compile time, so no capability is tested at run time.
 */
#pragma once

#include <cstddef> // `std::size_t`

#include <limits> // `std::numeric_limits` - the distance a failed round reports
#include <span>   // `std::span`

#include <numkong/geospatial.h> // `nk_sqeuclideans3d_i32_*`, `nk_clearances_packed_f32_*` and kin

#include "usearch/utilities/types.hpp"
#include "usearch/utilities/ram_buffers.hpp"
#include "usearch/utilities/ram_device.hpp" // `forkunion_allocator_t`
#include "usearch/geospatial/ram_index.hpp"

namespace unum::usearch::geospatial {
using namespace unum::usearch::utilities;

#pragma region Kits

/** The geospatial kernels for any CPU. */
struct i32_serial_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_serial_k;
    static constexpr bool compiled_k = true;
    static constexpr auto sqeuclideans3d = nk_sqeuclideans3d_i32_serial;
    static constexpr auto terrain_pack_size = nk_terrain_pack_size_f32_serial;
    static constexpr auto terrain_pack = nk_terrain_pack_f32_serial;
    static constexpr auto terrain_heights = nk_terrain_heights_packed_f32_serial;
    static constexpr auto clearances = nk_clearances_packed_f32_serial;
};

#if NUMKONG_TARGET_NEON

/** The geospatial kernels for Arm NEON. */
struct i32_neon_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = true;
    static constexpr auto sqeuclideans3d = nk_sqeuclideans3d_i32_neon;
    static constexpr auto terrain_pack_size = nk_terrain_pack_size_f32_serial;
    static constexpr auto terrain_pack = nk_terrain_pack_f32_serial;
    static constexpr auto terrain_heights = nk_terrain_heights_packed_f32_serial;
    static constexpr auto clearances = nk_clearances_packed_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c geospatial_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct i32_neon_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_neon_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_NEON

#if NUMKONG_TARGET_HASWELL

/** The geospatial kernels for Intel Haswell AVX2. */
struct i32_haswell_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = true;
    static constexpr auto sqeuclideans3d = nk_sqeuclideans3d_i32_haswell;
    static constexpr auto terrain_pack_size = nk_terrain_pack_size_f32_serial;
    static constexpr auto terrain_pack = nk_terrain_pack_f32_serial;
    static constexpr auto terrain_heights = nk_terrain_heights_packed_f32_serial;
    static constexpr auto clearances = nk_clearances_packed_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c geospatial_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct i32_haswell_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_haswell_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_HASWELL

#if NUMKONG_TARGET_SKYLAKE

/** The geospatial kernels for Intel Skylake-X AVX-512. */
struct i32_skylake_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = true;
    static constexpr auto sqeuclideans3d = nk_sqeuclideans3d_i32_skylake;
    static constexpr auto terrain_pack_size = nk_terrain_pack_size_f32_serial;
    static constexpr auto terrain_pack = nk_terrain_pack_f32_serial;
    static constexpr auto terrain_heights = nk_terrain_heights_packed_f32_serial;
    static constexpr auto clearances = nk_clearances_packed_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c geospatial_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct i32_skylake_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_skylake_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_SKYLAKE

#if NUMKONG_TARGET_RVV

/** The geospatial kernels for the RISC-V Vector extension. */
struct i32_rvv_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = true;
    static constexpr auto sqeuclideans3d = nk_sqeuclideans3d_i32_rvv;
    static constexpr auto terrain_pack_size = nk_terrain_pack_size_f32_serial;
    static constexpr auto terrain_pack = nk_terrain_pack_f32_serial;
    static constexpr auto terrain_heights = nk_terrain_heights_packed_f32_serial;
    static constexpr auto clearances = nk_clearances_packed_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c geospatial_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct i32_rvv_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_rvv_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_RVV

#if NUMKONG_TARGET_V128RELAXED

/** The geospatial kernels for WebAssembly Relaxed SIMD. */
struct i32_v128relaxed_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = true;
    static constexpr auto sqeuclideans3d = nk_sqeuclideans3d_i32_v128relaxed;
    static constexpr auto terrain_pack_size = nk_terrain_pack_size_f32_serial;
    static constexpr auto terrain_pack = nk_terrain_pack_f32_serial;
    static constexpr auto terrain_heights = nk_terrain_heights_packed_f32_serial;
    static constexpr auto clearances = nk_clearances_packed_f32_serial;
};
#else

/** The kit this build cannot carry, kept so @c geospatial_ladder stays one list rather than a run
 *  of preprocessor branches. */
struct i32_v128relaxed_geospatial_kit_t {
    static constexpr nk_capability_t capability_k = nk_cap_v128relaxed_k;
    static constexpr bool compiled_k = false;
};
#endif // NUMKONG_TARGET_V128RELAXED

/** The widest geospatial kit this build compiled in among the backends @p caps_ admits, newest
 *  first; each kit states its @c capability_k and its @c compiled_k, so an unselected kit's body
 *  never has to exist. */
template <nk_capability_t caps_>
struct geospatial_ladder;

template <nk_capability_t caps_>
    requires((caps_ & nk_cap_cpus_k) != 0)
struct geospatial_ladder<caps_> {
    using kit_t = typename highest_kit<  //
        caps_,                           //
        i32_serial_geospatial_kit_t,     //
        i32_skylake_geospatial_kit_t,    //
        i32_haswell_geospatial_kit_t,    //
        i32_neon_geospatial_kit_t,       //
        i32_rvv_geospatial_kit_t,        //
        i32_v128relaxed_geospatial_kit_t //
        >::type;
    static constexpr nk_capability_t picked_k = kit_t::capability_k;
};

#pragma endregion Kits

#pragma region Metric

/**
 *  @brief The batched squared-chord metric: positions staged per round and scored in one N×M kernel
 *      call over the N queries a reset opened, exact in 64-bit integers.
 *
 *  Transparent, so a round may open on a stored observation, a bare position, or a geodetic one,
 *  converted once here rather than by every caller. The round's query stays readable through
 *  @ref query, which is how the octant selector learns the base node without the selector concept
 *  naming it.
 *
 *  @tparam kit_type_ The kernels, as one @c geospatial_ladder names them.
 *  @tparam allocator_type_ Places the staging and the kernel's squares.
 */
template <typename kit_type_, typename allocator_type_ = forkunion_allocator_t>
class chord_metric {
  public:
    using kit_t = kit_type_;
    using distance_t = i64_t;
    using allocator_t = allocator_type_;
    using is_transparent = void;

  private:

    /** The round's query, the first one of a batch, in the stored form. */
    earth_fixed_t query_ {};

    /** Every query's three axes as the kernel reads them, @b [queries,3]. */
    growing_buffer<i32_t, allocator_t> query_coordinates_;

    /** The round's candidate positions, three axes each, in push order. */
    growing_buffer<i32_t, allocator_t> staged_;

    /** The kernel's exact @b [queries,candidates] squares. */
    growing_buffer<nk_u64_t, allocator_t> squares_;

    /** Candidates the round holds, and the most it may. */
    std::size_t staged_count_ = 0;
    std::size_t staging_limit_ = 0;

    /** The stored form of every query form a round opens on. */
    static earth_fixed_t const& fixed_of_(earth_fixed_t const& query) noexcept { return query; }
    static earth_fixed_t const& fixed_of_(earth_observation_t const& query) noexcept { return query.position; }
    static earth_fixed_t fixed_of_(geodetic_t const& query) noexcept { return to_earth_fixed(query); }

  public:
    explicit chord_metric(allocator_t const& allocator = {}) noexcept
        : query_coordinates_(allocator), staged_(allocator), squares_(allocator) {}
    chord_metric(chord_metric const&) = delete;
    chord_metric& operator=(chord_metric const&) = delete;
    chord_metric(chord_metric&&) noexcept = default;
    chord_metric& operator=(chord_metric&&) noexcept = default;

    /**
     *  @brief Sizes the staging and the squares.
     *  @param[in] staging_limit How many candidates one round may hold before @c push refuses; at
     *      least the index's round capacity, or rounds saturate rather than fail.
     */
    status_t init(std::size_t staging_limit) noexcept {
        if (!staging_limit) return status_t::config_mismatch_k;
        if (status_t const queried = query_coordinates_.reserve(3); failed(queried)) return queried;
        if (status_t const staged = staged_.resize_uninitialized(staging_limit * 3); failed(staged)) return staged;
        if (status_t const squared = squares_.resize_uninitialized(staging_limit); failed(squared)) return squared;
        staging_limit_ = staging_limit;
        return status_t::success_k;
    }

    /** The position the current round scores against. */
    earth_fixed_t const& query() const noexcept { return query_; }

    /** Candidates one round may hold, as @ref init sized it. */
    std::size_t staging_limit() const noexcept { return staging_limit_; }

    /** Opens a round for a stored-form query, discarding whatever the previous round staged. */
    void reset(earth_fixed_t const& query) noexcept {
        query_ = query;
        query_coordinates_.clear();
        append_query_(query);
        staged_count_ = 0;
    }

    /** Opens a round for a stored observation's position. */
    void reset(earth_observation_t const& query) noexcept { reset(query.position); }

    /** Opens a round for a geodetic query, converting it once. */
    void reset(geodetic_t const& query) noexcept { reset(to_earth_fixed(query)); }

    /** Opens rounds scoring every query of @p queries at once, each converted once, so each
     *  @ref drain writes @b [queries,candidates]; a refused growth leaves no query, and every
     *  drain reports the largest distance. */
    template <typename query_type_>
        requires requires(query_type_ const& query) { fixed_of_(query); }
    void reset(std::span<query_type_ const> queries) noexcept {
        query_coordinates_.clear();
        staged_count_ = 0;
        if (failed(query_coordinates_.reserve(queries.size() * 3)) ||
            failed(squares_.resize_uninitialized(queries.size() * staging_limit_)))
            return;
        for (query_type_ const& query : queries) append_query_(fixed_of_(query));
        if (!queries.empty()) query_ = fixed_of_(queries.front());
    }

    /** Stages one candidate's position, or refuses once the round is full. */
    basic::metric_staging_t push(earth_observation_t const& candidate) noexcept {
        if (staged_count_ == staging_limit_) return basic::metric_staging_t::refused_k;
        i32_t* const axes = staged_.data() + staged_count_ * 3;
        axes[0] = candidate.position.x_centimetres;
        axes[1] = candidate.position.y_centimetres;
        axes[2] = candidate.position.z_centimetres;
        ++staged_count_;
        return basic::metric_staging_t::staged_k;
    }

    /** Scores the round in one N×M call and closes it, writing @b [queries,candidates]; squared
     *  chords between Earth positions stay below 2⁶³, and a failed round reports the largest
     *  distance. */
    void drain(std::span<i64_t> distances, i64_t) noexcept {
        std::size_t const queries = query_coordinates_.size() / 3;
        nk_status_t const scored = queries * staged_count_ != distances.size()
                                       ? nk_unexpected_dimensions_k
                                       : kit_t::sqeuclideans3d(query_coordinates_.data(), queries, staged_.data(),
                                                               staged_count_, squares_.data(),
                                                               sizeof(nk_u64_t) * staged_count_, nullptr);
        for (std::size_t lane = 0; lane != distances.size(); ++lane)
            distances[lane] = scored == nk_success_k ? static_cast<i64_t>(squares_[lane])
                                                     : (std::numeric_limits<i64_t>::max)();
        staged_count_ = 0;
    }

  private:

    /** Appends @p query 's three axes, within the reserved queries. */
    void append_query_(earth_fixed_t const& query) noexcept {
        query_coordinates_.push_back_reserved(query.x_centimetres);
        query_coordinates_.push_back_reserved(query.y_centimetres);
        query_coordinates_.push_back_reserved(query.z_centimetres);
    }
};

using chord_metric_t = chord_metric<i32_serial_geospatial_kit_t>;

static_assert(basic::metric<chord_metric_t, earth_observation_t, i64_t>);
static_assert(basic::transparent_metric<chord_metric_t>);

#pragma endregion Metric

} // namespace unum::usearch::geospatial
