/**
 *  @file src/geospatial_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief What one geospatial variant is spelled with and compiled through: the crossings of an
 *      observation, the geometry every geospatial variant adopts, the terrain and the judging of
 *      relations, and the variants over the basic indexes.
 *
 *  Every geospatial class unit in @c src/geospatial/ includes this to instantiate its variants, and
 *  @c geospatial_index.cpp adds the variant, the traits and the exported C entry points on top of
 *  it.
 */
#pragma once

#include <cmath> // `std::sqrt`, `std::floor`, `std::ceil`, `std::fmin`, `std::fmax`

#include <array>       // `std::array` - a quad of stored observations, a sight line's points
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid128_t` share sixteen bytes
#include <memory>      // `std::unique_ptr` - a GPU shard built on its device
#include <new>         // `std::nothrow` - a GPU shard built on its device
#include <span>        // `std::span` - a quad's entries and verdicts
#include <string_view> // `std::string_view` - tokens and paths

#include "usearch/geospatial/ram_metric.hpp"
#include "usearch/shapes/ram_metric.hpp" // `us::shapes::status_of` - NumKong's statuses crossing
#if USEARCH_ARCH_CUDA_
#include "usearch/geospatial/vram_nvidia_index.hpp"
#endif

#include "basic_specializations.cuh"

#pragma region Geospatial Variants

/** The engine's form of a C observation, field by field. */
inline us::geospatial::geodetic_observation_t geodetic_of(
    usearch_geospatial_observation_t const& observation) noexcept {
    return {observation.latitude_degrees,
            observation.longitude_degrees,
            observation.altitude_metres,
            observation.accuracy_metres,
            observation.heading_degrees,
            observation.pitch_degrees,
            observation.heading_accuracy_degrees,
            observation.field_of_view_degrees,
            observation.range_metres,
            observation.begin_seconds,
            observation.end_seconds};
}

/** The C form of an engine observation, field by field. */
inline usearch_geospatial_observation_t observation_of(
    us::geospatial::geodetic_observation_t const& observation) noexcept {
    return {observation.latitude_degrees,
            observation.longitude_degrees,
            observation.altitude_metres,
            observation.accuracy_metres,
            observation.heading_degrees,
            observation.pitch_degrees,
            observation.heading_accuracy_degrees,
            observation.field_of_view_degrees,
            observation.range_metres,
            observation.begin_seconds,
            observation.end_seconds};
}

/** One containment match before it is written: the query it answers, the entry's key and its chord
 *  in metres. */
struct geospatial_within_match_t {
    std::size_t query;
    usearch_key_t key;
    usearch_distance_t score;
};

/** One thread's line-of-sight scratch, kept across quads and calls. */
struct line_of_sight_buffers_t {
    growing_buffer<f32_t, thread_allocator_t> window_heights;
    growing_buffer<byte_t, thread_allocator_t> window_pack;

    explicit line_of_sight_buffers_t(thread_allocator_t const& allocator) noexcept
        : window_heights(allocator), window_pack(allocator) {}
};

/**
 *  @brief What every geospatial variant on the host shares, whatever holds its observations: the
 *      metric one class's kit scores with, the terrain its lines of sight run over, and the tests
 *      a quad of stored observations passes.
 *
 *  A borrowed terrain is packed once, at @ref adopt_terrain, and read in place by every call; a
 *  terrain filled by a callback is filled and packed per call, over the window the call's positions
 *  span. Its verbs are instantiated in the class's unit beside the variants deriving from it.
 */
template <cpu_class_t cpu_class_>
struct geospatial_terrain {
    using kit_t = typename us::geospatial::geospatial_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;
    using value_t = us::geospatial::earth_observation_t;
    using metric_t = us::geospatial::chord_metric<kit_t, thread_allocator_t>;
    using relation_t = us::geospatial::relation_t;

    /** The grid lines of sight run over, empty for none. */
    us::geospatial::terrain_t grid;

    /** The maxima of a borrowed grid, packed once. */
    growing_buffer<byte_t, forkunion_allocator_t> terrain_pack {forkunion_allocator_t {}};

    /** Adopts @p grid, packing it when it is borrowed whole; refuses a grid with no extent or
     *  steps. */
    status_t adopt_terrain(us::geospatial::terrain_t const& adopted) noexcept;

    /** Whether a terrain was adopted, which lines of sight need. */
    bool has_terrain() const noexcept { return !grid.empty(); }

    /** The bytes the packed terrain holds. */
    std::size_t terrain_memory_usage() const noexcept { return terrain_pack.memory_usage(); }

    /** The terrain's height under @p observation when it records no altitude, else zero; the
     *  ellipsoid's own zero off the grid or beside a missing cell. */
    f64_t ground_metres(usearch_geospatial_observation_t const& observation,
                        line_of_sight_buffers_t& sight) const noexcept;

    /** A factory of the variant's metric over a thread's arena, sized for a round of candidates. */
    static auto metric_factory() noexcept {
        return [](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            metric_t made {allocator};
            status_t const ready = made.init(staging);
            return expected<metric_t> {std::move(made), ready};
        };
    }

    /** One line-of-sight scratch per thread of @p executor, each over that thread's arena. */
    static status_t prepare_sights(growing_buffer<line_of_sight_buffers_t, thread_allocator_t>& sights,
                                   forkunion_executor_t& executor) noexcept;

    /** Asks the relation about stored @p entries against @p query into @p admissions: the relation
     *  and time, then the terrain. */
    template <std::size_t extent_>
    void ask_relation_values(relation_t relation, usearch_geospatial_tolerance_t tolerance, value_t const& query,
                             std::span<value_t const, extent_> entries, std::span<admission_t, extent_> admissions,
                             line_of_sight_buffers_t& sight) const noexcept;

  private:

    /** Keeps the admissions of @p entries whose sight lines over the terrain clear it under
     *  @p relation, rejecting the rest; a pair off the grid counts as clear. */
    template <std::size_t extent_>
    void clear_sight_(relation_t relation, value_t const& query, std::span<value_t const, extent_> entries,
                      std::span<admission_t, extent_> admissions, line_of_sight_buffers_t& sight) const noexcept;
};

/** One geospatial variant: the basic shards over observations, scored with one class's kit, with
 *  the terrain its lines of sight run over. */
template <cpu_class_t cpu_class_>
struct geospatial_ram_variant {
    using executor_t = forkunion_executor_t;
    template <typename word_type_>
    using atomic_ref = typename atomic_ref_for_cpu_class<cpu_class_>::template type<word_type_>;
    using kit_t = typename us::geospatial::geospatial_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;
    using cache_hints_t = typename cache_hints_for_cpu_class<cpu_class_>::type;
    using shard_t = us::geospatial::ram_index<us::utilities::uuid128_t, waiter_of<cpu_class_>,
                                              atomic_ref_for_cpu_class<cpu_class_>::template type,
                                              forkunion_allocator_t, cache_hints_t>;
    using index_t = basic::ram_indexes<shard_t>;
    using value_t = us::geospatial::earth_observation_t;
    using node_id_t = typename shard_t::node_id_t;
    using scored_node_t = typename shard_t::scored_node_t;
    using metric_t = us::geospatial::chord_metric<kit_t, thread_allocator_t>;
    using relation_t = us::geospatial::relation_t;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;

    index_t index;
    geospatial_geometry_t geometry;

    /** The metric, the terrain and the relation tests every call of the variant reads. */
    geospatial_terrain<cpu_class_> terrain;

    /** The largest stored deviation in whole metres, raised through @c atomic_ref; it widens every
     *  radius walk. */
    u32_t largest_deviation_metres = 0;

    status_t adopt_terrain(us::geospatial::terrain_t const& grid) noexcept { return terrain.adopt_terrain(grid); }
    bool has_terrain() const noexcept { return terrain.has_terrain(); }
    std::size_t terrain_memory_usage() const noexcept { return terrain.terrain_memory_usage(); }

    explicit geospatial_ram_variant(forkunion_allocator_t const& allocator) noexcept
        : index(allocator, machine_cache_line_bytes()) {}

    /** Reserves the entries @p capacity names. */
    status_t reserve(reserve_capacity_t capacity) noexcept { return index.reserve(capacity.entries).status; }

    /** Empties the index and forgets its largest deviation. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared)) atomic_ref<u32_t>(largest_deviation_metres).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** Long links every insertion draws, appended past the selector's budget. */
    std::size_t long_links() const noexcept {
        constexpr std::size_t cones_k = us::geospatial::surface_keep_octants_t::cones_k;
        return geometry.connectivity > cones_k ? geometry.connectivity - cones_k : 0;
    }

    /** The widest neighbor list a walk copies, which sizes every round's staging. */
    std::size_t widest_list() const noexcept { return geometry.connectivity + long_links(); }

    expected<traversal_stats_t> insert(usearch_keys_view_t keys, observations_view_t observations,
                                       insert_config_t config, executor_t& executor, progress_ticks_t& ticks) noexcept;
    expected<traversal_stats_t> search(geospatial_queries_t query_batch, matches_outputs_t& outputs,
                                       search_config_t search, predicate_t const& predicate, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Opens a refinement that first moves every shard of each of @p donors into this index, then
     *  re-links across, its walks as wide as @p expansion or the 128 exact cone winners need, as an
     *  insertion's; the cone selector keeps the connectivity, and the long links past it yield. */
    expected<refine_plan_t> plan_absorb(std::span<index_t*> donors, refine_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        std::size_t const breadth = config.expansion_factor
                                        ? config.expansion_factor
                                        : us::geospatial::surface_keep_octants_t::exact_candidates_k;
        return index.plan_absorb(donors, {breadth, config.memory_limit},
                                 us::geospatial::surface_keep_octants_t {geometry.connectivity}, planning);
    }

    /** Opens a refinement of every live shard in place, its walks as wide as @p expansion or the
     *  exact cone winners need. */
    expected<refine_plan_t> plan_refine(refine_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        std::size_t const breadth = config.expansion_factor
                                        ? config.expansion_factor
                                        : us::geospatial::surface_keep_octants_t::exact_candidates_k;
        return index.plan_refine({breadth, config.memory_limit},
                                 us::geospatial::surface_keep_octants_t {geometry.connectivity}, planning);
    }

    /** Opens a split of every live shard into the same shard of each of @p receivers, its part
     *  roots drawn from the config's seed, then re-links each receiver's picked nodes, its walks as
     *  wide as those of @ref plan_refine. */
    expected<refine_plan_t> plan_split(std::span<index_t*> receivers, split_config_t config) noexcept {
        typename shard_t::traversal_context_t planning {index.size(), index.get_allocator()};
        std::size_t const breadth = config.refine.expansion_factor
                                        ? config.refine.expansion_factor
                                        : us::geospatial::surface_keep_octants_t::exact_candidates_k;
        return index.plan_split(receivers, {{breadth, config.refine.memory_limit}, config.seed},
                                us::geospatial::surface_keep_octants_t {geometry.connectivity}, planning);
    }

    /** Drives @p plan, then draws every node's long links again by the insertion's rule and seeds,
     *  since the re-selection keeps only the cone selector's budget. */
    status_t refine(refine_plan_t& plan, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Widens the largest deviation to a consumed @p donor 's, whose nodes this index now holds. */
    void count_absorbed_donor(geospatial_ram_variant const& donor) noexcept {
        raise_to<atomic_ref>(largest_deviation_metres,
                             atomic_ref<u32_t const>(donor.largest_deviation_metres).load(std::memory_order_relaxed));
    }

    /** Bounds the deviation of the part @p receiver now holds by this index's, which held it. */
    void count_split_receiver(geospatial_ram_variant& receiver) const noexcept { receiver.count_absorbed_donor(*this); }

    expected<traversal_stats_t> within(geospatial_queries_t query_batch, within_outputs_t& outputs,
                                       search_config_t search, predicate_t const& predicate,
                                       growing_arena<forkunion_allocator_t>& staging, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Writes the checkpoint under a path, or into a buffer of @ref storage_requirements bytes. */
    template <typename checkpoint_type_>
    status_t save(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reads the checkpoint at a path or in a buffer, then widens to its largest deviation. */
    template <typename checkpoint_type_>
    status_t load(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** The bytes a save into memory takes, under the provenance the save records. */
    expected<std::size_t> storage_requirements() const noexcept;

    /** Serving a checkpoint in place answers @c unsupported_k until the index serves it. */
    status_t view(std::string_view) noexcept { return status_t::unsupported_k; }
    expected<sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                     sample_config_t config, executor_t& executor) noexcept;

  private:

    /** Draws the long links of every published node in @p nodes of @p shard, each from a generator
     *  keyed by its ID, so a redraw over a re-selected neighbor list draws exactly what the
     *  insertion's draws drew. */
    status_t draw_long_links_(shard_t& shard, iota_view<node_id_t> nodes, thread_scratch<shard_t, metric_t>& scratch,
                              executor_t& executor) noexcept;

    /** One shard's IDs a batch inserted, kept for the long links drawn once it publishes. */
    struct inserted_range_t {
        shard_t* shard = nullptr;
        iota_view<node_id_t> nodes {};
    };

    /** The relation, time and terrain against @c query as a predicate over a quad of node IDs of
     *  @c shard, asked before the caller's own. */
    struct relation_predicate_t {
        geospatial_ram_variant const& variant;
        shard_t const& shard;
        relation_t relation;
        usearch_geospatial_tolerance_t tolerance;
        value_t const& query;
        line_of_sight_buffers_t& sight;

        template <std::size_t extent_>
        void operator()(std::span<node_id_t const, extent_> nodes,
                        std::span<admission_t, extent_> admissions) const noexcept {
            std::array<value_t, extent_> entries;
            for (std::size_t lane = 0; lane != extent_; ++lane) entries[lane] = shard.published_value_of(nodes[lane]);
            variant.terrain.ask_relation_values(relation, tolerance, query, std::span<value_t const, extent_>(entries),
                                                admissions, sight);
        }
    };

    /** One thread's share of a containment call, kept across the queries it takes: the matches it
     *  found, the ball a scan collects and its line-of-sight scratch. */
    struct within_share_t {
        growing_buffer<geospatial_within_match_t, thread_allocator_t> found;
        growing_buffer<scored_node_t, thread_allocator_t> ball;
        line_of_sight_buffers_t sight;

        explicit within_share_t(thread_allocator_t const& allocator) noexcept
            : found(allocator), ball(allocator), sight(allocator) {}
    };

    /** Writes @p ball of @p shard, ascending, into the matches of query @p query_index in @p share
     *  as far as the relation and @p admits let in; an aggregating @p admits answers at most
     *  @p round_capacity candidates a call. */
    template <typename admits_type_>
    status_t write_related_matches_(shard_t const& shard, relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                    value_t const& query, std::size_t query_index, std::span<scored_node_t const> ball,
                                    admits_type_&& admits, std::size_t round_capacity,
                                    within_share_t& share) const noexcept;
};

template <cpu_class_t cpu_class_>
status_t geospatial_terrain<cpu_class_>::adopt_terrain(us::geospatial::terrain_t const& adopted) noexcept {
    if (adopted.empty()) return status_t::success_k;
    if (!(adopted.row_step_degrees > 0) || !(adopted.column_step_degrees > 0)) return status_t::config_mismatch_k;
    grid = adopted;
    if (!grid.heights_stride) grid.heights_stride = grid.columns * sizeof(f32_t);
    if (!grid.heights) return status_t::success_k;
    nk_size_t bytes = 0;
    if (nk_status_t const sized = kit_t::terrain_pack_size(grid.rows, grid.columns, &bytes); sized != nk_success_k)
        return us::shapes::status_of(sized);
    if (status_t const reserved = terrain_pack.resize_uninitialized(bytes); failed(reserved)) return reserved;
    std::array<f64_t, 2> const steps = grid.step_metres();
    return us::shapes::status_of(kit_t::terrain_pack(grid.heights, grid.rows, grid.columns, grid.heights_stride,
                                                     steps[0], steps[1], terrain_pack.data(), nullptr));
}

template <cpu_class_t cpu_class_>
status_t geospatial_terrain<cpu_class_>::prepare_sights(
    growing_buffer<line_of_sight_buffers_t, thread_allocator_t>& sights, forkunion_executor_t& executor) noexcept {
    if (failed(sights.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        sights.push_back_reserved(line_of_sight_buffers_t {thread_allocator_t {executor.arena(thread)}});
    return status_t::success_k;
}

template <cpu_class_t cpu_class_>
f64_t geospatial_terrain<cpu_class_>::ground_metres(usearch_geospatial_observation_t const& observation,
                                                    line_of_sight_buffers_t& sight) const noexcept {
    if (grid.empty() || observation.altitude_metres == observation.altitude_metres) return 0;
    std::array<f64_t, 2> const cell = grid.grid_point(observation.latitude_degrees, observation.longitude_degrees);
    f64_t const last_row = static_cast<f64_t>(grid.rows - 1), last_column = static_cast<f64_t>(grid.columns - 1);
    if (!(cell[0] >= 0 && cell[0] <= last_column && cell[1] >= 0 && cell[1] <= last_row)) return 0;
    f32_t height = 0;
    if (grid.heights) {
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0]), static_cast<f32_t>(cell[1])};
        if (kit_t::terrain_heights(grid.heights, terrain_pack.data(), point.data(), 1, &height, nullptr) !=
            nk_success_k)
            return 0;
    }
    else {
        // A filled grid answers for the two-by-two window around the point, packed on its own.
        std::size_t const first_column = min_of(static_cast<std::size_t>(cell[0]),
                                                grid.columns > 1 ? grid.columns - 2 : 0);
        std::size_t const first_row = min_of(static_cast<std::size_t>(cell[1]), grid.rows > 1 ? grid.rows - 2 : 0);
        std::size_t const window_rows = min_of<std::size_t>(2, grid.rows),
                          window_columns = min_of<std::size_t>(2, grid.columns);
        nk_size_t bytes = 0;
        if (failed(sight.window_heights.resize_uninitialized(window_rows * window_columns)) ||
            kit_t::terrain_pack_size(window_rows, window_columns, &bytes) != nk_success_k ||
            failed(sight.window_pack.resize_uninitialized(bytes)))
            return 0;
        grid.callback(first_row, first_column, window_rows, window_columns, sight.window_heights.data(),
                      window_columns * sizeof(f32_t), grid.state);
        std::array<f64_t, 2> const steps = grid.step_metres();
        if (kit_t::terrain_pack(sight.window_heights.data(), window_rows, window_columns,
                                window_columns * sizeof(f32_t), steps[0], steps[1], sight.window_pack.data(),
                                nullptr) != nk_success_k)
            return 0;
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0] - static_cast<f64_t>(first_column)),
                                          static_cast<f32_t>(cell[1] - static_cast<f64_t>(first_row))};
        if (kit_t::terrain_heights(sight.window_heights.data(), sight.window_pack.data(), point.data(), 1, &height,
                                   nullptr) != nk_success_k)
            return 0;
    }
    return height == height ? static_cast<f64_t>(height) : 0;
}

template <cpu_class_t cpu_class_>
template <std::size_t extent_>
void geospatial_terrain<cpu_class_>::clear_sight_(relation_t relation, value_t const& query,
                                                  std::span<value_t const, extent_> entries,
                                                  std::span<admission_t, extent_> admissions,
                                                  line_of_sight_buffers_t& sight) const noexcept {
    // The query looks at the entries, except under `sees`, where every entry looks at the query.
    bool const entries_look = relation == relation_t::sees_k;
    std::array<f64_t, 3> const query_cell = grid.grid_point(query.position);
    std::array<std::array<f64_t, 3>, extent_> entry_cells;
    for (std::size_t lane = 0; lane != extent_; ++lane) entry_cells[lane] = grid.grid_point(entries[lane].position);

    // A borrowed grid is read in place; a filled one is fetched over the window the on-grid
    // positions span.
    f32_t const* heights = grid.heights;
    byte_t const* pack = terrain_pack.data();
    f64_t first_row = 0, first_column = 0;
    if (!heights) {
        f64_t const last_row = static_cast<f64_t>(grid.rows - 1), last_column = static_cast<f64_t>(grid.columns - 1);
        f64_t low_row = last_row, high_row = 0, low_column = last_column, high_column = 0;
        auto span_cell = [&](std::array<f64_t, 3> const& cell) noexcept {
            if (!(cell[0] >= 0 && cell[0] <= last_column && cell[1] >= 0 && cell[1] <= last_row)) return;
            low_column = std::fmin(low_column, cell[0]), high_column = std::fmax(high_column, cell[0]);
            low_row = std::fmin(low_row, cell[1]), high_row = std::fmax(high_row, cell[1]);
        };
        span_cell(query_cell);
        for (std::array<f64_t, 3> const& entry_cell : entry_cells) span_cell(entry_cell);
        if (low_row > high_row || low_column > high_column) return;
        first_row = std::floor(low_row), first_column = std::floor(low_column);
        std::size_t const window_rows = min_of<std::size_t>(static_cast<std::size_t>(std::ceil(high_row)) + 1,
                                                            grid.rows) -
                                        static_cast<std::size_t>(first_row);
        std::size_t const window_columns = min_of<std::size_t>(static_cast<std::size_t>(std::ceil(high_column)) + 1,
                                                               grid.columns) -
                                           static_cast<std::size_t>(first_column);
        nk_size_t bytes = 0;
        if (failed(sight.window_heights.resize_uninitialized(window_rows * window_columns)) ||
            kit_t::terrain_pack_size(window_rows, window_columns, &bytes) != nk_success_k ||
            failed(sight.window_pack.resize_uninitialized(bytes)))
            return;
        grid.callback(static_cast<std::size_t>(first_row), static_cast<std::size_t>(first_column), window_rows,
                      window_columns, sight.window_heights.data(), window_columns * sizeof(f32_t), grid.state);
        std::array<f64_t, 2> const steps = grid.step_metres();
        if (kit_t::terrain_pack(sight.window_heights.data(), window_rows, window_columns,
                                window_columns * sizeof(f32_t), steps[0], steps[1], sight.window_pack.data(),
                                nullptr) != nk_success_k)
            return;
        heights = sight.window_heights.data(), pack = sight.window_pack.data();
    }
    auto to_point = [&](std::array<f64_t, 3> const& cell) noexcept {
        return std::array<f32_t, 3> {static_cast<f32_t>(cell[0] - first_column),
                                     static_cast<f32_t>(cell[1] - first_row), static_cast<f32_t>(cell[2])};
    };
    std::array<f32_t, 3> const query_point = to_point(query_cell);
    std::array<f32_t, extent_ * 3> entry_points;
    for (std::size_t lane = 0; lane != extent_; ++lane) {
        std::array<f32_t, 3> const point = to_point(entry_cells[lane]);
        entry_points[lane * 3 + 0] = point[0], entry_points[lane * 3 + 1] = point[1],
                                entry_points[lane * 3 + 2] = point[2];
    }
    constexpr f64_t refraction_k = us::geospatial::terrain_refraction_k;
    std::array<f32_t, extent_> clearances;
    nk_status_t const tested = entries_look ? kit_t::clearances(heights, pack, entry_points.data(), nullptr, extent_,
                                                                query_point.data(), 1, refraction_k, clearances.data(),
                                                                sizeof(f32_t), nullptr)
                                            : kit_t::clearances(heights, pack, query_point.data(), nullptr, 1,
                                                                entry_points.data(), extent_, refraction_k,
                                                                clearances.data(), extent_ * sizeof(f32_t), nullptr);
    if (tested != nk_success_k) return;
    // A negative clearance is an obstruction; NaN leaves the grid, which no terrain can block.
    for (std::size_t lane = 0; lane != extent_; ++lane)
        if (clearances[lane] < 0) admissions[lane] = admission_t::reject_k;
}

template <cpu_class_t cpu_class_>
template <std::size_t extent_>
void geospatial_terrain<cpu_class_>::ask_relation_values(relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                                         value_t const& query,
                                                         std::span<value_t const, extent_> entries,
                                                         std::span<admission_t, extent_> admissions,
                                                         line_of_sight_buffers_t& sight) const noexcept {
    for (std::size_t lane = 0; lane != extent_; ++lane)
        admissions[lane] = us::geospatial::admits(relation, tolerance.metres, tolerance.seconds * 1000.0, query,
                                                  entries[lane])
                               ? admission_t::admit_k
                               : admission_t::reject_k;
    if (relation != relation_t::near_k && !grid.empty()) clear_sight_(relation, query, entries, admissions, sight);
}

template <cpu_class_t cpu_class_>
template <typename admits_type_>
status_t geospatial_ram_variant<cpu_class_>::write_related_matches_(shard_t const& shard, relation_t relation,
                                                                    usearch_geospatial_tolerance_t tolerance,
                                                                    value_t const& query, std::size_t query_index,
                                                                    std::span<scored_node_t const> ball,
                                                                    admits_type_&& admits, std::size_t round_capacity,
                                                                    within_share_t& share) const noexcept {
    relation_predicate_t const related {*this, shard, relation, tolerance, query, share.sight};
    status_t appended = status_t::success_k;
    auto const write_match = [&](scored_node_t const& match) noexcept {
        if (failed(appended)) return;
        appended = share.found.push_back(
            {query_index, std::bit_cast<usearch_key_t>(shard.key_of(match.node_id).value),
             std::sqrt(static_cast<f64_t>(match.distance)) / us::geospatial::centimetres_per_metre_k});
    };
    std::size_t gathered = 0;
    auto const ask_related = [&]<std::size_t extent_>(std::span<scored_node_t const, extent_> scored) noexcept {
        std::array<node_id_t, extent_> ids;
        for (std::size_t lane = 0; lane != extent_; ++lane) ids[lane] = scored[lane].node_id;
        std::array<admission_t, extent_> related_admissions;
        related(std::span<node_id_t const, extent_>(ids), std::span<admission_t, extent_>(related_admissions));
        std::array<node_id_t, extent_> kept_ids;
        std::array<scored_node_t, extent_> kept_scored;
        std::size_t kept = 0;
        for (std::size_t lane = 0; lane != extent_; ++lane) {
            if (related_admissions[lane] != admission_t::admit_k) continue;
            kept_ids[kept] = ids[lane];
            kept_scored[kept] = scored[lane];
            ++kept;
        }
        if (!kept) return;
        // An aggregating predicate answers what it gathered before the quad outgrows its buffers.
        if (gathered + kept > round_capacity) {
            settle_predicate(admits, write_match);
            gathered = 0;
        }
        with_static_extent<extent_>(kept, [&]<std::size_t kept_extent_>() noexcept {
            ask_quad(admits, std::span<node_id_t const, kept_extent_>(kept_ids.data(), kept_extent_),
                     std::span<scored_node_t const, kept_extent_>(kept_scored.data(), kept_extent_), write_match);
        });
        gathered += kept;
    };
    std::size_t first = 0;
    for (; ball.size() - first >= quad_lanes_k; first += quad_lanes_k)
        ask_related(std::span<scored_node_t const, quad_lanes_k>(ball.data() + first, quad_lanes_k));
    with_static_extent<quad_lanes_k>(ball.size() - first, [&]<std::size_t extent_>() noexcept {
        ask_related(std::span<scored_node_t const, extent_>(ball.data() + first, extent_));
    });
    settle_predicate(admits, write_match);
    return appended;
}

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> geospatial_ram_variant<cpu_class_>::insert(usearch_keys_view_t keys,
                                                                       observations_view_t observations,
                                                                       insert_config_t insert, executor_t& executor,
                                                                       progress_ticks_t& ticks) noexcept {
    using key_t = typename index_t::key_t;
    std::size_t const count = observations.size();
    std::size_t const breadth = insert.expansion_factor ? insert.expansion_factor
                                                        : us::geospatial::surface_keep_octants_t::exact_candidates_k;
    std::size_t const list_width = widest_list();
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, executor.size(), list_width, breadth, index.size(),
                                                  terrain.metric_factory());
        failed(prepared))
        return {{}, prepared};

    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<key_t, thread_allocator_t> identities {caller};
    growing_buffer<value_t, thread_allocator_t> values {caller};
    growing_buffer<line_of_sight_buffers_t, thread_allocator_t> sights {caller};
    if (failed(identities.reserve_exactly(count)) || failed(values.reserve_exactly(count)))
        return {{}, status_t::bad_alloc_k};
    if (status_t const prepared = terrain.prepare_sights(sights, executor); failed(prepared)) return {{}, prepared};
    for (std::size_t offset = 0; offset != count; ++offset) {
        identities.push_back_reserved(uuid_of(key_at(keys, offset)));
        values.push_back_reserved(value_t {});
    }
    if (status_t const placed = drain_in_order<atomic_ref_for_cpu_class<cpu_class_>::template type>(
            executor, count,
            [&](std::size_t offset, std::size_t thread) noexcept {
                f64_t const ground = terrain.ground_metres(observations[offset], sights[thread]);
                values[offset] = us::geospatial::to_earth_observation(geodetic_of(observations[offset]), ground);
            });
        failed(placed))
        return {{}, placed};
    u32_t deviation = 0;
    for (value_t const& value : values) deviation = max_of<u32_t>(deviation, value.deviation_metres);

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    us::geospatial::surface_keep_octants_t const selector {geometry.connectivity};
    typename index_t::insert_config_t const config {breadth};
    if (status_t const inserted = drain_in_order<atomic_ref_for_cpu_class<cpu_class_>::template type>(
            executor, count,
            [&](std::size_t offset, std::size_t thread) noexcept {
                plan.insert(offset, to_writable_span(values).subspan(offset, 1), to_span(identities).subspan(offset, 1),
                            scratch.metrics[thread], scratch.traversals[thread], selector, config);
                [[maybe_unused]] continuation_t const heard = ticks(offset);
            });
        failed(inserted))
        return {{}, inserted};
    // Every shard's range is kept before the plan publishes and forgets them; a range the buffer
    // cannot keep goes undrawn, and the batch still publishes, as its entries are already inserted.
    growing_buffer<inserted_range_t, thread_allocator_t> ranges {caller};
    status_t recorded = status_t::success_k;
    plan.for_each_range([&](shard_t& shard, iota_view<node_id_t> nodes) noexcept {
        keep_first_failure<atomic_ref>(recorded, ranges.push_back({&shard, nodes}));
    });
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    raise_to<atomic_ref>(largest_deviation_metres, deviation);
    if (failed(recorded)) return {{}, recorded};

    // Long links reach across the published graph, so they are drawn once the whole batch is
    // visible.
    for (inserted_range_t const& range : ranges)
        if (status_t const linked = draw_long_links_(*range.shard, range.nodes, scratch, executor); failed(linked))
            return {{}, linked};
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
status_t geospatial_ram_variant<cpu_class_>::draw_long_links_(shard_t& shard, iota_view<node_id_t> nodes,
                                                              thread_scratch<shard_t, metric_t>& scratch,
                                                              executor_t& executor) noexcept {
    std::size_t const draws = long_links();
    status_t linked = status_t::success_k;
    status_t const forked = drain_in_order<atomic_ref_for_cpu_class<cpu_class_>::template type>(
        executor, nodes.size(), [&](std::size_t position, std::size_t thread) noexcept {
            node_id_t const node = static_cast<node_id_t>(nodes.first + position);
            splitmix64_t generator = seeded_stream(static_cast<u64_t>(node));
            keep_first_failure<atomic_ref>(linked,
                                           us::geospatial::draw_long_links(shard, node, draws, scratch.metrics[thread],
                                                                           scratch.traversals[thread], generator));
        });
    return failed(forked) ? forked : linked;
}

template <cpu_class_t cpu_class_>
status_t geospatial_ram_variant<cpu_class_>::refine(refine_plan_t& plan, executor_t& executor,
                                                    progress_ticks_t& ticks) noexcept {
    auto [scratch, prepare_status] = prepare_refinement<shard_t>(plan, widest_list(), terrain.metric_factory(),
                                                                 executor);
    if (failed(prepare_status)) return prepare_status;
    if (status_t const refined = drive_refinement(
            plan, scratch, us::geospatial::surface_keep_octants_t {geometry.connectivity}, ticks, executor);
        failed(refined))
        return refined;
    // The re-selection kept the cone winners and the longest of the rest, so the drawn links of
    // every node it rewrote are drawn again by the insertion's rule.
    status_t linked = status_t::success_k;
    plan.for_each_shard_range([&](shard_t& shard, iota_view<node_id_t> nodes) noexcept {
        keep_first_failure<atomic_ref>(linked, draw_long_links_(shard, nodes, scratch, executor));
    });
    return linked;
}

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> geospatial_ram_variant<cpu_class_>::search(
    geospatial_queries_t query_batch, matches_outputs_t& outputs, search_config_t search, predicate_t const& predicate,
    executor_t& executor, progress_ticks_t& ticks) noexcept {
    usearch_geospatial_observation_t const* const queries = query_batch.observations.data();
    relation_t const relation = static_cast<relation_t>(query_batch.relation);
    usearch_geospatial_tolerance_t const tolerance = query_batch.tolerance;
    std::size_t const wanted = outputs.wanted, expansion = search.expansion_factor;
    using key_t = typename index_t::key_t;
    using distance_t = typename index_t::distance_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the merges resolve them");
    // Candidates fewer than the ranking they fill would short every query.
    std::size_t const breadth = max_of(expansion ? expansion : default_expansion_factor(widest_list()), wanted);
    thread_scratch<shard_t, metric_t> scratch {executor};
    if (status_t const prepared = scratch.prepare(executor, executor.size(), widest_list(), breadth, index.size(),
                                                  terrain.metric_factory());
        failed(prepared))
        return {{}, prepared};
    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<line_of_sight_buffers_t, thread_allocator_t> sights {caller};
    if (status_t const prepared = terrain.prepare_sights(sights, executor); failed(prepared)) return {{}, prepared};
    growing_buffer<node_id_t, thread_allocator_t> watermarks {caller};
    growing_buffer<distance_t, thread_allocator_t> distances {caller};
    if (failed(distances.resize_uninitialized(executor.size() * wanted))) return {{}, status_t::bad_alloc_k};
    if (status_t const taken = snapshot_shards(index, watermarks); failed(taken)) return {{}, taken};
    typename index_t::search_config_t const config {breadth};
    using traversal_context_t = typename thread_scratch<shard_t, metric_t>::traversal_context_t;
    status_t searched = status_t::success_k;
    bool const scanning = search.traversal == traversal_t::scan_k;
    status_t const asked = with_key_predicates<scored_node_t>(
        predicate, scratch.round_capacity, executor, [&](auto&& admits_of) noexcept {
            status_t const forked = drain_in_order<atomic_ref_for_cpu_class<cpu_class_>::template type>(
                executor, query_batch.observations.size(), [&](std::size_t query_index, std::size_t thread) noexcept {
                    if (failed(atomic_ref<status_t>(searched).load(std::memory_order_relaxed))) return;
                    line_of_sight_buffers_t& sight = sights[thread];
                    value_t const query = us::geospatial::to_earth_observation(
                        geodetic_of(queries[query_index]), terrain.ground_metres(queries[query_index], sight));
                    traversal_context_t& context = scratch.traversals[thread];
                    metric_t& metric = scratch.metrics[thread];
                    auto const relate = [&](shard_t const& shard) noexcept {
                        return relation_predicate_t {*this, shard, relation, tolerance, query, sight};
                    };
                    auto&& admits = admits_of(thread);
                    std::span<key_t> const keys {reinterpret_cast<key_t*>(outputs.keys) + query_index * wanted, wanted};
                    std::span<distance_t> const scores {distances.data() + thread * wanted, wanted};
                    auto const [found, status] = scanning ? index.scan(query, to_span(watermarks), keys, scores, relate,
                                                                       admits, metric, context)
                                                          : index.search_if(query, to_span(watermarks), keys, scores,
                                                                            config, relate, admits, metric, context);
                    if (failed(status)) return keep_first_failure<atomic_ref>(searched, status);
                    for (std::size_t match = 0; match != found; ++match)
                        outputs.scores[query_index * wanted + match] =
                            std::sqrt(static_cast<usearch_distance_t>(scores[match])) /
                            us::geospatial::centimetres_per_metre_k;
                    outputs.counts[query_index] = found;
                    if (ticks(query_index) == continuation_t::cancel_k)
                        keep_first_failure<atomic_ref>(searched, status_t::cancelled_k);
                });
            keep_first_failure<atomic_ref>(searched, forked);
        });
    if (failed(asked)) return {{}, asked};
    if (failed(searched)) return {{}, searched};
    return {scratch.telemetry(), status_t::success_k};
}

/** Places each draw at its Earth-fixed position in Earth radii and reports each parent's chord in
 *  metres, as search does, whichever residency's walk @p sample runs, called as
 *  <tt>sample(coordinates_of) → expected<sample_result_t></tt>. */
template <typename sample_type_>
expected<sample_result_t> sample_on_earth(basic_sample_outputs_t outputs, sample_type_&& sample) noexcept {
    constexpr f64_t radius_centimetres_k = NUMKONG_EARTH_ELLIPSOID_EQUATORIAL_RADIUS *
                                           us::geospatial::centimetres_per_metre_k;
    auto const coordinates_of = [](us::geospatial::earth_observation_t const& value, std::span<f32_t, 3> xyz) noexcept {
        xyz[0] = static_cast<f32_t>(value.position.x_centimetres / radius_centimetres_k);
        xyz[1] = static_cast<f32_t>(value.position.y_centimetres / radius_centimetres_k);
        xyz[2] = static_cast<f32_t>(value.position.z_centimetres / radius_centimetres_k);
    };
    expected<sample_result_t> const sampled = sample(coordinates_of);
    if (failed(sampled.status)) return sampled;
    // The metric scores squared centimetres, which search reports as metres.
    for (f32_t& distance : outputs.parent_distances.first(sampled.value.written))
        distance = static_cast<f32_t>(std::sqrt(static_cast<f64_t>(distance)) /
                                      us::geospatial::centimetres_per_metre_k);
    return sampled;
}

template <cpu_class_t cpu_class_>
expected<sample_result_t> geospatial_ram_variant<cpu_class_>::sample(basic_sample_inputs_t inputs,
                                                                     basic_sample_outputs_t outputs,
                                                                     sample_config_t config,
                                                                     executor_t& executor) noexcept {
    return sample_on_earth(outputs, [&](auto const& coordinates_of) noexcept {
        return sample_entries(index, widest_list(), terrain.metric_factory(), inputs, outputs, config, executor,
                              coordinates_of);
    });
}

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> geospatial_ram_variant<cpu_class_>::within(
    geospatial_queries_t query_batch, within_outputs_t& outputs, search_config_t search, predicate_t const& predicate,
    growing_arena<forkunion_allocator_t>& staging, executor_t& executor, progress_ticks_t& ticks) noexcept {
    usearch_geospatial_observation_t const* const queries = query_batch.observations.data();
    std::size_t const count = query_batch.observations.size();
    relation_t const relation = static_cast<relation_t>(query_batch.relation);
    usearch_geospatial_tolerance_t const tolerance = query_batch.tolerance;
    using distance_t = typename shard_t::distance_t;
    using traversal_context_t = typename thread_scratch<shard_t, metric_t>::traversal_context_t;
    using ball_t = growing_buffer<scored_node_t, thread_allocator_t>;

    // Collects every scanned entry within `reach_squared`, the containment a scan reports.
    struct ball_fold_t {
        ball_t& ball;
        distance_t reach_squared;
        status_t status = status_t::success_k;

        walk_demand_t absorb(std::span<node_id_t const> node_ids, std::span<distance_t const> distances) noexcept {
            for (std::size_t lane = 0; lane != node_ids.size(); ++lane)
                if (!(distances[lane] > reach_squared))
                    if (status = ball.push_back({node_ids[lane], distances[lane]}); failed(status))
                        return walk_demand_t::sated_k;
            return walk_demand_t::continue_k;
        }
    };
    thread_scratch<shard_t, metric_t> scratch {executor};
    std::size_t breadth = 0;
    for (shard_t const& one : index.live_shards())
        breadth = max_of(breadth, one.non_zero_expansion_factor(typename shard_t::search_config_t {}));
    if (status_t const prepared = scratch.prepare(executor, executor.size(), widest_list(), breadth, index.size(),
                                                  terrain.metric_factory());
        failed(prepared))
        return {{}, prepared};
    growing_buffer<within_share_t, thread_allocator_t> shares {thread_allocator_t {executor.arena(0)}};
    if (failed(shares.reserve_exactly(executor.size()))) return {{}, status_t::bad_alloc_k};
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        shares.push_back_reserved(within_share_t {thread_allocator_t {executor.arena(thread)}});
    f64_t const largest_deviation = atomic_ref<u32_t>(largest_deviation_metres).load(std::memory_order_relaxed);

    status_t walked = status_t::success_k;
    status_t const asked = with_key_predicates<scored_node_t>(
        predicate, scratch.round_capacity, executor,
        [&]<typename admits_of_type_>(admits_of_type_&& admits_of) noexcept {
            using admits_t = std::invoke_result_t<admits_of_type_&, std::size_t>;
            status_t const forked = drain_in_order<atomic_ref_for_cpu_class<cpu_class_>::template type>(
                executor, count, [&](std::size_t query_index, std::size_t thread) noexcept {
                    if (failed(atomic_ref<status_t>(walked).load(std::memory_order_relaxed))) return;
                    within_share_t& share = shares[thread];
                    value_t const query = us::geospatial::to_earth_observation(
                        geodetic_of(queries[query_index]), terrain.ground_metres(queries[query_index], share.sight));
                    // The walk reaches as far as the widest pair could match, and `admits` tightens
                    // back to each pair's own.
                    f64_t const deviations = std::sqrt(static_cast<f64_t>(query.deviation_metres) *
                                                           query.deviation_metres +
                                                       largest_deviation * largest_deviation);
                    i64_t const reach = static_cast<i64_t>(
                        std::ceil((tolerance.metres + 3 * deviations) * us::geospatial::centimetres_per_metre_k));
                    traversal_context_t& context = scratch.traversals[thread];
                    admits_t admits = admits_of(thread);
                    // Each shard's ball appends to the query's matches, ascending within the shard.
                    for (shard_t& one : index.live_shards()) {
                        // A walk leaves the ball at the front of the candidates; a scan collects it
                        // from every entry.
                        std::span<scored_node_t const> ball;
                        if (search.traversal == traversal_t::scan_k) {
                            share.ball.clear();
                            ball_fold_t fold {share.ball, static_cast<distance_t>(reach) * reach};
                            auto const [found_entries, status] = one.scan(query.position, fold, scratch.metrics[thread],
                                                                          context);
                            if (failed(status)) return keep_first_failure<atomic_ref>(walked, status);
                            if (failed(fold.status)) return keep_first_failure<atomic_ref>(walked, fold.status);
                            partial_sort(to_writable_span(share.ball), share.ball.size(),
                                         typename scored_node_t::closer_t {});
                            ball = to_span(share.ball);
                        }
                        else {
                            us::geospatial::radius_result_t const walked_ball = us::geospatial::within(
                                one, query.position, reach, us::geospatial::radius_margin_t::spanner_k, widest_list(),
                                scratch.metrics[thread], context);
                            if (!walked_ball) return keep_first_failure<atomic_ref>(walked, walked_ball.status);
                            ball = to_span(context.results).first(walked_ball.count);
                        }
                        auto const key_of = [&one](node_id_t node_id) noexcept { return one.key_of(node_id).value; };
                        keep_first_failure<atomic_ref>(
                            walked,
                            write_related_matches_(one, relation, tolerance, query, query_index, ball,
                                                   node_predicate {admits, key_of}, scratch.round_capacity, share));
                    }
                    if (ticks(query_index) == continuation_t::cancel_k)
                        keep_first_failure<atomic_ref>(walked, status_t::cancelled_k);
                });
            keep_first_failure<atomic_ref>(walked, forked);
        });
    if (failed(asked)) return {{}, asked};
    if (failed(walked)) return {{}, walked};

    // Each query ranked whole on one thread in ascending order, so counting per query places every
    // match.
    std::size_t* const offsets = carve<std::size_t>(staging, count + 1);
    if (!offsets) return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0; query != count + 1; ++query) offsets[query] = 0;
    std::size_t total = 0;
    for (within_share_t const& share : shares)
        for (geospatial_within_match_t const& match : share.found) ++offsets[match.query + 1], ++total;
    for (std::size_t query = 0; query != count; ++query) offsets[query + 1] += offsets[query];
    usearch_key_t* const keys = carve<usearch_key_t>(staging, total);
    usearch_distance_t* const scores = carve<usearch_distance_t>(staging, total);
    growing_buffer<std::size_t, thread_allocator_t> cursors {thread_allocator_t {executor.arena(0)}};
    if ((total && (!keys || !scores)) || failed(cursors.resize_uninitialized(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0; query != count; ++query) cursors[query] = offsets[query];
    for (within_share_t const& share : shares)
        for (geospatial_within_match_t const& match : share.found) {
            std::size_t const slot = cursors[match.query]++;
            keys[slot] = match.key, scores[slot] = match.score;
        }
    outputs.matches = {keys, scores, offsets};
    outputs.written = total;
    return {scratch.telemetry(), status_t::success_k};
}

/** Hands @p write, as <tt>write(config) → status_t</tt>, the save configuration recording the
 *  connectivity @p geometry froze, as every residency's checkpoint records it. */
template <typename write_type_>
status_t with_geospatial_provenance(geospatial_geometry_t const& geometry, write_type_&& write) noexcept {
    std::array<char, 24> connectivity_text;
    std::array<safegraphs_custom_field_t, 1> const fields {
        {{"connectivity", decimal_of(geometry.connectivity, connectivity_text)}}};
    save_config_t config;
    config.provenance = fields;
    return write(config);
}

template <cpu_class_t cpu_class_>
template <typename checkpoint_type_>
status_t geospatial_ram_variant<cpu_class_>::save(checkpoint_type_ checkpoint, executor_t& executor,
                                                  progress_ticks_t& ticks) noexcept {
    return with_geospatial_provenance(geometry, [&](save_config_t const& config) noexcept {
        return index.save(checkpoint, us::geospatial::earth_observation_codec_t {}, executor, config,
                          token_of(geometry), ticks);
    });
}

template <cpu_class_t cpu_class_>
expected<std::size_t> geospatial_ram_variant<cpu_class_>::storage_requirements() const noexcept {
    expected<std::size_t> required {0, status_t::success_k};
    status_t const sized = with_geospatial_provenance(geometry, [&](save_config_t const& config) noexcept {
        required = index.storage_requirements(us::geospatial::earth_observation_codec_t {}, config, token_of(geometry));
        return required.status;
    });
    return {required.value, sized};
}

template <cpu_class_t cpu_class_>
template <typename checkpoint_type_>
status_t geospatial_ram_variant<cpu_class_>::load(checkpoint_type_ checkpoint, executor_t& executor,
                                                  progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = index.load(checkpoint, us::geospatial::earth_observation_codec_t {}, executor,
                                           load_config_t {}, ticks);
        failed(loaded))
        return loaded;
    for (shard_t const& one : index.live_shards())
        for (std::size_t node = 0; node != one.size(); ++node)
            raise_to<atomic_ref>(largest_deviation_metres,
                                 one.published_value_of(static_cast<node_id_t>(node)).deviation_metres);
    return status_t::success_k;
}

/** One geospatial alternative on block storage: the basic NVMe index over the observations a RAM
 *  variant saved, read-only, scored with the RAM variant's metric and judged by its relations over
 *  the same terrain, each read entry tested before its read buffer recycles. Its verbs are
 *  instantiated in the class's unit, as the RAM variant's are. */
template <cpu_class_t cpu_class_>
struct geospatial_nvme_variant {
    using executor_t = nvme_executor_t;
    template <typename word_type_>
    using atomic_ref = typename atomic_ref_for_cpu_class<cpu_class_>::template type<word_type_>;
    using index_t = basic::nvme_indexes<us::geospatial::earth_observation_codec_t, us::utilities::uuid128_t, u32_t,
                                        nk_u64_t, default_io_engine_t, row_kit_of<cpu_class_>,
                                        atomic_ref_for_cpu_class<cpu_class_>::template type, waiter_of<cpu_class_>>;
    using value_t = us::geospatial::earth_observation_t;
    using relation_t = us::geospatial::relation_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;

    index_t index;
    geospatial_geometry_t geometry;

    /** The metric, the terrain and the relation tests every call of the view reads. */
    geospatial_terrain<cpu_class_> terrain;

    explicit geospatial_nvme_variant(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    status_t adopt_terrain(us::geospatial::terrain_t const& grid) noexcept { return terrain.adopt_terrain(grid); }
    bool has_terrain() const noexcept { return terrain.has_terrain(); }
    std::size_t terrain_memory_usage() const noexcept { return terrain.terrain_memory_usage(); }

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        return index.view(path, paths, token_of(geometry), executor);
    }

    expected<traversal_stats_t> search(geospatial_queries_t query_batch, matches_outputs_t& outputs,
                                       search_config_t search, predicate_t const& predicate, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Reports every entry within the tolerance that the relation and @p predicate admit through
     *  one exact scan of every entry, whichever traversal @p search asks: a view holds no radius
     *  walk, and the scan reports the set the complete walk would, ascending per query. */
    expected<traversal_stats_t> within(geospatial_queries_t query_batch, within_outputs_t& outputs,
                                       search_config_t search, predicate_t const& predicate,
                                       growing_arena<forkunion_allocator_t>& staging, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** A view already is its checkpoint, so saving answers @c unsupported_k. */
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    expected<sample_result_t> sample(basic_sample_inputs_t inputs, basic_sample_outputs_t outputs,
                                     sample_config_t config, executor_t& executor) noexcept;

  private:

    /** One thread's share of a containment scan: the matches it found and its line-of-sight
     *  scratch. */
    struct within_share_t {
        growing_buffer<geospatial_within_match_t, thread_allocator_t> found;
        line_of_sight_buffers_t sight;

        explicit within_share_t(thread_allocator_t const& allocator) noexcept : found(allocator), sight(allocator) {}
    };

    /** The relation, time and terrain one executor thread asks about stored observations, over its
     *  own line-of-sight scratch. */
    struct thread_relation_t {
        geospatial_nvme_variant const& variant;
        relation_t relation;
        usearch_geospatial_tolerance_t tolerance;
        line_of_sight_buffers_t sight;

        template <std::size_t extent_>
        void operator()(value_t const& query, std::span<value_t const, extent_> entries,
                        std::span<admission_t, extent_> admissions) noexcept {
            variant.terrain.ask_relation_values(relation, tolerance, query, entries, admissions, sight);
        }
    };

    /** One relation per executor thread, rebuilt for every call of the view, since each call
     *  rewinds the arenas their sights grow in. */
    struct geospatial_relations_t {
        using thread_predicate_t = thread_relation_t;

        geospatial_nvme_variant const& variant;
        relation_t relation;
        usearch_geospatial_tolerance_t tolerance;
        growing_buffer<thread_relation_t, forkunion_allocator_t> threads {forkunion_allocator_t {}};

        /** Builds one relation per thread of @p executor, each over that thread's arena. */
        status_t prepare(executor_t& executor) noexcept {
            threads.clear();
            if (failed(threads.reserve_exactly(executor.size()))) return status_t::bad_alloc_k;
            for (std::size_t thread = 0; thread != executor.size(); ++thread)
                threads.push_back_reserved(
                    thread_relation_t {variant, relation, tolerance,
                                       line_of_sight_buffers_t {thread_allocator_t {executor.arena(thread)}}});
            return status_t::success_k;
        }

        thread_relation_t& of_thread(std::size_t thread) noexcept { return threads[thread]; }
    };
};

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> geospatial_nvme_variant<cpu_class_>::within(
    geospatial_queries_t query_batch, within_outputs_t& outputs, search_config_t, predicate_t const& predicate,
    growing_arena<forkunion_allocator_t>& staging, executor_t& executor, progress_ticks_t& ticks) noexcept {
    usearch_geospatial_observation_t const* const queries = query_batch.observations.data();
    std::size_t const count = query_batch.observations.size();
    relation_t const relation = static_cast<relation_t>(query_batch.relation);
    usearch_geospatial_tolerance_t const tolerance = query_batch.tolerance;
    using key_t = typename index_t::key_t;
    using distance_t = typename index_t::distance_t;
    // No stored deviation bounds the view's entries, so every entry reaches every query and the
    // relation tightens each pair to its own reach.
    distance_t* const reaches = carve<distance_t>(staging, count);
    value_t* const observations = carve<value_t>(staging, count);
    if (!reaches || !observations) return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0; query != count; ++query) reaches[query] = basic::unbounded_radius<distance_t>();
    // The scan rewinds every thread's arena before it writes anything, so the shares stay empty
    // until then.
    growing_buffer<within_share_t, forkunion_allocator_t> shares {forkunion_allocator_t {}};
    if (failed(shares.reserve_exactly(executor.size()))) return {{}, status_t::bad_alloc_k};
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        shares.push_back_reserved(within_share_t {thread_allocator_t {executor.arena(thread)}});

    auto query_of = [&](std::size_t query_index, thread_allocator_t const& scratch) noexcept {
        line_of_sight_buffers_t sight {scratch};
        observations[query_index] = us::geospatial::to_earth_observation(
            geodetic_of(queries[query_index]), terrain.ground_metres(queries[query_index], sight));
        return expected<value_t> {observations[query_index], status_t::success_k};
    };
    status_t appended = status_t::success_k;
    auto scan_with = [&]<typename predicate_type_>(predicate_type_& admits) noexcept {
        return index.scan_within(
            std::span<distance_t const>(reaches, count), terrain.metric_factory(), query_of,
            [&]<std::size_t extent_>(std::size_t thread, std::size_t query, std::span<key_t const, extent_> keys,
                                     std::span<value_t const, extent_> values,
                                     std::span<distance_t const, extent_> distances) noexcept {
                within_share_t& share = shares[thread];
                std::array<admission_t, extent_> related;
                terrain.ask_relation_values(relation, tolerance, observations[query], values,
                                            std::span<admission_t, extent_>(related), share.sight);
                std::array<key_t, extent_> kept_keys;
                std::array<geospatial_within_match_t, extent_> kept_matches;
                std::size_t kept = 0;
                for (std::size_t lane = 0; lane != extent_; ++lane) {
                    if (related[lane] != admission_t::admit_k) continue;
                    kept_keys[kept] = keys[lane];
                    kept_matches[kept] = {
                        query, std::bit_cast<usearch_key_t>(keys[lane]),
                        std::sqrt(static_cast<f64_t>(distances[lane])) / us::geospatial::centimetres_per_metre_k};
                    ++kept;
                }
                with_static_extent<extent_>(kept, [&]<std::size_t kept_extent_>() noexcept {
                    ask_quad(
                        admits, std::span<key_t const, kept_extent_>(kept_keys.data(), kept_extent_),
                        std::span<geospatial_within_match_t const, kept_extent_>(kept_matches.data(), kept_extent_),
                        [&](geospatial_within_match_t const& match) noexcept {
                            keep_first_failure<atomic_ref>(appended, share.found.push_back(match));
                        });
                });
            },
            executor);
    };
    // Every quad answers before its entries' buffers recycle, so a predicate of either cost is
    // asked a quad at a time.
    dummy_predicate_t no_predicate;
    cheap_predicate_t cheap {};
    if (cheap_predicate_t const* const named = std::get_if<cheap_predicate_t>(&predicate)) cheap = *named;
    if (expensive_predicate_t const* const named = std::get_if<expensive_predicate_t>(&predicate))
        cheap = {named->predicate};
    expected<typename index_t::search_result_t> const scanned = cheap.predicate.callback ? scan_with(cheap)
                                                                                         : scan_with(no_predicate);
    if (failed(scanned.status)) return {{}, scanned.status};
    if (failed(appended)) return {{}, appended};
    [[maybe_unused]] continuation_t const heard = ticks(count, count);

    // Threads land a query's matches in any order, so counting per query places every match, and
    // each query's matches then sort ascending, as the RAM variant writes them.
    std::size_t* const offsets = carve<std::size_t>(staging, count + 1);
    if (!offsets) return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0; query != count + 1; ++query) offsets[query] = 0;
    std::size_t total = 0;
    for (within_share_t const& share : shares)
        for (geospatial_within_match_t const& match : share.found) ++offsets[match.query + 1], ++total;
    for (std::size_t query = 0; query != count; ++query) offsets[query + 1] += offsets[query];
    usearch_key_t* const keys = carve<usearch_key_t>(staging, total);
    usearch_distance_t* const scores = carve<usearch_distance_t>(staging, total);
    growing_buffer<std::size_t, thread_allocator_t> cursors {thread_allocator_t {executor.arena(0)}};
    growing_buffer<geospatial_within_match_t, thread_allocator_t> ordered {thread_allocator_t {executor.arena(0)}};
    if ((total && (!keys || !scores)) || failed(cursors.resize_uninitialized(count)) ||
        failed(ordered.resize_uninitialized(total)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0; query != count; ++query) cursors[query] = offsets[query];
    for (within_share_t const& share : shares)
        for (geospatial_within_match_t const& match : share.found) ordered[cursors[match.query]++] = match;
    auto closer = [](geospatial_within_match_t const& first, geospatial_within_match_t const& second) noexcept {
        return first.score < second.score;
    };
    for (std::size_t query = 0; query != count; ++query) {
        std::span<geospatial_within_match_t> const query_matches = to_writable_span(ordered).subspan(
            offsets[query], offsets[query + 1] - offsets[query]);
        partial_sort(query_matches, query_matches.size(), closer);
    }
    for (std::size_t slot = 0; slot != total; ++slot)
        keys[slot] = ordered[slot].key, scores[slot] = ordered[slot].score;
    outputs.matches = {keys, scores, offsets};
    outputs.written = total;
    traversal_stats_t telemetry;
    telemetry.computed_distances = scanned.value.computed_distances;
    return {telemetry, status_t::success_k};
}

template <cpu_class_t cpu_class_>
expected<traversal_stats_t> geospatial_nvme_variant<cpu_class_>::search(
    geospatial_queries_t query_batch, matches_outputs_t& outputs, search_config_t search, predicate_t const& predicate,
    executor_t& executor, progress_ticks_t& ticks) noexcept {
    usearch_geospatial_observation_t const* const queries = query_batch.observations.data();
    relation_t const relation = static_cast<relation_t>(query_batch.relation);
    usearch_geospatial_tolerance_t const tolerance = query_batch.tolerance;
    std::size_t const count = query_batch.observations.size(), wanted = outputs.wanted;
    auto make_metric = terrain.metric_factory();
    // Each query is built once on the arena it hears, without a thread, so it builds a sight
    // there.
    auto query_of = [&](std::size_t query_index, thread_allocator_t const& scratch) noexcept {
        line_of_sight_buffers_t sight {scratch};
        value_t const query = us::geospatial::to_earth_observation(geodetic_of(queries[query_index]),
                                                                   terrain.ground_metres(queries[query_index], sight));
        return expected<value_t> {query, status_t::success_k};
    };
    geospatial_relations_t relations {*this, relation, tolerance};
    expected<traversal_stats_t> const ranked = rank_nvme_queries(index, count, make_metric, query_of, relations,
                                                                 outputs, search, predicate, executor, ticks);
    if (failed(ranked.status)) return ranked;
    for (std::size_t query = 0; query != count; ++query)
        for (std::size_t match = 0; match != outputs.counts[query]; ++match)
            outputs.scores[query * wanted + match] = std::sqrt(outputs.scores[query * wanted + match]) /
                                                     us::geospatial::centimetres_per_metre_k;
    return ranked;
}

template <cpu_class_t cpu_class_>
expected<sample_result_t> geospatial_nvme_variant<cpu_class_>::sample(basic_sample_inputs_t inputs,
                                                                      basic_sample_outputs_t outputs,
                                                                      sample_config_t config,
                                                                      executor_t& executor) noexcept {
    auto query_of = [](value_t const& stored, thread_allocator_t const&) noexcept {
        return expected<value_t> {stored, status_t::success_k};
    };
    return sample_on_earth(outputs, [&](auto const& coordinates_of) noexcept {
        return sample_entries(*this, terrain.metric_factory(), query_of, inputs, outputs, config, executor,
                              coordinates_of);
    });
}

#if USEARCH_ARCH_CUDA_

/** The fatbin of the geospatial kit and its judging kernel, defined in
 *  @c src/geospatial/vram_variants_nvidia.cu. */
struct geospatial_nvidia_fatbin_t {
    static bytes_view_t bytes() noexcept;
};

/** The capabilities the geospatial kit compiles under on NVIDIA: the Ampere floor, which one cubin
 *  per class carries. */
inline constexpr nk_capability_t geospatial_nvidia_capabilities_k = nk_cap_cuda_k | nk_cap_ampere_k;

/** The face's members reading the kit, defined in @c src/geospatial/vram_variants_nvidia.cu. */
extern template struct us::geospatial::vram_nvidia_index<u32_t, geospatial_nvidia_capabilities_k,
                                                         geospatial_nvidia_fatbin_t>;

/** Builds one shard of the geospatial face on a CUDA device. */
struct geospatial_vram_nvidia_build_t {
    using face_t =
        us::geospatial::vram_nvidia_index<u32_t, geospatial_nvidia_capabilities_k, geospatial_nvidia_fatbin_t>;

    std::unique_ptr<face_t> operator()(int device) const noexcept {
        return std::unique_ptr<face_t> {new (std::nothrow)
                                            face_t(cuda_virtual_memory_allocator<byte_t> {gpu_drivers().cuda, device})};
    }
};

/** The geospatial alternative on one NVIDIA GPU: the basic VRAM shards over observations under the
 *  squared chord with 32-bit node IDs, and the kernel judging each walk's candidates under a
 *  search's relation, lines of sight over a borrowed terrain included. A batch encodes its
 *  observations on the host, as the RAM variants do, standing those without an altitude on the
 *  terrain. The judging kernel and the terrain sit on one device, so a search judges the shards
 *  placed there. */
struct geospatial_vram_nvidia_variant {
    using executor_t = cuda_executor_t;
    using face_t = geospatial_vram_nvidia_build_t::face_t;
    using index_t = basic::vram_nvidia_indexes<face_t, us::utilities::uuid128_t, geospatial_vram_nvidia_build_t, f64_t>;
    using refine_plan_t = index_t::refine_plan_t;
    using value_t = us::geospatial::earth_observation_t;
    using node_id_t = face_t::node_id_t;
    static constexpr gpu_class_t gpu_class_k = gpu_class_t::nvidia_ampere_k;

    index_t index;

    /** Holds no shard until the handle places it on CUDA device @p device, where the judging
     *  kernel and the terrain load. */
    geospatial_vram_nvidia_variant(int device, geospatial_geometry_t const&) noexcept
        : index(forkunion_allocator_t {no_memory_domain_k}, geospatial_vram_nvidia_build_t {}), device(device) {}

    geospatial_geometry_t geometry;

    /** The CUDA device the judging kernel and the terrain load on, the one the shard sits on. */
    int device = 0;

    /** The terrain lines of sight run over, packed on the host, where it stands the observations
     *  recording no altitude on the ground. */
    geospatial_terrain<cpu_class_t::serial_k> host_terrain;

    /** The terrain's heights and its pack on the index's device, where the judging kernel reads
     *  them; empty for none. */
    cuda_unique_ptr<f32_t> device_heights;
    cuda_unique_ptr<byte_t> device_terrain_pack;

    /** The largest stored deviation in whole metres, raised through a reference; it widens every
     *  radius walk. */
    u32_t largest_deviation_metres = 0;

    /** The index's fatbin, loaded once more for the kernel judging the walks' candidates, and that
     *  kernel. */
    cuda_module_t admit_module;
    cuda_function_handle_t admit = nullptr;

    /** Loads the judging kernel onto the index's device. */
    status_t load_admit() noexcept {
        auto [module, status] = cuda_module_t::load(gpu_drivers().cuda, device, geospatial_nvidia_fatbin_t::bytes());
        if (failed(status)) return status;
        admit_module = std::move(module);
        vram_kernel_symbol_t symbol;
        vram_kernel_symbol(symbol, "geospatial", "cuda_admit", {"cuda_admit_arguments", 1},
                           geospatial_nvidia_capabilities_k, itanium_type<node_id_t> {});
        admit = admit_module.function(symbol.text);
        return admit ? status_t::success_k : status_t::device_error_k;
    }

    status_t reserve(reserve_capacity_t capacity) noexcept {
        return vram_nvidia_reserve(index, capacity.entries, geometry.connectivity, capacity.entries * sizeof(value_t));
    }

    /** Adopts @p grid on the host and copies its heights and pack to the index's device, whose
     *  kernels must be loaded; refuses a grid filled by a callback, which no device can call. */
    status_t adopt_terrain(us::geospatial::terrain_t const& grid) noexcept {
        if (grid.callback) return status_t::unsupported_k;
        if (status_t const adopted = host_terrain.adopt_terrain(grid); failed(adopted)) return adopted;
        if (!host_terrain.has_terrain()) return status_t::success_k;
        us::geospatial::terrain_t const& borrowed = host_terrain.grid;
        std::size_t const heights_bytes = (borrowed.rows - 1) * borrowed.heights_stride +
                                          borrowed.columns * sizeof(f32_t);
        std::size_t const pack_bytes = host_terrain.terrain_pack.size();
        cuda_api_t const& api = gpu_drivers().cuda;
        if (cuda_set_device(api, device) != cuda_success_k) return status_t::device_error_k;
        device_heights = cuda_make_unique(cuda_device_allocator<f32_t> {api},
                                          divide_round_up(heights_bytes, sizeof(f32_t)));
        device_terrain_pack = cuda_make_unique(cuda_device_allocator_t {api}, pack_bytes);
        if (!device_heights || !device_terrain_pack) return status_t::bad_alloc_k;
        if (cuda_copy(api, device_heights.get(), borrowed.heights, heights_bytes) != cuda_success_k ||
            cuda_copy(api, device_terrain_pack.get(), host_terrain.terrain_pack.data(), pack_bytes) != cuda_success_k)
            return status_t::device_error_k;
        return status_t::success_k;
    }

    /** Whether a terrain was adopted, which lines of sight need. */
    bool has_terrain() const noexcept { return host_terrain.has_terrain(); }

    /** Bytes the terrain holds: its pack on the host, and its heights and pack on the device. */
    std::size_t terrain_memory_usage() const noexcept {
        return host_terrain.terrain_memory_usage() + device_heights.get_deleter().count * sizeof(f32_t) +
               device_terrain_pack.get_deleter().count;
    }

    expected<traversal_stats_t> insert(usearch_keys_view_t keys, observations_view_t observations,
                                       insert_config_t config, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Walks every query to candidates of the call's breadth, then keeps the first wanted entries
     *  of all of them that the relation and tolerance admit, on the device too. The queries encode
     *  into @p pages, the call arena's host pages, and the matches land in @p outputs there. A
     *  device walk takes no host predicate. */
    expected<traversal_stats_t> search(geospatial_queries_t query_batch, matches_outputs_t& outputs,
                                       search_config_t search, growing_arena<forkunion_allocator_t>& pages,
                                       executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Reports every observation the relation and tolerance admit, doubling the walks' breadth from
     *  the call's until each query's farthest candidate passes the radius the cone graph's spanner
     *  bound widens, every shard counting then writing its admitted candidates on the device,
     *  appended shard by shard into @p outputs on @p pages; an exact scan answers
     *  @c unsupported_k, and a device walk takes no host predicate. */
    expected<traversal_stats_t> within(geospatial_queries_t query_batch, within_outputs_t& outputs,
                                       search_config_t search, predicate_t const& predicate,
                                       growing_arena<forkunion_allocator_t>& pages, executor_t& executor,
                                       progress_ticks_t& ticks) noexcept;

    /** Writes the observations as the RAM variants write them: a lone live shard's monolith, or one
     *  per live shard beside the manifest the path names. */
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept {
        std::array<char, 24> connectivity_text;
        std::array<safegraphs_custom_field_t, 1> const fields {
            {{"connectivity", decimal_of(geometry.connectivity, connectivity_text)}}};
        save_config_t config;
        config.provenance = fields;
        return vram_nvidia_save(index, path, us::geospatial::earth_observation_codec_t {}, config, token_of(geometry),
                                executor, ticks);
    }

    /** Reads a checkpoint any residency wrote, at a path or in host memory, into the placed shards,
     *  one part per shard. */
    template <typename checkpoint_type_>
    status_t load(checkpoint_type_ checkpoint, executor_t& executor, progress_ticks_t& ticks) noexcept {
        if (status_t const loaded = index.load(checkpoint, us::geospatial::earth_observation_codec_t {}, executor,
                                               load_config_t {}, static_cast<connectivity_t>(geometry.connectivity),
                                               ticks);
            failed(loaded))
            return loaded;
        // The observations went straight to the device, so their deviations come back once.
        growing_buffer<value_t, forkunion_allocator_t> stored {forkunion_allocator_t {no_memory_domain_k}};
        for (face_t const& shard : index.live_shards()) {
            if (failed(stored.resize_uninitialized(shard.size()))) return status_t::bad_alloc_k;
            if (stored.empty()) continue;
            if (cuda_copy(shard.api(), stored.data(), shard.storage().values, stored.size() * sizeof(value_t)) !=
                cuda_success_k)
                return status_t::device_error_k;
            for (value_t const& value : stored)
                raise_to<fu::standard_atomic_ref>(largest_deviation_metres, u32_t {value.deviation_metres});
        }
        return status_t::success_k;
    }

    /** Empties the index and its largest deviation, retiring every shard behind a fresh shard
     *  generation on the same devices. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared))
            fu::standard_atomic_ref<u32_t>(largest_deviation_metres).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** Widens the largest deviation to a consumed @p donor 's, whose nodes this index now holds. */
    void count_absorbed_donor(geospatial_vram_nvidia_variant const& donor) noexcept {
        raise_to<fu::standard_atomic_ref>(
            largest_deviation_metres,
            fu::standard_atomic_ref<u32_t const>(donor.largest_deviation_metres).load(std::memory_order_relaxed));
    }

    /** Bounds the deviation of the part @p receiver now holds by this index's, which held it. */
    void count_split_receiver(geospatial_vram_nvidia_variant& receiver) const noexcept {
        receiver.count_absorbed_donor(*this);
    }

    /** Sampling answers @c unsupported_k until the shards sample on their devices. */
    expected<sample_result_t> sample(basic_sample_inputs_t, basic_sample_outputs_t, sample_config_t,
                                     executor_t&) noexcept {
        return {{}, status_t::unsupported_k};
    }

  private:

    /** The judging arguments every walk of @p query_batch shares on @p face: its storage, the
     *  relation and its tolerance, and the terrain lines of sight clear, if any. */
    us::geospatial::cuda_admit_arguments<node_id_t> admitting_(face_t const& face,
                                                               geospatial_queries_t const& query_batch) const noexcept {
        us::geospatial::cuda_admit_arguments<node_id_t> admitting;
        admitting.storage = face.storage();
        admitting.relation = static_cast<us::geospatial::relation_t>(query_batch.relation);
        admitting.radius_metres = query_batch.tolerance.metres;
        admitting.slack_milliseconds = query_batch.tolerance.seconds * 1000.0;
        if (device_heights) {
            us::geospatial::terrain_t const& borrowed = host_terrain.grid;
            admitting.terrain_heights = device_heights.get();
            admitting.terrain_pack = device_terrain_pack.get();
            admitting.north_latitude_degrees = borrowed.north_latitude_degrees;
            admitting.west_longitude_degrees = borrowed.west_longitude_degrees;
            admitting.row_step_degrees = borrowed.row_step_degrees;
            admitting.column_step_degrees = borrowed.column_step_degrees;
            admitting.drop_scale = (1.0 - us::geospatial::terrain_refraction_k) / (2.0 * NUMKONG_EARTH_MEAN_RADIUS);
        }
        return admitting;
    }

    /** The @p count observations encoded on @p pages as staged entries, standing those without an
     *  altitude on the terrain. */
    expected<tape_view<byte_t const>> encode_(usearch_geospatial_observation_t const* observations, std::size_t count,
                                              growing_arena<forkunion_allocator_t>& pages) const noexcept {
        value_t* const values = carve<value_t>(pages, count);
        std::size_t* const offsets = carve<std::size_t>(pages, count + 1);
        if (!values || !offsets) return {{}, status_t::bad_alloc_k};
        // A borrowed grid never grows the sight's buffers, which serve filled grids alone.
        line_of_sight_buffers_t sight {thread_allocator_t {}};
        for (std::size_t offset = 0; offset != count; ++offset) {
            values[offset] = us::geospatial::to_earth_observation(
                geodetic_of(observations[offset]), host_terrain.ground_metres(observations[offset], sight));
            offsets[offset] = offset * sizeof(value_t);
        }
        offsets[count] = count * sizeof(value_t);
        return {{reinterpret_cast<byte_t const*>(values), {offsets, count + 1}}, status_t::success_k};
    }
};

inline expected<traversal_stats_t> geospatial_vram_nvidia_variant::insert(usearch_keys_view_t keys,
                                                                          observations_view_t observations,
                                                                          insert_config_t config, executor_t& executor,
                                                                          progress_ticks_t& ticks) noexcept {
    std::size_t const count = observations.size();
    // The batch's own pages, since batches routed to disjoint shards may land at once.
    growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};
    auto const [entries, encode_status] = encode_(observations.data(), count, staging);
    if (failed(encode_status)) return {{}, encode_status};
    expected<traversal_stats_t> const inserted = vram_nvidia_insert(index, keys, entries, geometry.connectivity,
                                                                    config.expansion_factor, ticks, executor);
    if (failed(inserted.status)) return inserted;
    value_t const* const values = reinterpret_cast<value_t const*>(entries.elements);
    for (std::size_t entry = 0; entry != count; ++entry)
        raise_to<fu::standard_atomic_ref>(largest_deviation_metres, u32_t {values[entry].deviation_metres});
    return inserted;
}

inline expected<traversal_stats_t> geospatial_vram_nvidia_variant::search(
    geospatial_queries_t query_batch, matches_outputs_t& outputs, search_config_t search,
    growing_arena<forkunion_allocator_t>& pages, executor_t& executor, progress_ticks_t& ticks) noexcept {
    std::size_t const count = query_batch.observations.size(), wanted = outputs.wanted,
                      expansion = search.expansion_factor;
    if (!wanted) {
        for (std::size_t query = 0; query != count; ++query) outputs.counts[query] = 0;
        return {{}, status_t::success_k};
    }
    auto const [queries, encode_status] = encode_(query_batch.observations.data(), count, pages);
    if (failed(encode_status)) return {{}, encode_status};
    // Candidates fewer than the ranking they fill would short every query.
    std::size_t const candidates_count = max_of(expansion ? expansion : default_expansion_factor(geometry.connectivity),
                                                wanted);
    face_t::search_config_t const config {static_cast<u32_t>(candidates_count), static_cast<u32_t>(candidates_count)};

    // Past the walk's own scratch: the queries, the judged metres and counts, the candidates and
    // the judged IDs, each column its own aligned carve.
    std::size_t const listed_count = count * candidates_count, judged = count * wanted;
    auto const aligned = [](std::size_t bytes) noexcept { return round_up_to_multiple(bytes, default_alignment_k); };
    std::size_t const judging_bytes = aligned(count * sizeof(value_t)) + aligned(judged * sizeof(f64_t)) +
                                      aligned(count * sizeof(u64_t)) + aligned(listed_count * sizeof(node_id_t)) +
                                      aligned(listed_count * sizeof(f32_t)) + aligned(judged * sizeof(node_id_t));
    // Each shard judges its own candidates on its stream into the sharded index's outputs, which
    // merges them by distance.
    auto judge = [&](face_t& face, nk::matrix_span<node_id_t> host_ids, nk::matrix_span<f64_t> host_metres,
                     std::span<std::size_t> host_counts, face_t::traversal_context_t& context,
                     std::size_t) noexcept -> status_t {
        // The judging kernel and the terrain load on one device, which every shard it judges
        // shares.
        if (face.device_specs().ordinal != device) return status_t::unsupported_k;
        face_t::traversal_context_t::allocator_t const allocator = context.allocator();
        value_t* const device_queries = cuda_arena_allocator<value_t>(allocator).allocate(count);
        f64_t* const matched_metres = cuda_arena_allocator<f64_t>(allocator).allocate(judged);
        u64_t* const matched_counts = cuda_arena_allocator<u64_t>(allocator).allocate(count);
        node_id_t* const candidate_ids = cuda_arena_allocator<node_id_t>(allocator).allocate(listed_count);
        f32_t* const candidate_distances = cuda_arena_allocator<f32_t>(allocator).allocate(listed_count);
        node_id_t* const matched_ids = cuda_arena_allocator<node_id_t>(allocator).allocate(judged);
        if (!device_queries || !matched_metres || !matched_counts || !candidate_ids || !candidate_distances ||
            !matched_ids)
            return status_t::bad_alloc_k;
        us::geospatial::cuda_admit_arguments<node_id_t> admitting = admitting_(face, query_batch);
        admitting.queries = device_queries;
        admitting.candidate_ids = candidate_ids;
        admitting.matched_ids = matched_ids;
        admitting.matched_metres = matched_metres;
        admitting.matched_counts = matched_counts;
        admitting.queries_count = static_cast<u32_t>(count);
        admitting.candidates_count = static_cast<u32_t>(candidates_count);
        admitting.wanted = static_cast<u32_t>(wanted);
        cuda_api_t const& api = admit_module.api();
        if (cuda_copy_async(api, device_queries, queries.elements, count * sizeof(value_t), context.stream()) !=
            cuda_success_k)
            return status_t::device_error_k;
        if (status_t const searched = face.search(queries, {candidate_ids, {count, candidates_count}},
                                                  {candidate_distances, {count, candidates_count}}, config, context)
                                          .status;
            failed(searched))
            return searched;
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(count, vram_warps_per_block_k),
                                std::size_t {face.device_specs().multiprocessor_count} * 8));
        if (failed(cuda_launch(api, admit, blocks, vram_warps_per_block_k * 32, 0, context.stream(),
                               cuda_overlap_t::serialized_k, admitting)) ||
            cuda_copy_async(api, host_ids.data(), matched_ids, judged * sizeof(node_id_t), context.stream()) !=
                cuda_success_k ||
            cuda_copy_async(api, host_metres.data(), matched_metres, judged * sizeof(f64_t), context.stream()) !=
                cuda_success_k ||
            cuda_copy_async(api, host_counts.data(), matched_counts, count * sizeof(u64_t), context.stream()) !=
                cuda_success_k)
            return status_t::device_error_k;
        return context.wait();
    };
    auto const [found, status] = index.search_every_shard(
        count, executor, judge,
        nk::matrix_span<us::utilities::uuid128_t> {reinterpret_cast<us::utilities::uuid128_t*>(outputs.keys), count,
                                                   wanted},
        nk::matrix_span<f64_t> {outputs.scores, count, wanted}, std::span<std::size_t> {outputs.counts, count}, queries,
        config, judging_bytes);
    if (failed(status)) return {{}, status};
    [[maybe_unused]] continuation_t const heard = ticks(count - 1);
    return {{.computed_distances = found.computed_distances,
             .visited_members = found.visited_members,
             .restarts = found.restarts},
            status_t::success_k};
}

inline expected<traversal_stats_t> geospatial_vram_nvidia_variant::within(
    geospatial_queries_t query_batch, within_outputs_t& outputs, search_config_t search, predicate_t const&,
    growing_arena<forkunion_allocator_t>& pages, executor_t& executor, progress_ticks_t& ticks) noexcept {
    std::size_t const count = query_batch.observations.size(), shards = index.live_shards_count();
    if (search.traversal == traversal_t::scan_k) return {{}, status_t::unsupported_k};
    std::size_t* const offsets = carve<std::size_t>(pages, count + 1);
    if (!offsets) return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0; query != count + 1; ++query) offsets[query] = 0;
    outputs = {{nullptr, nullptr, offsets}, 0};
    if (!count || !shards) return {{}, status_t::success_k};
    auto const [queries, encode_status] = encode_(query_batch.observations.data(), count, pages);
    if (failed(encode_status)) return {{}, encode_status};
    // The squared chord each query's candidates must pass, the walk's reach as the RAM variants
    // take it, widened by the cone graph's spanner bound.
    f64_t* const covering_chords = carve<f64_t>(pages, count);
    if (!covering_chords) return {{}, status_t::bad_alloc_k};
    value_t const* const encoded = reinterpret_cast<value_t const*>(queries.elements);
    f64_t const largest_deviation =
        fu::standard_atomic_ref<u32_t>(largest_deviation_metres).load(std::memory_order_relaxed);
    for (std::size_t query = 0; query != count; ++query) {
        f64_t const query_deviation = encoded[query].deviation_metres;
        f64_t const deviations = std::sqrt(query_deviation * query_deviation + largest_deviation * largest_deviation);
        f64_t const reach = std::ceil((query_batch.tolerance.metres + 3 * deviations) *
                                      us::geospatial::centimetres_per_metre_k);
        f64_t const widened = reach * (1.0 + 2.0 * us::geospatial::cone_graph_stretch_k);
        covering_chords[query] = widened * widened;
    }
    // The fan-out's own rankings stay one entry wide and empty, since every shard keeps its run.
    us::utilities::uuid128_t* const unused_keys = carve<us::utilities::uuid128_t>(pages, count);
    f64_t* const unused_scores = carve<f64_t>(pages, count);
    std::size_t* const unused_counts = carve<std::size_t>(pages, count);
    if (!unused_keys || !unused_scores || !unused_counts) return {{}, status_t::bad_alloc_k};
    auto const aligned = [](std::size_t bytes) noexcept { return round_up_to_multiple(bytes, default_alignment_k); };
    std::size_t const published = index.size();
    std::size_t breadth = search.expansion_factor ? search.expansion_factor
                                                  : default_expansion_factor(geometry.connectivity);
    std::size_t listed_count = 0, walked = 0;
    u64_t* runs_offsets = nullptr;
    node_id_t* runs_ids = nullptr;
    f64_t* runs_metres = nullptr;
    face_t const** runs_faces = nullptr;
    traversal_stats_t stats;
    for (;; breadth = min_of(breadth * 2, published)) {
        face_t::search_config_t const config {static_cast<u32_t>(breadth), static_cast<u32_t>(breadth)};
        // Each shard's run comes back whole: its offsets from zero, then the IDs and metres they
        // span, at most a breadth per query; the shard is remembered to resolve its IDs into keys.
        listed_count = count * breadth;
        walked = 0;
        runs_offsets = carve<u64_t>(pages, shards * (count + 1));
        runs_ids = carve<node_id_t>(pages, shards * listed_count);
        runs_metres = carve<f64_t>(pages, shards * listed_count);
        runs_faces = carve<face_t const*>(pages, shards);
        f32_t* const runs_distances = carve<f32_t>(pages, shards * listed_count);
        if (!runs_offsets || !runs_ids || !runs_metres || !runs_faces || !runs_distances)
            return {{}, status_t::bad_alloc_k};
        std::size_t const judging_bytes =
            aligned(count * sizeof(value_t)) + aligned(count * sizeof(u64_t)) + aligned((count + 1) * sizeof(u64_t)) +
            aligned(listed_count * sizeof(node_id_t)) + aligned(listed_count * sizeof(f32_t)) +
            aligned(listed_count * sizeof(node_id_t)) + aligned(listed_count * sizeof(f64_t));
        auto judge = [&](face_t& face, nk::matrix_span<node_id_t>, nk::matrix_span<f64_t>,
                         std::span<std::size_t> host_counts, face_t::traversal_context_t& context,
                         std::size_t) noexcept -> status_t {
            for (std::size_t& counted : host_counts) counted = 0;
            if (face.device_specs().ordinal != device) return status_t::unsupported_k;
            std::size_t const run = walked++;
            runs_faces[run] = &face;
            u64_t* const run_offsets = runs_offsets + run * (count + 1);
            face_t::traversal_context_t::allocator_t const allocator = context.allocator();
            value_t* const device_queries = cuda_arena_allocator<value_t>(allocator).allocate(count);
            u64_t* const matched_counts = cuda_arena_allocator<u64_t>(allocator).allocate(count);
            u64_t* const matched_offsets = cuda_arena_allocator<u64_t>(allocator).allocate(count + 1);
            node_id_t* const candidate_ids = cuda_arena_allocator<node_id_t>(allocator).allocate(listed_count);
            f32_t* const candidate_distances = cuda_arena_allocator<f32_t>(allocator).allocate(listed_count);
            node_id_t* const matched_ids = cuda_arena_allocator<node_id_t>(allocator).allocate(listed_count);
            f64_t* const matched_metres = cuda_arena_allocator<f64_t>(allocator).allocate(listed_count);
            if (!device_queries || !matched_counts || !matched_offsets || !candidate_ids || !candidate_distances ||
                !matched_ids || !matched_metres)
                return status_t::bad_alloc_k;
            us::geospatial::cuda_admit_arguments<node_id_t> admitting = admitting_(face, query_batch);
            admitting.queries = device_queries;
            admitting.candidate_ids = candidate_ids;
            admitting.queries_count = static_cast<u32_t>(count);
            admitting.candidates_count = static_cast<u32_t>(breadth);
            admitting.matched_ids = matched_ids;
            admitting.matched_metres = matched_metres;
            admitting.matched_counts = matched_counts;
            admitting.matched_offsets = matched_offsets;
            cuda_api_t const& api = face.api();
            run_offsets[0] = 0;
            if (cuda_copy_async(api, device_queries, queries.elements, count * sizeof(value_t), context.stream()) !=
                    cuda_success_k ||
                cuda_copy_async(api, matched_offsets, run_offsets, sizeof(u64_t), context.stream()) != cuda_success_k)
                return status_t::device_error_k;
            if (status_t const searched = face.search(queries, {candidate_ids, {count, breadth}},
                                                      {candidate_distances, {count, breadth}}, config, context)
                                              .status;
                failed(searched))
                return searched;
            if (status_t const counted = face.launch_within_count_(admitting, context); failed(counted)) return counted;
            if (status_t const written = face.launch_within_write_(admitting, context); failed(written)) return written;
            if (cuda_copy_async(api, run_offsets, matched_offsets, (count + 1) * sizeof(u64_t), context.stream()) !=
                    cuda_success_k ||
                cuda_copy_async(api, runs_distances + run * listed_count, candidate_distances,
                                listed_count * sizeof(f32_t), context.stream()) != cuda_success_k)
                return status_t::device_error_k;
            if (status_t const waited = context.wait(); failed(waited)) return waited;
            std::size_t const admitted = static_cast<std::size_t>(run_offsets[count]);
            if (cuda_copy_async(api, runs_ids + run * listed_count, matched_ids, admitted * sizeof(node_id_t),
                                context.stream()) != cuda_success_k ||
                cuda_copy_async(api, runs_metres + run * listed_count, matched_metres, admitted * sizeof(f64_t),
                                context.stream()) != cuda_success_k)
                return status_t::device_error_k;
            return context.wait();
        };
        auto const [found, status] = index.search_every_shard(
            count, executor, judge, nk::matrix_span<us::utilities::uuid128_t> {unused_keys, count, 1},
            nk::matrix_span<f64_t> {unused_scores, count, 1}, std::span<std::size_t> {unused_counts, count}, queries,
            config, judging_bytes);
        if (failed(status)) return {{}, status};
        stats.computed_distances += found.computed_distances;
        stats.visited_members += found.visited_members;
        stats.restarts += found.restarts;
        // A candidate past the walk's end ranks infinitely far, as nothing else can be reached.
        bool covered = true;
        for (std::size_t run = 0; run != walked; ++run)
            for (std::size_t query = 0; query != count; ++query)
                covered = covered &&
                          runs_distances[run * listed_count + query * breadth + breadth - 1] > covering_chords[query];
        if (covered || breadth >= published) break;
    }

    // Each query's matches append shard by shard, as the RAM variants append each shard's ball.
    for (std::size_t run = 0; run != walked; ++run)
        for (std::size_t query = 0; query != count; ++query)
            offsets[query + 1] += static_cast<std::size_t>(runs_offsets[run * (count + 1) + query + 1] -
                                                           runs_offsets[run * (count + 1) + query]);
    for (std::size_t query = 0; query != count; ++query) offsets[query + 1] += offsets[query];
    std::size_t const total = offsets[count];
    usearch_key_t* const keys = carve<usearch_key_t>(pages, total);
    usearch_distance_t* const scores = carve<usearch_distance_t>(pages, total);
    if (total && (!keys || !scores)) return {{}, status_t::bad_alloc_k};
    for (std::size_t query = 0, slot = 0; query != count; ++query)
        for (std::size_t run = 0; run != walked; ++run) {
            u64_t const* const run_offsets = runs_offsets + run * (count + 1);
            for (u64_t match = run_offsets[query]; match != run_offsets[query + 1]; ++match, ++slot) {
                std::size_t const position = run * listed_count + static_cast<std::size_t>(match);
                keys[slot] = std::bit_cast<usearch_key_t>(runs_faces[run]->key_of(runs_ids[position]).value);
                scores[slot] = runs_metres[position];
            }
        }
    outputs = {{keys, scores, offsets}, total};
    [[maybe_unused]] continuation_t const heard = ticks(count - 1);
    return {stats, status_t::success_k};
}

#endif

#pragma endregion Geospatial Variants

#pragma region Geospatial Bodies

/** The geospatial variants each class unit compiles: one per CPU class on the host and on storage,
 *  and one on NVIDIA. */
template <auto class_>
struct geospatial_variants_of<vendored_residency_t::ram_k, class_> {
    using type = type_list<geospatial_ram_variant<class_>>;
};
template <auto class_>
struct geospatial_variants_of<vendored_residency_t::nvme_k, class_> {
    using type = type_list<geospatial_nvme_variant<class_>>;
};
#if USEARCH_ARCH_CUDA_
template <>
struct geospatial_variants_of<vendored_residency_t::vram_nvidia_k, gpu_class_t::nvidia_ampere_k> {
    using type = type_list<geospatial_vram_nvidia_variant>;
};
#endif

/** The engine's form of a C terrain; a path is refused before it gets here. */
inline us::geospatial::terrain_t terrain_of(usearch_geospatial_terrain_t const& terrain) noexcept {
    return {terrain.rows,
            terrain.columns,
            terrain.north_latitude_degrees,
            terrain.west_longitude_degrees,
            terrain.row_step_degrees,
            terrain.column_step_degrees,
            terrain.heights,
            terrain.heights_stride,
            terrain.callback,
            terrain.state};
}

/** Whether @p variant serves @p relation, which the traits already know: lines of sight need its
 *  terrain, so @c visible without one answers @c unsupported_k. */
template <typename variant_type_>
status_t serves_relation(variant_type_ const& variant, usearch_geospatial_relation_t relation) noexcept {
    if (relation != usearch_geospatial_relation_visible_k || variant.has_terrain()) return status_t::success_k;
    return status_t::unsupported_k;
}

/** What the geospatial verbs do to one variant: adopt its terrain, describe it, walk a radius, and
 *  land its observations. */
template <>
struct variant_bodies<geospatial_traits_t> {
    using geometry_t = geospatial_geometry_t;
    using geometry_extras_t = geospatial_traits_t::geometry_extras_t;

    /** What a view keeps of an observation it read: the stored form, decoded as it reaches the
     *  view. */
    using kept_t = us::geospatial::earth_observation_t;

    /** Whether @p donor enters @p variant: the same connectivity, since every observation is
     *  scored alike. */
    template <typename variant_type_, typename donor_type_>
    static bool matches(variant_type_ const& variant, donor_type_ const& donor) noexcept {
        return variant.geometry.connectivity == donor.geometry.connectivity;
    }

    /** The connectivity @p geometry freezes, a zero taking the cone winners and long links of the
     *  default. */
    static std::size_t connectivity_of(geometry_t const& geometry) noexcept {
        return geometry.connectivity ? geometry.connectivity : us::geospatial::surface_keep_octants_t {}.connectivity;
    }

    /** Takes @p geometry into the CPU @p variant, which adopts @p terrain. */
    template <typename variant_type_>
    static status_t adopt(variant_type_& variant, geometry_t const& geometry,
                          geometry_extras_t const& terrain) noexcept {
        variant.geometry.connectivity = connectivity_of(geometry);
        return variant.adopt_terrain(terrain_of(terrain));
    }

#if USEARCH_ARCH_CUDA_

    /** Takes @p geometry into the GPU @p variant, its connectivity rounded up to the power of two a
     *  GPU neighbor row takes, loads its judging kernel, and copies the terrain to its device. */
    static status_t adopt(geospatial_vram_nvidia_variant& variant, geometry_t const& geometry,
                          geometry_extras_t const& terrain) noexcept {
        variant.geometry.connectivity = std::bit_ceil(connectivity_of(geometry));
        if (status_t const loaded = variant.load_admit(); failed(loaded)) return loaded;
        return variant.adopt_terrain(terrain_of(terrain));
    }
#endif

    template <typename variant_type_>
    static void describe(variant_type_ const& variant, usearch_geospatial_index_description_t& description) noexcept {
        description.metadata = geospatial_traits_t::metadata_of(variant.geometry);
        description.memory_usage += variant.terrain_memory_usage();
    }

    template <typename variant_type_>
    static status_t serves_search(variant_type_ const& variant, geospatial_queries_t const& queries) noexcept {
        return serves_relation(variant, queries.relation);
    }

    static status_t keep(nvme_executor_t&, std::size_t, kept_t const& stored, kept_t& kept) noexcept {
        kept = stored;
        return status_t::success_k;
    }

    /** A device's stored observation, decoded by the codec its checkpoints write. */
    static status_t keep_stored(bytes_view_t stored, kept_t& kept) noexcept {
        return us::geospatial::earth_observation_codec_t {}.load(stored, kept);
    }

    /** Whether @p variant walks the radius @p queries ask: a relation it serves, and no host
     *  predicate on a device. */
    template <typename variant_type_>
    static status_t serves_within(variant_type_ const& variant, geospatial_queries_t const& queries,
                                  predicate_t const& predicate) noexcept {
        if (!std::holds_alternative<no_predicate_t>(predicate) && arena_residency_of(variant) != ram_residency_k)
            return status_t::unsupported_k;
        return serves_relation(variant, queries.relation);
    }

    /** Writes @p count observations, the @c i th as @p value_at hands it, in @p memory. */
    template <typename variant_type_, typename value_at_type_>
    static status_t write_values(variant_type_ const&, std::size_t count, value_at_type_&& value_at,
                                 growing_arena<forkunion_allocator_t>& memory,
                                 usearch_geospatial_observation_t const** outputs) noexcept {
        usearch_geospatial_observation_t* const observations = carve<usearch_geospatial_observation_t>(memory, count);
        if (count && !observations) return status_t::bad_alloc_k;
        for (std::size_t entry = 0; entry != count; ++entry)
            observations[entry] = observation_of(us::geospatial::to_geodetic_observation(value_at(entry)));
        *outputs = observations;
        return status_t::success_k;
    }
};

#pragma endregion Geospatial Bodies
