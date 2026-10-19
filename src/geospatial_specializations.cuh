/**
 *  @file src/geospatial_specializations.cuh
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief What one geospatial variant is spelled with and compiled through: the crossings of an
 *      observation, the geometry every geospatial variant freezes, the terrain and the judging of
 *      relations, and the variants over the basic indexes.
 *
 *  Every geospatial class unit in @c src/geospatial/ includes this to instantiate its variants, and
 *  @c geospatial_index.cpp adds the variant, the traits and the exported C entry points on top of
 *  it.
 */
#pragma once

#include <cmath> // `std::sqrt`, `std::floor`, `std::ceil`, `std::fmin`, `std::fmax`

#include <array>       // `std::array` - a quad of stored observations, a sight line's points
#include <bit>         // `std::bit_cast` - the C key struct & `us::utilities::uuid_t` share sixteen bytes
#include <mutex>       // `std::lock_guard` - one batch lands its keys and rows at a time on a GPU
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

/** One containment match before it lands: the query it answers, the entry's key and its chord in
 *  metres. */
struct geospatial_within_match_t {
    std::size_t query;
    usearch_key_t key;
    usearch_distance_t score;
};

/** The connectivity a geospatial handle freezes, as an init or a checkpoint states it; every
 *  variant holds one. */
struct geospatial_geometry_t {

    /** Neighbors per node: a winner per cone and long links for the rest, rounded up to a power of
     *  two on a GPU; zero until frozen, which takes the default. */
    std::size_t connectivity = 0;
};

/** The checkpoint token of the one geospatial kind, the squared chord. */
inline std::string_view token_of(geospatial_geometry_t const&) noexcept { return "chord"; }

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
    us::geospatial::terrain_t terrain;

    /** The maxima of a borrowed grid, packed once. */
    growing_buffer<byte_t, forkunion_allocator_t> terrain_pack {forkunion_allocator_t {}};

    /** Adopts @p grid, packing it when it is borrowed whole; refuses a grid with no extent or
     *  steps. */
    status_t adopt_terrain(us::geospatial::terrain_t const& grid) noexcept;

    /** Whether a terrain was adopted, which lines of sight need. */
    bool has_terrain() const noexcept { return !terrain.empty(); }

    /** The bytes the packed terrain holds. */
    std::size_t terrain_memory_usage() const noexcept { return terrain_pack.memory_usage(); }

  protected:

    /** A factory of the variant's metric over a thread's arena, sized for a round of candidates. */
    static auto metric_factory_() noexcept {
        return [](thread_allocator_t const& allocator, std::size_t staging) noexcept {
            metric_t made {allocator};
            status_t const ready = made.init(staging);
            return expected<metric_t> {std::move(made), ready};
        };
    }

    /** The terrain's height under @p observation when it records no altitude, else zero; the
     *  ellipsoid's own zero off the grid or beside a missing cell. */
    f64_t ground_metres_(usearch_geospatial_observation_t const& observation,
                         thread_allocator_t const& scratch) const noexcept;

    /** Keeps the verdicts of @p entries whose sight lines over the terrain clear it under
     *  @p relation, rejecting the rest; a pair off the grid counts as clear. */
    void clear_sight_(relation_t relation, value_t const& query, std::span<value_t const> entries,
                      std::span<admission_t> verdicts, thread_allocator_t const& scratch) const noexcept;

    /** Judges up to four stored @p entries against @p query into @p verdicts: the relation and
     *  time, then the terrain. */
    void judge_values_(relation_t relation, usearch_geospatial_tolerance_t tolerance, value_t const& query,
                       std::span<value_t const> entries, std::span<admission_t> verdicts,
                       thread_allocator_t const& scratch) const noexcept;
};

/** One geospatial variant: the basic index over observations, scored with one class's kit, with the
 *  terrain its lines of sight run over. */
template <cpu_class_t cpu_class_>
struct geospatial_ram_variant : geospatial_terrain<cpu_class_> {
    using executor_t = forkunion_executor_t;
    using kit_t = typename us::geospatial::geospatial_ladder<numkong_capabilities_of(cpu_class_)>::kit_t;
    using index_t =
        us::geospatial::ram_index<us::utilities::uuid_t, dummy_waiter_t, standard_atomic_ref, forkunion_allocator_t>;
    using value_t = us::geospatial::earth_observation_t;
    using node_id_t = typename index_t::node_id_t;
    using metric_t = us::geospatial::chord_metric<kit_t, thread_allocator_t>;
    using relation_t = us::geospatial::relation_t;
    using refine_plan_t = typename index_t::refine_plan_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;

    index_t index;
    geospatial_geometry_t geometry;

    /** The largest stored deviation in whole metres, raised through @c atomic_ref; it widens every
     *  radius walk. */
    u32_t largest_deviation_metres = 0;

    explicit geospatial_ram_variant(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Reserves the entries @p arguments name. */
    status_t reserve(usearch_geospatial_reserve_t const& arguments) noexcept {
        return index.reserve(arguments.entries).status;
    }

    /** Empties the index and forgets its largest deviation. */
    status_t clear() noexcept {
        status_t const cleared = index.clear();
        if (succeeded(cleared))
            standard_atomic_ref<u32_t>(largest_deviation_metres).store(0, std::memory_order_relaxed);
        return cleared;
    }

    /** Long links every insertion draws, appended past the selector's budget. */
    std::size_t long_links() const noexcept {
        constexpr std::size_t cones_k = us::geospatial::surface_keep_octants_t::cones_k;
        return geometry.connectivity > cones_k ? geometry.connectivity - cones_k : 0;
    }

    /** The widest row a walk copies, which sizes every round's staging. */
    std::size_t widest_row() const noexcept { return geometry.connectivity + long_links(); }

    expected<traversal_telemetry_t> insert(usearch_geospatial_insert_t const& arguments, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;
    expected<traversal_telemetry_t> search(usearch_geospatial_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** Opens a refinement once @p donors land, its walks as wide as @p expansion or the 128 exact
     *  cone winners need, as an insertion's; the cone selector keeps the connectivity, and the long
     *  links past it give way. */
    expected<refine_plan_t> plan_refine(std::span<index_t*> donors, std::size_t expansion) noexcept {
        typename index_t::traversal_context_t planning {index.size(), index.get_allocator()};
        std::size_t const breadth = expansion ? expansion : us::geospatial::surface_keep_octants_t::exact_candidates_k;
        return index.plan_absorb(donors, planning, us::geospatial::surface_keep_octants_t {geometry.connectivity},
                                 {breadth});
    }

    /** Drives @p plan, then draws every node's long links again by the insertion's rule and seeds,
     *  since the re-selection keeps only the cone selector's budget. */
    status_t refine(refine_plan_t& plan, refine_cursor_t& cursor, progress_ticks_t& ticks,
                    executor_t& executor) noexcept;

    /** Widens the largest deviation to a landed @p donor 's, whose rows this index now holds. */
    void absorbed(geospatial_ram_variant const& donor) noexcept {
        raise_to(largest_deviation_metres,
                 standard_atomic_ref<u32_t const>(donor.largest_deviation_metres).load(std::memory_order_relaxed));
    }

    expected<traversal_telemetry_t> within(usearch_geospatial_observation_t const* queries, std::size_t count,
                                           relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                           usearch_predicate_t predicate, growing_arena<forkunion_allocator_t>& staging,
                                           usearch_matches_tape_view_t& matches, std::size_t& written,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;
    status_t save(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;
    status_t load(std::string_view path, executor_t& executor, progress_ticks_t& ticks) noexcept;

    /** Serving a checkpoint in place and sampling answer @c unsupported_k until the index serves
     *  them. */
    status_t view(std::string_view) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_geospatial_sample_t const&) noexcept { return status_t::unsupported_k; }

  private:

    /** Draws the long links of every published node in @p nodes, each from a generator keyed by
     *  its ID, so a redraw over a re-selected row lands where the insertion's draws landed. */
    status_t draw_long_links_(iota_view<node_id_t> nodes, thread_scratch<index_t, metric_t>& scratch,
                              executor_t& executor) noexcept;

    /** Judges the candidates' quad from @p first of @p context against @p query: the relation, time
     *  and terrain, then the predicate. */
    template <typename context_type_>
    void judge_(relation_t relation, usearch_geospatial_tolerance_t tolerance, value_t const& query,
                context_type_ const& context, std::size_t first, std::size_t quad, usearch_predicate_t predicate,
                std::array<us::utilities::uuid_t, 4>& identities, std::array<admission_t, 4>& verdicts,
                thread_allocator_t const& scratch) const noexcept;
};

template <cpu_class_t cpu_class_>
status_t geospatial_terrain<cpu_class_>::adopt_terrain(us::geospatial::terrain_t const& grid) noexcept {
    if (grid.empty()) return status_t::success_k;
    if (!(grid.row_step_degrees > 0) || !(grid.column_step_degrees > 0)) return status_t::config_mismatch_k;
    terrain = grid;
    if (!terrain.heights_stride) terrain.heights_stride = terrain.columns * sizeof(f32_t);
    if (!terrain.heights) return status_t::success_k;
    nk_size_t bytes = 0;
    if (nk_status_t const sized = kit_t::terrain_pack_size(terrain.rows, terrain.columns, &bytes);
        sized != nk_success_k)
        return us::shapes::status_of(sized);
    if (status_t const reserved = terrain_pack.resize_uninitialized(bytes); failed(reserved)) return reserved;
    std::array<f64_t, 2> const steps = terrain.step_metres();
    return us::shapes::status_of(kit_t::terrain_pack(terrain.heights, terrain.rows, terrain.columns,
                                                     terrain.heights_stride, steps[0], steps[1], terrain_pack.data(),
                                                     nullptr));
}

template <cpu_class_t cpu_class_>
f64_t geospatial_terrain<cpu_class_>::ground_metres_(usearch_geospatial_observation_t const& observation,
                                                     thread_allocator_t const& scratch) const noexcept {
    if (terrain.empty() || observation.altitude_metres == observation.altitude_metres) return 0;
    std::array<f64_t, 2> const cell = terrain.grid_point(observation.latitude_degrees, observation.longitude_degrees);
    f64_t const last_row = static_cast<f64_t>(terrain.rows - 1), last_column = static_cast<f64_t>(terrain.columns - 1);
    if (!(cell[0] >= 0 && cell[0] <= last_column && cell[1] >= 0 && cell[1] <= last_row)) return 0;
    f32_t height = 0;
    if (terrain.heights) {
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0]), static_cast<f32_t>(cell[1])};
        if (kit_t::terrain_heights(terrain.heights, terrain_pack.data(), point.data(), 1, &height, nullptr) !=
            nk_success_k)
            return 0;
    }
    else {
        // A filled grid answers for the two-by-two window around the point, packed on its own.
        std::size_t const first_column = min_of(static_cast<std::size_t>(cell[0]),
                                                terrain.columns > 1 ? terrain.columns - 2 : 0);
        std::size_t const first_row = min_of(static_cast<std::size_t>(cell[1]),
                                             terrain.rows > 1 ? terrain.rows - 2 : 0);
        std::size_t const window_rows = min_of<std::size_t>(2, terrain.rows),
                          window_columns = min_of<std::size_t>(2, terrain.columns);
        std::array<f32_t, 4> window {};
        terrain.callback(first_row, first_column, window_rows, window_columns, window.data(),
                         window_columns * sizeof(f32_t), terrain.state);
        nk_size_t bytes = 0;
        if (kit_t::terrain_pack_size(window_rows, window_columns, &bytes) != nk_success_k) return 0;
        growing_buffer<byte_t, thread_allocator_t> pack {scratch};
        if (failed(pack.resize_uninitialized(bytes))) return 0;
        std::array<f64_t, 2> const steps = terrain.step_metres();
        if (kit_t::terrain_pack(window.data(), window_rows, window_columns, window_columns * sizeof(f32_t), steps[0],
                                steps[1], pack.data(), nullptr) != nk_success_k)
            return 0;
        std::array<f32_t, 2> const point {static_cast<f32_t>(cell[0] - static_cast<f64_t>(first_column)),
                                          static_cast<f32_t>(cell[1] - static_cast<f64_t>(first_row))};
        if (kit_t::terrain_heights(window.data(), pack.data(), point.data(), 1, &height, nullptr) != nk_success_k)
            return 0;
    }
    return height == height ? static_cast<f64_t>(height) : 0;
}

template <cpu_class_t cpu_class_>
void geospatial_terrain<cpu_class_>::clear_sight_(relation_t relation, value_t const& query,
                                                  std::span<value_t const> entries, std::span<admission_t> verdicts,
                                                  thread_allocator_t const& scratch) const noexcept {
    // The query looks at the entries, except under `sees`, where every entry looks at the query.
    bool const entries_look = relation == relation_t::sees_k;
    std::array<f64_t, 3> const query_cell = terrain.grid_point(query.position);
    std::array<std::array<f64_t, 3>, 4> entry_cells;
    for (std::size_t lane = 0; lane != entries.size(); ++lane)
        entry_cells[lane] = terrain.grid_point(entries[lane].position);

    // A borrowed grid is read in place; a filled one is fetched over the window the on-grid
    // positions span.
    f32_t const* heights = terrain.heights;
    byte_t const* pack = terrain_pack.data();
    f64_t first_row = 0, first_column = 0;
    growing_buffer<f32_t, thread_allocator_t> window {scratch};
    growing_buffer<byte_t, thread_allocator_t> window_pack {scratch};
    if (!heights) {
        f64_t const last_row = static_cast<f64_t>(terrain.rows - 1),
                    last_column = static_cast<f64_t>(terrain.columns - 1);
        f64_t low_row = last_row, high_row = 0, low_column = last_column, high_column = 0;
        auto span_cell = [&](std::array<f64_t, 3> const& cell) noexcept {
            if (!(cell[0] >= 0 && cell[0] <= last_column && cell[1] >= 0 && cell[1] <= last_row)) return;
            low_column = std::fmin(low_column, cell[0]), high_column = std::fmax(high_column, cell[0]);
            low_row = std::fmin(low_row, cell[1]), high_row = std::fmax(high_row, cell[1]);
        };
        span_cell(query_cell);
        for (std::size_t lane = 0; lane != entries.size(); ++lane) span_cell(entry_cells[lane]);
        if (low_row > high_row || low_column > high_column) return;
        first_row = std::floor(low_row), first_column = std::floor(low_column);
        std::size_t const window_rows = min_of<std::size_t>(static_cast<std::size_t>(std::ceil(high_row)) + 1,
                                                            terrain.rows) -
                                        static_cast<std::size_t>(first_row);
        std::size_t const window_columns = min_of<std::size_t>(static_cast<std::size_t>(std::ceil(high_column)) + 1,
                                                               terrain.columns) -
                                           static_cast<std::size_t>(first_column);
        nk_size_t bytes = 0;
        if (failed(window.resize_uninitialized(window_rows * window_columns)) ||
            kit_t::terrain_pack_size(window_rows, window_columns, &bytes) != nk_success_k ||
            failed(window_pack.resize_uninitialized(bytes)))
            return;
        terrain.callback(static_cast<std::size_t>(first_row), static_cast<std::size_t>(first_column), window_rows,
                         window_columns, window.data(), window_columns * sizeof(f32_t), terrain.state);
        std::array<f64_t, 2> const steps = terrain.step_metres();
        if (kit_t::terrain_pack(window.data(), window_rows, window_columns, window_columns * sizeof(f32_t), steps[0],
                                steps[1], window_pack.data(), nullptr) != nk_success_k)
            return;
        heights = window.data(), pack = window_pack.data();
    }
    auto to_point = [&](std::array<f64_t, 3> const& cell) noexcept {
        return std::array<f32_t, 3> {static_cast<f32_t>(cell[0] - first_column),
                                     static_cast<f32_t>(cell[1] - first_row), static_cast<f32_t>(cell[2])};
    };
    std::array<f32_t, 3> const query_point = to_point(query_cell);
    std::array<f32_t, 12> entry_points;
    for (std::size_t lane = 0; lane != entries.size(); ++lane) {
        std::array<f32_t, 3> const point = to_point(entry_cells[lane]);
        entry_points[lane * 3 + 0] = point[0], entry_points[lane * 3 + 1] = point[1],
                                entry_points[lane * 3 + 2] = point[2];
    }
    // Light bends with a seventh of the Earth's curvature, which lowers the bulge under every sight
    // line.
    f64_t const refraction = 1.0 / 7;
    std::array<f32_t, 4> clearances;
    nk_status_t const tested = entries_look
                                   ? kit_t::clearances(heights, pack, entry_points.data(), nullptr, entries.size(),
                                                       query_point.data(), 1, refraction, clearances.data(),
                                                       sizeof(f32_t), nullptr)
                                   : kit_t::clearances(heights, pack, query_point.data(), nullptr, 1,
                                                       entry_points.data(), entries.size(), refraction,
                                                       clearances.data(), entries.size() * sizeof(f32_t), nullptr);
    if (tested != nk_success_k) return;
    // A negative clearance is an obstruction; NaN leaves the grid, which no terrain can block.
    for (std::size_t lane = 0; lane != entries.size(); ++lane)
        if (clearances[lane] < 0) verdicts[lane] = admission_t::reject_k;
}

template <cpu_class_t cpu_class_>
void geospatial_terrain<cpu_class_>::judge_values_(relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                                   value_t const& query, std::span<value_t const> entries,
                                                   std::span<admission_t> verdicts,
                                                   thread_allocator_t const& scratch) const noexcept {
    for (std::size_t lane = 0; lane != entries.size(); ++lane)
        verdicts[lane] = us::geospatial::admits(relation, tolerance.metres, tolerance.seconds * 1000.0, query,
                                                entries[lane])
                             ? admission_t::admit_k
                             : admission_t::reject_k;
    if (relation != relation_t::near_k && !terrain.empty()) clear_sight_(relation, query, entries, verdicts, scratch);
}

template <cpu_class_t cpu_class_>
template <typename context_type_>
void geospatial_ram_variant<cpu_class_>::judge_(relation_t relation, usearch_geospatial_tolerance_t tolerance,
                                                value_t const& query, context_type_ const& context, std::size_t first,
                                                std::size_t quad, usearch_predicate_t predicate,
                                                std::array<us::utilities::uuid_t, 4>& identities,
                                                std::array<admission_t, 4>& verdicts,
                                                thread_allocator_t const& scratch) const noexcept {
    std::array<value_t, 4> entries;
    for (std::size_t lane = 0; lane != quad; ++lane) {
        node_id_t const node = context.results[first + lane].node_id;
        entries[lane] = index.published_value_of(node);
        identities[lane] = index.key_of(node).value;
    }
    this->judge_values_(relation, tolerance, query, std::span<value_t const>(entries).first(quad),
                        std::span {verdicts}.first(quad), scratch);
    if (predicate.callback) {
        std::array<admission_t, 4> asked;
        asked.fill(admission_t::admit_k);
        key_predicate_t {predicate}(std::span {identities}.first(quad), std::span {asked}.first(quad));
        for (std::size_t lane = 0; lane != quad; ++lane)
            if (asked[lane] != admission_t::admit_k) verdicts[lane] = admission_t::reject_k;
    }
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_ram_variant<cpu_class_>::insert(usearch_geospatial_insert_t const& arguments,
                                                                           progress_ticks_t& ticks,
                                                                           executor_t& executor) noexcept {
    using key_t = typename index_t::key_t;
    usearch_geospatial_observation_t const* const observations = arguments.observations;
    std::size_t const count = arguments.count;
    std::size_t const breadth = arguments.expansion_factor ? arguments.expansion_factor
                                                           : us::geospatial::surface_keep_octants_t::exact_candidates_k;
    std::size_t const row_width = widest_row();
    thread_scratch<index_t, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, row_width, breadth, index.size(), this->metric_factory_());
        failed(prepared))
        return {{}, prepared};

    thread_allocator_t const caller {executor.arena(0)};
    growing_buffer<key_t, thread_allocator_t> identities {caller};
    growing_buffer<value_t, thread_allocator_t> values {caller};
    if (failed(identities.reserve_exactly(count)) || failed(values.reserve_exactly(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) {
        identities.push_back_reserved(uuid_of(key_at(arguments.keys, row)));
        values.push_back_reserved(value_t {});
    }
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        thread_allocator_t const allocator {executor.arena(thread)};
        f64_t const ground = this->ground_metres_(observations[row], allocator);
        values[row] = us::geospatial::to_earth_observation(geodetic_of(observations[row]), ground);
    });
    u32_t deviation = 0;
    for (value_t const& value : values) deviation = max_of<u32_t>(deviation, value.deviation_metres);

    auto [plan, plan_status] = index.plan_insert(count);
    if (failed(plan_status)) return {{}, plan_status};
    us::geospatial::surface_keep_octants_t const selector {geometry.connectivity};
    typename index_t::insert_config_t const config {breadth};
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        plan.insert(row, to_writable_span(values).subspan(row, 1), to_span(identities).subspan(row, 1),
                    scratch.metrics[thread], scratch.contexts[thread], selector, config);
        [[maybe_unused]] continuation_t const heard = ticks(row);
    });
    auto const range = plan.range;
    if (status_t const finished = plan.finish(); failed(finished)) return {{}, finished};
    raise_to(largest_deviation_metres, deviation);

    // Long links reach across the published graph, so they are drawn once the whole batch is
    // visible.
    if (status_t const linked = draw_long_links_(range, scratch, executor); failed(linked)) return {{}, linked};
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
status_t geospatial_ram_variant<cpu_class_>::draw_long_links_(iota_view<node_id_t> nodes,
                                                              thread_scratch<index_t, metric_t>& scratch,
                                                              executor_t& executor) noexcept {
    std::size_t const draws = long_links();
    status_t linked = status_t::success_k;
    drain_in_order(executor, nodes.size(), [&](std::size_t position, std::size_t thread) noexcept {
        node_id_t const node = static_cast<node_id_t>(nodes.first + position);
        splitmix64_t generator {mix_bits(static_cast<u64_t>(node))};
        keep_first_failure(linked, us::geospatial::draw_long_links(index, node, draws, scratch.metrics[thread],
                                                                   scratch.contexts[thread], generator));
    });
    return linked;
}

template <cpu_class_t cpu_class_>
status_t geospatial_ram_variant<cpu_class_>::refine(refine_plan_t& plan, refine_cursor_t& cursor,
                                                    progress_ticks_t& ticks, executor_t& executor) noexcept {
    if (status_t const refined = drive_refinement<index_t>(
            plan, cursor, this->metric_factory_(), us::geospatial::surface_keep_octants_t {geometry.connectivity},
            widest_row(), ticks, executor);
        failed(refined))
        return refined;
    // The re-selection kept the cone winners and the longest of the rest, so the drawn links of
    // every row it rewrote are drawn again by the insertion's rule.
    thread_scratch<index_t, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, widest_row(),
                                                  index.non_zero_expansion_factor(typename index_t::search_config_t {}),
                                                  index.size(), this->metric_factory_());
        failed(prepared))
        return prepared;
    return draw_long_links_({0, static_cast<node_id_t>(index.size())}, scratch, executor);
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_ram_variant<cpu_class_>::search(usearch_geospatial_search_t const& arguments,
                                                                           matches_landing_t& landing,
                                                                           progress_ticks_t& ticks,
                                                                           executor_t& executor) noexcept {
    usearch_geospatial_observation_t const* const queries = arguments.queries;
    relation_t const relation = static_cast<relation_t>(arguments.relation);
    usearch_geospatial_tolerance_t const tolerance = arguments.tolerance;
    usearch_predicate_t const predicate = arguments.predicate;
    std::size_t const wanted = arguments.wanted, expansion = arguments.expansion_factor;
    // Candidates fewer than the row they fill would short every row.
    std::size_t const breadth = max_of(index.non_zero_expansion_factor({expansion}), wanted);
    thread_scratch<index_t, metric_t> scratch;
    if (status_t const prepared = scratch.prepare(executor, widest_row(), breadth, index.size(),
                                                  this->metric_factory_());
        failed(prepared))
        return {{}, prepared};
    typename index_t::search_config_t const config {breadth};
    status_t searched = status_t::success_k;
    drain_in_order(executor, arguments.count, [&](std::size_t row, std::size_t thread) noexcept {
        if (failed(standard_atomic_ref<status_t>(searched).load(std::memory_order_relaxed))) return;
        thread_allocator_t const allocator {executor.arena(thread)};
        value_t const query = us::geospatial::to_earth_observation(geodetic_of(queries[row]),
                                                                   this->ground_metres_(queries[row], allocator));
        auto& context = scratch.contexts[thread];
        auto const [found, status] = index.search(query, scratch.metrics[thread], context, config);
        if (failed(status)) return keep_first_failure(searched, status);
        usearch_key_t* const keys = landing.keys + row * wanted;
        usearch_distance_t* const scores = landing.scores + row * wanted;
        std::size_t const listed_count = context.results.size();
        std::size_t landed = 0;
        for (std::size_t first = 0; first < listed_count && landed != wanted; first += 4) {
            std::size_t const quad = min_of<std::size_t>(4, listed_count - first);
            std::array<us::utilities::uuid_t, 4> identities;
            std::array<admission_t, 4> verdicts;
            judge_(relation, tolerance, query, context, first, quad, predicate, identities, verdicts, allocator);
            for (std::size_t lane = 0; lane != quad && landed != wanted; ++lane) {
                if (verdicts[lane] != admission_t::admit_k) continue;
                keys[landed] = std::bit_cast<usearch_key_t>(identities[lane]);
                scores[landed] = std::sqrt(static_cast<usearch_distance_t>(context.results[first + lane].distance)) /
                                 us::geospatial::centimetres_per_metre_k;
                ++landed;
            }
        }
        landing.counts[row] = landed;
        if (ticks(row) == continuation_t::cancel_k) keep_first_failure(searched, status_t::cancelled_k);
    });
    if (failed(searched)) return {{}, searched};
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_ram_variant<cpu_class_>::within(
    usearch_geospatial_observation_t const* queries, std::size_t count, relation_t relation,
    usearch_geospatial_tolerance_t tolerance, usearch_predicate_t predicate,
    growing_arena<forkunion_allocator_t>& staging, usearch_matches_tape_view_t& matches, std::size_t& written,
    progress_ticks_t& ticks, executor_t& executor) noexcept {
    using found_t = growing_buffer<geospatial_within_match_t, thread_allocator_t>;
    thread_scratch<index_t, metric_t> scratch;
    std::size_t const breadth = index.non_zero_expansion_factor(typename index_t::search_config_t {});
    if (status_t const prepared = scratch.prepare(executor, widest_row(), breadth, index.size(),
                                                  this->metric_factory_());
        failed(prepared))
        return {{}, prepared};
    growing_buffer<found_t, forkunion_allocator_t> found {forkunion_allocator_t {}};
    if (failed(found.reserve_exactly(executor.size()))) return {{}, status_t::bad_alloc_k};
    for (std::size_t thread = 0; thread != executor.size(); ++thread)
        found.push_back_reserved(found_t {thread_allocator_t {executor.arena(thread)}});
    f64_t const largest_deviation =
        standard_atomic_ref<u32_t>(largest_deviation_metres).load(std::memory_order_relaxed);

    status_t walked = status_t::success_k;
    drain_in_order(executor, count, [&](std::size_t row, std::size_t thread) noexcept {
        if (failed(standard_atomic_ref<status_t>(walked).load(std::memory_order_relaxed))) return;
        thread_allocator_t const allocator {executor.arena(thread)};
        value_t const query = us::geospatial::to_earth_observation(geodetic_of(queries[row]),
                                                                   this->ground_metres_(queries[row], allocator));
        // The walk reaches as far as the widest pair could match, and `admits` tightens back to
        // each pair's own.
        f64_t const deviations = std::sqrt(static_cast<f64_t>(query.deviation_metres) * query.deviation_metres +
                                           largest_deviation * largest_deviation);
        i64_t const reach = static_cast<i64_t>(
            std::ceil((tolerance.metres + 3 * deviations) * us::geospatial::centimetres_per_metre_k));
        auto& context = scratch.contexts[thread];
        us::geospatial::radius_result_t const ball = us::geospatial::within(
            index, query.position, reach, us::geospatial::radius_margin_t::spanner_k, widest_row(),
            scratch.metrics[thread], context);
        if (!ball) return keep_first_failure(walked, ball.status);
        for (std::size_t first = 0; first < ball.count; first += 4) {
            std::size_t const quad = min_of<std::size_t>(4, ball.count - first);
            std::array<us::utilities::uuid_t, 4> identities;
            std::array<admission_t, 4> verdicts;
            judge_(relation, tolerance, query, context, first, quad, predicate, identities, verdicts, allocator);
            for (std::size_t lane = 0; lane != quad; ++lane) {
                if (verdicts[lane] != admission_t::admit_k) continue;
                f64_t const metres = std::sqrt(static_cast<f64_t>(context.results[first + lane].distance)) /
                                     us::geospatial::centimetres_per_metre_k;
                if (failed(found[thread].push_back({row, std::bit_cast<usearch_key_t>(identities[lane]), metres})))
                    return keep_first_failure(walked, status_t::bad_alloc_k);
            }
        }
        if (ticks(row) == continuation_t::cancel_k) keep_first_failure(walked, status_t::cancelled_k);
    });
    if (failed(walked)) return {{}, walked};

    // Each query landed whole on one thread in ascending order, so counting per query places every
    // match.
    std::size_t* const offsets = carve<std::size_t>(staging, count + 1);
    if (!offsets) return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count + 1; ++row) offsets[row] = 0;
    std::size_t total = 0;
    for (found_t const& thread_found : found)
        for (geospatial_within_match_t const& match : thread_found) ++offsets[match.query + 1], ++total;
    for (std::size_t row = 0; row != count; ++row) offsets[row + 1] += offsets[row];
    usearch_key_t* const keys = carve<usearch_key_t>(staging, total);
    usearch_distance_t* const scores = carve<usearch_distance_t>(staging, total);
    growing_buffer<std::size_t, thread_allocator_t> cursors {thread_allocator_t {executor.arena(0)}};
    if ((total && (!keys || !scores)) || failed(cursors.resize_uninitialized(count)))
        return {{}, status_t::bad_alloc_k};
    for (std::size_t row = 0; row != count; ++row) cursors[row] = offsets[row];
    for (found_t const& thread_found : found)
        for (geospatial_within_match_t const& match : thread_found) {
            std::size_t const slot = cursors[match.query]++;
            keys[slot] = match.key, scores[slot] = match.score;
        }
    matches = {keys, scores, offsets};
    written = total;
    return {scratch.telemetry(), status_t::success_k};
}

template <cpu_class_t cpu_class_>
status_t geospatial_ram_variant<cpu_class_>::save(std::string_view path, executor_t& executor,
                                                  progress_ticks_t& ticks) noexcept {
    std::array<char, 24> connectivity_text;
    std::array<safegraphs_custom_field_t, 1> const fields {
        {{"connectivity", decimal_of(geometry.connectivity, connectivity_text)}}};
    save_config_t config;
    config.provenance = fields;
    return safegraphs_write_index(path, index, us::geospatial::earth_observation_codec_t {}, executor, config,
                                  token_of(geometry), ticks);
}

template <cpu_class_t cpu_class_>
status_t geospatial_ram_variant<cpu_class_>::load(std::string_view path, executor_t& executor,
                                                  progress_ticks_t& ticks) noexcept {
    if (status_t const loaded = safegraphs_read_index(path, index, us::geospatial::earth_observation_codec_t {},
                                                      executor, load_config_t {}, ticks);
        failed(loaded))
        return loaded;
    std::size_t const nodes = index.size();
    for (std::size_t node = 0; node != nodes; ++node)
        raise_to(largest_deviation_metres, index.published_value_of(static_cast<node_id_t>(node)).deviation_metres);
    return status_t::success_k;
}

/** One geospatial alternative on block storage: the basic NVMe index over the observations a RAM
 *  variant saved, read-only, scored with the RAM variant's metric and judged by its relations over
 *  the same terrain, each landed row tested before its read buffer recycles. Its verbs are
 *  instantiated in the class's unit, as the RAM variant's are. */
template <cpu_class_t cpu_class_>
struct geospatial_nvme_variant : geospatial_terrain<cpu_class_> {
    using executor_t = nvme_executor_t;
    using index_t = basic::nvme_index<us::geospatial::earth_observation_codec_t, us::utilities::uuid_t, u32_t, i64_t,
                                      default_io_engine_t, row_kit_of<cpu_class_>>;
    using value_t = us::geospatial::earth_observation_t;
    using relation_t = us::geospatial::relation_t;
    static constexpr cpu_class_t cpu_class_k = cpu_class_;

    index_t index;
    geospatial_geometry_t geometry;

    explicit geospatial_nvme_variant(forkunion_allocator_t const& allocator) noexcept : index(allocator) {}

    /** Serves the checkpoint at @p path in place, its parts from @p paths when any are named. */
    status_t view(std::string_view path, std::span<std::string_view const> paths, executor_t& executor) noexcept {
        return index.view(path, paths, token_of(geometry), executor);
    }

    expected<traversal_telemetry_t> search(usearch_geospatial_search_t const& arguments, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** A radius walk answers @c unsupported_k until the view serves it. */
    expected<traversal_telemetry_t> within(usearch_geospatial_observation_t const*, std::size_t, relation_t,
                                           usearch_geospatial_tolerance_t, usearch_predicate_t,
                                           growing_arena<forkunion_allocator_t>&, usearch_matches_tape_view_t&,
                                           std::size_t&, progress_ticks_t&, executor_t&) noexcept {
        return {{}, status_t::unsupported_k};
    }

    /** A view already is its checkpoint, so saving answers @c unsupported_k, as does sampling until
     *  the view serves it. */
    status_t save(std::string_view, executor_t&, progress_ticks_t&) noexcept { return status_t::unsupported_k; }
    status_t sample(usearch_geospatial_sample_t const&) noexcept { return status_t::unsupported_k; }
};

template <cpu_class_t cpu_class_>
expected<traversal_telemetry_t> geospatial_nvme_variant<cpu_class_>::search(
    usearch_geospatial_search_t const& arguments, matches_landing_t& landing, progress_ticks_t& ticks,
    executor_t& executor) noexcept {
    using key_t = typename index_t::key_t;
    static_assert(sizeof(key_t) == sizeof(usearch_key_t), "Keys land in the arena as the walks resolve them");
    usearch_geospatial_observation_t const* const queries = arguments.queries;
    relation_t const relation = static_cast<relation_t>(arguments.relation);
    usearch_geospatial_tolerance_t const tolerance = arguments.tolerance;
    usearch_predicate_t const predicate = arguments.predicate;
    std::size_t const count = arguments.count, wanted = arguments.wanted, expansion = arguments.expansion_factor;
    auto query_of = [&](std::size_t row, thread_allocator_t const& scratch) noexcept {
        value_t const query = us::geospatial::to_earth_observation(geodetic_of(queries[row]),
                                                                   this->ground_metres_(queries[row], scratch));
        return expected<value_t> {query, status_t::success_k};
    };
    auto judge = [&](value_t const& query, std::span<value_t const> entries, std::span<admission_t> verdicts,
                     thread_allocator_t const& scratch) noexcept {
        this->judge_values_(relation, tolerance, query, entries, verdicts, scratch);
    };
    // Candidates fewer than the row they fill would short every row.
    typename index_t::search_config_t const config {max_of(index.non_zero_expansion_factor({expansion}), wanted)};
    std::span<key_t> const keys {reinterpret_cast<key_t*>(landing.keys), count * wanted};
    std::span<usearch_distance_t> const scores {landing.scores, count * wanted};
    std::span<std::size_t> const counts {landing.counts, count};
    auto const [found, status] = [&]() noexcept {
        if (!predicate.callback)
            return index.search(count, this->metric_factory_(), query_of, executor, config, keys, scores, counts,
                                dummy_predicate_t {}, judge);
        key_predicate_t keyed {predicate};
        return index.search(count, this->metric_factory_(), query_of, executor, config, keys, scores, counts, keyed,
                            judge);
    }();
    if (failed(status)) return {{}, status};
    for (std::size_t row = 0; row != count; ++row)
        for (std::size_t match = 0; match != counts[row]; ++match)
            scores[row * wanted + match] = std::sqrt(scores[row * wanted + match]) /
                                           us::geospatial::centimetres_per_metre_k;
    [[maybe_unused]] continuation_t const heard = ticks(count - 1);
    return {{.computed_distances = found.computed_distances,
             .visited_members = found.visited_members,
             .restarts = found.restarts,
             .saturated_walks = found.saturated_walks},
            status_t::success_k};
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

/** The geospatial alternative on one NVIDIA GPU: the basic VRAM index over observations under the
 *  squared chord with 32-bit node IDs, and the kernel judging each walk's candidates under a
 *  search's relation. A batch encodes its observations on the host, as the RAM variants do. It
 *  holds no terrain, so lines of sight stay the RAM variants'. */
struct geospatial_vram_nvidia_variant
    : vram_nvidia_variant<
          us::geospatial::vram_nvidia_index<u32_t, geospatial_nvidia_capabilities_k, geospatial_nvidia_fatbin_t>> {
    using vram_nvidia_variant::vram_nvidia_variant;
    using value_t = us::geospatial::earth_observation_t;
    using node_id_t = index_t::node_id_t;

    geospatial_geometry_t geometry;

    /** Held by the one batch encoding its observations, landing its keys and inserting its rows. */
    std::mutex inserting;

    /** The host pages a batch encodes its observations into, rewound per batch. */
    growing_arena<forkunion_allocator_t> staging {forkunion_allocator_t {no_memory_domain_k}};

    /** The index's fatbin, loaded once more for the kernel judging the walks' candidates, and that
     *  kernel. */
    cuda_module_t admit_module;
    cuda_function_handle_t admit = nullptr;

    /** Loads the judging kernel onto the index's device. */
    status_t load_admit() noexcept {
        auto [module, status] = cuda_module_t::load(index.device_specs().ordinal, geospatial_nvidia_fatbin_t::bytes());
        if (failed(status)) return status;
        admit_module = std::move(module);
        vram_kernel_symbol_t symbol;
        vram_kernel_symbol(symbol, "geospatial", "cuda_admit", {"cuda_admit_arguments", 1},
                           geospatial_nvidia_capabilities_k, itanium_type<node_id_t> {});
        admit = admit_module.function(symbol.text);
        return admit ? status_t::success_k : status_t::device_error_k;
    }

    status_t reserve(usearch_geospatial_reserve_t const& arguments) noexcept {
        return reserve_(arguments.entries, geometry.connectivity, arguments.entries * sizeof(value_t));
    }

    /** Holds no terrain, as lines of sight stay the RAM variants'. */
    static constexpr bool has_terrain() noexcept { return false; }

    /** Packs no terrain, so it holds no terrain bytes. */
    static constexpr std::size_t terrain_memory_usage() noexcept { return 0; }

    expected<traversal_telemetry_t> insert(usearch_geospatial_insert_t const& arguments, progress_ticks_t& ticks,
                                           executor_t& executor) noexcept;

    /** Walks every query to candidates of the call's breadth, then keeps the first wanted entries
     *  of all of them that the relation and tolerance admit, on the device too. The queries encode
     *  into @p pages, the call arena's host pages, and the matches land in @p landing there. A
     *  device walk takes no host predicate. */
    expected<traversal_telemetry_t> search(usearch_geospatial_search_t const& arguments,
                                           growing_arena<forkunion_allocator_t>& pages, matches_landing_t& landing,
                                           progress_ticks_t& ticks, executor_t& executor) noexcept;

    /** A radius walk answers @c unsupported_k until the device serves it. */
    expected<traversal_telemetry_t> within(usearch_geospatial_observation_t const*, std::size_t,
                                           us::geospatial::relation_t, usearch_geospatial_tolerance_t,
                                           usearch_predicate_t, growing_arena<forkunion_allocator_t>&,
                                           usearch_matches_tape_view_t&, std::size_t&, progress_ticks_t&,
                                           executor_t&) noexcept {
        return {{}, status_t::unsupported_k};
    }

  private:

    /** Encodes the @p count observations into @p pages as a tape of rows at @p values, their byte
     *  starts at @p offsets. */
    static status_t encode_(usearch_geospatial_observation_t const* observations, std::size_t count,
                            growing_arena<forkunion_allocator_t>& pages, value_t*& values,
                            std::size_t*& offsets) noexcept {
        values = carve<value_t>(pages, count);
        offsets = carve<std::size_t>(pages, count + 1);
        if (!values || !offsets) return status_t::bad_alloc_k;
        for (std::size_t row = 0; row != count; ++row) {
            values[row] = us::geospatial::to_earth_observation(geodetic_of(observations[row]));
            offsets[row] = row * sizeof(value_t);
        }
        offsets[count] = count * sizeof(value_t);
        return status_t::success_k;
    }
};

inline expected<traversal_telemetry_t> geospatial_vram_nvidia_variant::insert(
    usearch_geospatial_insert_t const& arguments, progress_ticks_t& ticks, executor_t& executor) noexcept {
    std::size_t const count = arguments.count;
    std::lock_guard<std::mutex> held {inserting};
    staging.clear();
    value_t* values = nullptr;
    std::size_t* offsets = nullptr;
    if (status_t const encoded = encode_(arguments.observations, count, staging, values, offsets); failed(encoded))
        return {{}, encoded};
    return insert_(arguments.keys, reinterpret_cast<byte_t const*>(values), {offsets, count + 1}, index.size(),
                   geometry.connectivity, arguments.expansion_factor, ticks, executor);
}

inline expected<traversal_telemetry_t> geospatial_vram_nvidia_variant::search(
    usearch_geospatial_search_t const& arguments, growing_arena<forkunion_allocator_t>& pages,
    matches_landing_t& landing, progress_ticks_t& ticks, executor_t& executor) noexcept {
    std::size_t const count = arguments.count, wanted = arguments.wanted, expansion = arguments.expansion_factor;
    if (!wanted) {
        for (std::size_t row = 0; row != count; ++row) landing.counts[row] = 0;
        return {{}, status_t::success_k};
    }
    value_t* values = nullptr;
    std::size_t* offsets = nullptr;
    if (status_t const encoded = encode_(arguments.queries, count, pages, values, offsets); failed(encoded))
        return {{}, encoded};
    byte_t const* const bytes = reinterpret_cast<byte_t const*>(values);
    std::span<std::size_t const> const tape {offsets, count + 1};
    // Candidates fewer than the row they fill would short every row.
    std::size_t const candidates_count = max_of(expansion ? expansion : default_expansion_factor(index.connectivity()),
                                                wanted);
    index_t::search_config_t const config {static_cast<u32_t>(candidates_count), static_cast<u32_t>(candidates_count)};

    // Past the walk's own scratch: the queries, the judged metres and counts, the candidates and
    // the judged IDs, the eight-byte columns first so every column keeps its alignment.
    std::size_t const listed_count = count * candidates_count, judged = count * wanted;
    std::size_t const judging_bytes = count * (sizeof(value_t) + sizeof(u64_t)) + judged * sizeof(f64_t) +
                                      listed_count * (sizeof(node_id_t) + sizeof(f32_t)) + judged * sizeof(node_id_t);
    std::size_t const bytes_per_stream = index.traversal_memory_usage(bytes, tape, config) +
                                         round_up_to_multiple(judging_bytes, default_alignment_k);
    // IDs land narrow at the front of the keys, then widen backward; metres and counts land whole.
    node_id_t* const ids = reinterpret_cast<node_id_t*>(landing.keys);
    auto judge = [&](index_t::traversal_context_t& context,
                     cuda_arena_allocator_t& allocator) noexcept -> expected<index_t::search_result_t> {
        value_t* const device_queries = reinterpret_cast<value_t*>(allocator.allocate(judging_bytes));
        if (!device_queries) return {{}, status_t::bad_alloc_k};
        f64_t* const matched_metres = reinterpret_cast<f64_t*>(device_queries + count);
        u64_t* const matched_counts = reinterpret_cast<u64_t*>(matched_metres + judged);
        node_id_t* const candidate_ids = reinterpret_cast<node_id_t*>(matched_counts + count);
        f32_t* const candidate_distances = reinterpret_cast<f32_t*>(candidate_ids + listed_count);
        node_id_t* const matched_ids = reinterpret_cast<node_id_t*>(candidate_distances + listed_count);
        us::geospatial::cuda_admit_arguments<node_id_t> admitting;
        admitting.storage = index.storage();
        admitting.queries = device_queries;
        admitting.candidate_ids = candidate_ids;
        admitting.matched_ids = matched_ids;
        admitting.matched_metres = matched_metres;
        admitting.matched_counts = matched_counts;
        admitting.queries_count = static_cast<u32_t>(count);
        admitting.candidates_count = static_cast<u32_t>(candidates_count);
        admitting.wanted = static_cast<u32_t>(wanted);
        admitting.relation = static_cast<us::geospatial::relation_t>(arguments.relation);
        admitting.radius_metres = arguments.tolerance.metres;
        admitting.slack_milliseconds = arguments.tolerance.seconds * 1000.0;
        if (cuda_copy_async(device_queries, values, count * sizeof(value_t), context.stream()) != cuda_success_k)
            return {{}, status_t::device_error_k};
        auto const [found, status] = index.search(bytes, tape, {candidate_ids, {count, candidates_count}},
                                                  {candidate_distances, {count, candidates_count}}, context, config);
        if (failed(status)) return {{}, status};
        u32_t const blocks = static_cast<u32_t>(
            min_of<std::size_t>(divide_round_up<std::size_t>(count, vram_warps_per_block_k),
                                std::size_t {index.device_specs().multiprocessor_count} * 8));
        if (failed(cuda_launch(admit, blocks, vram_warps_per_block_k * 32, 0, context.stream(),
                               cuda_overlap_t::serialized_k, admitting)) ||
            cuda_copy_async(ids, matched_ids, judged * sizeof(node_id_t), context.stream()) != cuda_success_k ||
            cuda_copy_async(landing.scores, matched_metres, judged * sizeof(f64_t), context.stream()) !=
                cuda_success_k ||
            cuda_copy_async(landing.counts, matched_counts, count * sizeof(u64_t), context.stream()) != cuda_success_k)
            return {{}, status_t::device_error_k};
        if (status_t const drained = context.wait(); failed(drained)) return {{}, drained};
        return {found, status_t::success_k};
    };
    expected<traversal_telemetry_t> const searched = walk_(executor, bytes_per_stream, count, ticks, judge);
    if (failed(searched.status)) return searched;
    for (std::size_t match = judged; match-- != 0;)
        if (match % wanted < landing.counts[match / wanted])
            landing.keys[match] = std::bit_cast<usearch_key_t>(keys[ids[match]]);
    return searched;
}

#endif

#pragma endregion Geospatial Variants
