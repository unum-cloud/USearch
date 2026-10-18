/**
 *  @file bench/geospatial_index_search.cpp
 *  @author Ash Vardanian
 *  @date October 15, 2026
 *  @brief The geospatial search benchmark: synthetic GeoNames-like places built into one RAM or
 *      VRAM index through the C interface and judged against an exact chord scan.
 *
 *  Places cluster into towns spread over the globe, a town's share of places falling as the inverse
 *  of its rank, so a few cities hold most places and most towns a handful, scattered over a few
 *  kilometres. A VRAM index lands its matches in a host arena, as a C caller reading them would. An
 *  NVMe run builds in RAM, saves a checkpoint to the temporary directory and searches a view of it.
 *
 *  @verbatim
 *  --places     10,001,000  Synthetic places, the queries among them
 *  --residency  ram         Where the index lives: ram, vram on GPU zero, or nvme
 *  @endverbatim
 */
#include <cmath>   // `std::asin`, `std::sqrt`, `std::pow`
#include <cstdint> // `std::uint8_t` - the checkpoint path's bytes
#include <cstdio>  // `std::remove` - the checkpoint an NVMe run saved

#include <array>       // `std::array` - the binary's flags
#include <filesystem>  // `std::filesystem::temp_directory_path` - where an NVMe run saves
#include <string>      // `std::string` - the checkpoint path
#include <string_view> // `std::string_view`
#include <vector>      // `std::vector` - places and their positions

#include "usearch/geospatial_index.h"
#include "usearch/geospatial/ram_index.hpp"

#include "basic.cuh"

namespace unum::usearch::bench {

/** Draws @p count GeoNames-like places from @p seed, as the file's docblock describes. */
inline std::vector<usearch_geospatial_observation_t> generate_places(std::size_t count, u64_t seed) {
    constexpr f64_t nan_k = std::numeric_limits<f64_t>::quiet_NaN();
    constexpr f64_t degrees_k = 180.0 / 3.14159265358979323846;
    constexpr std::size_t places_per_town_k = 64;
    constexpr f64_t spread_degrees_k = 0.02;
    splitmix64_t random_generator {seed};
    std::size_t const towns = std::max<std::size_t>(1, count / places_per_town_k);
    std::vector<std::array<f64_t, 2>> centres(towns);
    for (std::array<f64_t, 2>& centre : centres)
        centre = {std::asin(2 * geospatial::uniform_unit(random_generator) - 1) * degrees_k,
                  360 * geospatial::uniform_unit(random_generator) - 180};
    std::vector<usearch_geospatial_observation_t> places(count);
    for (usearch_geospatial_observation_t& place : places) {
        // A log-uniform rank draws a town in inverse proportion to its rank.
        std::size_t const town = std::min(
            towns - 1,
            static_cast<std::size_t>(std::pow(static_cast<f64_t>(towns), geospatial::uniform_unit(random_generator))) -
                1);
        // The sum of three uniforms approximates a normal offset.
        f64_t offsets[2];
        for (f64_t& offset : offsets)
            offset = spread_degrees_k *
                     (geospatial::uniform_unit(random_generator) + geospatial::uniform_unit(random_generator) +
                      geospatial::uniform_unit(random_generator) - 1.5);
        f64_t const latitude = std::min(90.0, std::max(-90.0, centres[town][0] + offsets[0]));
        f64_t const longitude = centres[town][1] + offsets[1];
        place = {latitude, longitude > 180 ? longitude - 360 : longitude,
                 nan_k,    nan_k,
                 nan_k,    nan_k,
                 nan_k,    nan_k,
                 nan_k,    nan_k,
                 nan_k};
    }
    return places;
}

/** Runs the geospatial search benchmark over the flags in @p argv. */
int bench_geospatial_index_search(int argc, char const* const* argv) {
    basic_settings_t settings;
    std::string_view residency = "ram";
    count_t places_count {.value = 10'001'000};
    std::array<flag_t, 2> const flags {{
        {"--places", "count", "synthetic places, the queries among them; 10,001,000 by default", &places_count},
        {"--residency", "name", "where the index lives: ram, vram on GPU zero, or nvme; ram by default", &residency},
    }};
    read_basic_settings(argc, argv, "usearch_bench_geospatial_index_search [flags]", settings, flags);
    if (residency != "ram" && residency != "vram" && residency != "nvme")
        return fmt::println(stderr, "--residency=\"{}\" does not parse, expected ram, vram or nvme", residency), 1;
    usearch_vram_options_t const vram {};

    std::vector<usearch_geospatial_observation_t> const places = generate_places(places_count.value, 42);
    tape_t tape;
    tape.elements = places.data();
    tape.offsets.resize(places.size() + 1);
    for (std::size_t row = 0; row != tape.offsets.size(); ++row) tape.offsets[row] = row;
    std::vector<geospatial::earth_fixed_t> positions(places.size());
    for (std::size_t row = 0; row != places.size(); ++row)
        positions[row] = geospatial::to_earth_fixed({places[row].latitude_degrees, places[row].longitude_degrees, 0});

    usearch_outcome_t outcome {};
    usearch_geospatial_index_t index = nullptr;
    if (residency == "vram") {
        usearch_geospatial_init_vram_t init {.index = &index,
                                             .outcome = &outcome,
                                             .connectivity = settings.connectivity.value,
                                             .vram = vram};
        usearch_geospatial_init_vram(&init);
    }
    else {
        usearch_geospatial_init_ram_t init {.index = &index,
                                            .outcome = &outcome,
                                            .connectivity = settings.connectivity.value};
        usearch_geospatial_init_ram(&init);
    }
    if (outcome.message) return fmt::println(stderr, "Init: {}", outcome.message), 1;
    usearch_geospatial_index_description_t description {};
    usearch_geospatial_describe_t describe {.index = index, .outcome = &outcome, .description = &description};
    usearch_geospatial_describe(&describe);
    fmt::println("- Places: GeoNames-like · residency: {} · class: {}", residency, description.capabilities);

    auto const observations_of = [](usearch_tape_view_t entries) noexcept {
        return static_cast<usearch_geospatial_observation_t const*>(entries.data) + entries.offsets[0];
    };
    auto const insert = [&](usearch_executor_t executor, usearch_keys_view_t keys, usearch_tape_view_t entries,
                            std::size_t count, usearch_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_geospatial_insert_t arguments {.index = index,
                                               .executor = executor,
                                               .outcome = reached,
                                               .count = count,
                                               .keys = keys,
                                               .observations = observations_of(entries),
                                               .expansion_factor = settings.expansion_add.value,
                                               .stats = stats};
        usearch_geospatial_insert(&arguments);
    };
    auto const search = [&](usearch_executor_t executor, usearch_arena_t arena, usearch_tape_view_t queries,
                            std::size_t count, std::size_t expansion, usearch_matches_view_t* matches,
                            usearch_traversal_stats_t* stats, usearch_outcome_t* reached) {
        usearch_geospatial_search_t arguments {.index = index,
                                               .executor = executor,
                                               .outcome = reached,
                                               .count = count,
                                               .queries = observations_of(queries),
                                               .relation = usearch_geospatial_relation_near_k,
                                               .matches = matches,
                                               .arena = arena,
                                               .wanted = settings.top_k.value,
                                               .expansion_factor = expansion,
                                               .stats = stats};
        usearch_geospatial_search(&arguments);
    };
    auto const exact = [&](std::size_t, std::size_t query, std::span<double> distances) {
        for (std::size_t row = 0; row != distances.size(); ++row)
            distances[row] = std::sqrt(
                                 static_cast<f64_t>(geospatial::chord_squared(positions[query], positions[row]))) /
                             geospatial::centimetres_per_metre_k;
    };

    // An NVMe run saves what it built and searches a view of the checkpoint, on storage threads.
    std::string const checkpoint =
        (std::filesystem::temp_directory_path() / "usearch-bench-geospatial.safegraphs").string();
    usearch_bytes_view_t const checkpoint_view {reinterpret_cast<std::uint8_t const*>(checkpoint.data()),
                                                checkpoint.size()};
    usearch_executor_t storage = nullptr;
    auto const serve = [&](usearch_executor_t executor, usearch_outcome_t* reached) {
        if (residency != "nvme") return executor;
        usearch_geospatial_save_t save {.index = index,
                                        .executor = executor,
                                        .outcome = reached,
                                        .path = checkpoint_view};
        usearch_geospatial_save(&save);
        if (reached->message) return executor;
        usearch_geospatial_free(index);
        index = nullptr;
        usearch_executor_for_nvme_t storage_arguments {.executor = &storage,
                                                       .outcome = reached,
                                                       .threads = settings.threads.value};
        usearch_executor_for_nvme(&storage_arguments);
        if (reached->message) return executor;
        usearch_geospatial_view_nvme_t view {.index = &index,
                                             .executor = storage,
                                             .outcome = reached,
                                             .path = checkpoint_view};
        usearch_geospatial_view_nvme(&view);
        return storage;
    };
    int const exit_code = run_basic_bench<usearch_traversal_stats_t>(
        settings, tape, residency == "vram" ? &vram : nullptr, insert, search, exact, serve);
    usearch_geospatial_free(index);
    usearch_executor_free(storage);
    if (residency == "nvme") std::remove(checkpoint.c_str());
    return exit_code;
}

} // namespace unum::usearch::bench

int main(int argc, char** argv) { return us::bench::bench_geospatial_index_search(argc, argv); }
