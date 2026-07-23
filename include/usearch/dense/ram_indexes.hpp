/**
 *  @brief Typed multi-shard views over CPU-resident dense indexes - one shard per memory
 *      domain - with compute-domain routing and merged cross-shard search.
 *  @author Ash Vardanian
 *  @file include/usearch/dense/ram_indexes.hpp
 *  @date July 23, 2026
 *
 *  The plural layer over @c dense_ram_index: a group of same-geometry shards, each pinned
 *  to one ForkUnion memory domain, viewed as one index. Insertions route every compute
 *  domain to its local shard, so graph memory and the threads walking it share a memory
 *  domain; searches scatter over every shard and gather through @ref merge_search_results,
 *  translating shard-local node ids into the caller's key space on the way out.
 */
#pragma once

#include <span> // `std::span` - the caller's merged ranking & the adopted shard list

#include <forkunion.h>         // `fu_topology_t`, `fu_local_memory_of` - the placement facts
#include <forkunion/types.hpp> // `fu::dynamic_array` - the shard & routing tables

#include "usearch/types.hpp"
#include "usearch/dense/ram_index.hpp"

namespace unum {
namespace usearch {

/**
 *  @brief A group of same-geometry CPU-resident shards viewed as one index.
 *
 *  The group owns only placement & gathering: which shard a compute domain feeds, and how
 *  per-shard rankings merge into one. Shards are adopted, never owned - an index holds
 *  atomics and never moves, so the caller keeps them address-stable. Construction,
 *  reservation, checkpointing, and ingestion stay per-shard verbs on @c index_t - the
 *  caller drives them slot by slot, keeping every failure attributable to one shard.
 */
template <typename index_type_>
struct dense_ram_indexes {
    using index_t = index_type_;
    using distance_t = typename index_t::distance_t;
    using search_config_t = typename index_t::search_config_t;
    using traversal_t = typename index_t::dense_traversal_context_t;
    using vector_view_t = typename index_t::vector_view_t;

    /** @brief Outcome of one scattered lookup: the merged hit count on @c success_k, and
     *      the traversal telemetry summed across every shard the query walked. */
    struct search_result_t {
        /** @brief @c success_k, or why one of the scattered lookups could not run. */
        status_t status = status_t::success_k;
        /** @brief Hits merged into the caller's ranking, never exceeding its capacity. */
        std::size_t count = 0;
        /** @brief Frontier candidates expanded across all walked shards. */
        std::size_t visited_members = 0;
        /** @brief Total distance computations across all walked shards. */
        std::size_t computed_distances = 0;
        /** @brief Starvation restarts - a graph-connectivity health signal. */
        std::size_t restarts = 0;

        explicit operator bool() const noexcept { return status == status_t::success_k; }
    };

    /** @brief The adopted shards, never owned; one per engaged memory domain. */
    fu::dynamic_array<index_t*> shards_;
    /** @brief Routes every prong to the shard local to its compute domain. */
    fu::dynamic_array<std::size_t> shard_of_compute_domain_;

    std::size_t shards_count() const noexcept { return shards_.size(); }
    index_t& shard(std::size_t slot) noexcept { return *shards_[slot]; }
    index_t const& shard(std::size_t slot) const noexcept { return *shards_[slot]; }

    /** @brief The shard slot local to @p compute_domain; slot 0 until @ref route runs. */
    std::size_t local_slot(std::size_t compute_domain) const noexcept {
        return compute_domain < shard_of_compute_domain_.size() ? shard_of_compute_domain_[compute_domain] : 0;
    }
    index_t& local_shard(std::size_t compute_domain) noexcept { return *shards_[local_slot(compute_domain)]; }

    /** @brief Published entries across every shard. */
    std::size_t size() const noexcept {
        std::size_t total = 0;
        for (index_t* one : shards_) total += one->size();
        return total;
    }

    /** @brief Adopts the caller's address-stable shards, replacing any previous adoption. */
    status_t adopt(std::span<index_t* const> shards) noexcept {
        if (!shards_.try_resize(shards.size())) return status_t::bad_alloc_k;
        for (std::size_t slot = 0; slot != shards.size(); ++slot) shards_[slot] = shards[slot];
        return status_t::success_k;
    }

    /**
     *  @brief Fills the compute-domain routing table from the machine's topology, pairing
     *      each compute domain with the shard pinned to its local memory domain.
     *
     *  @param[in] topology The ForkUnion topology the driving pool was built from.
     *  @param[in] memory_domains The engaged domains, positionally matching the shards;
     *      with one shard or fewer domains every compute domain routes to slot 0.
     */
    status_t route(fu_topology_t topology, std::span<fu_memory_domain_id_t const> memory_domains) noexcept {
        std::size_t const compute_domains = fu_compute_domains_count(topology);
        if (!shard_of_compute_domain_.try_resize(compute_domains)) return status_t::bad_alloc_k;
        if (memory_domains.size() < 2) return status_t::success_k;
        for (std::size_t compute_domain = 0; compute_domain != compute_domains; ++compute_domain) {
            std::size_t const local_memory = fu_local_memory_of(topology, compute_domain);
            for (std::size_t slot = 0; slot != memory_domains.size(); ++slot)
                if (static_cast<std::size_t>(memory_domains[slot]) == local_memory)
                    shard_of_compute_domain_[compute_domain] = slot;
        }
        return status_t::success_k;
    }

    /**
     *  @brief Scatters one query over every shard and gathers the rankings into one,
     *      ascending by distance - the honest end-to-end multi-domain lookup.
     *
     *  @param[in] query_vector The query, in the shards' stored scalar type.
     *  @param[inout] traversal The calling thread's traversal context, reused per shard.
     *  @param[in] config The per-shard search configuration, applied uniformly.
     *  @param[in] keys_of_shard Indexable per-slot id-to-key mappings, so shards with
     *      colliding id spaces land in one shared ranking.
     *  @param[inout] keys Keys of the merged ranking; its length is the capacity.
     *  @param[inout] distances Distances of the same ranking, parallel to @p keys.
     */
    template <typename keys_of_shard_type_, typename key_type_, typename distance_type_>
    search_result_t search(vector_view_t query_vector, traversal_t& traversal, search_config_t config,
                           keys_of_shard_type_ const& keys_of_shard, std::span<key_type_> keys,
                           std::span<distance_type_> distances) noexcept {
        search_result_t merged;
        for (std::size_t slot = 0; slot != shards_.size(); ++slot) {
            auto found = shards_[slot]->search(query_vector, traversal, config);
            if (!found) return merged.status = found.status, merged;
            merged.visited_members += found.visited_members;
            merged.computed_distances += found.computed_distances;
            merged.restarts += found.restarts;
            merged.count = merge_search_results(to_span(traversal.results), keys_of_shard[slot], keys, distances,
                                                merged.count);
        }
        return merged;
    }
};

} // namespace usearch
} // namespace unum
