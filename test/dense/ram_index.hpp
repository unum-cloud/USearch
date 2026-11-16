/**
 *  @file test/dense/ram_index.hpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief The cells of the CPU-resident @c dense::ram_index the per-dtype units compose: the full
 *      suite over the allocation ledger, and the starved cell over ledgered scratch.
 *
 *  The scenarios live in @c dense.cuh, templated over the index type. Cells are gated by a
 *  compile-time concept, so a kit absent from the build reports as skipped. The flaky waiter
 *  oversleeps at random, widening the interleaving windows around locks and watermark turns far
 *  beyond hardware timing. Removal, update, and filtering tests are deliberately absent: those
 *  features are banned at this layer by design.
 */
#pragma once

#include <cstring> // `std::memcmp` - a stored row against its source

#include <string_view> // `std::string_view` - the phase a round stands in

#include <fmt/format.h>

#include <numkong/numkong.hpp>

#include "usearch/dense/ram_index.hpp"

#include "dense.cuh"

namespace unum::usearch::test::dense {

/** Whether the stored row of @p node in @p index holds exactly the bytes of row @p row of
 *  @p vectors. */
template <typename index_type_, typename vectors_type_>
static bool ram_index_row_matches(index_type_ const& index, node_id_t node, vectors_type_ const& vectors,
                                  std::size_t row) {
    std::size_t const bytes = packed_bytes(index_type_::vector_dtype_k, index.dimensions());
    return std::memcmp(index.vector(node).byte_data(), vectors.view().row(row).as_vector().byte_data(), bytes) == 0;
}

/** A lone index keeps its keys without a reservation: the first insert grows it, every key resolves
 *  to both of its rows through @c key_of, @c count, @c find_each and @c get, a cut carries every
 *  key beside its row, and a save and load round trip keeps every key, into a wider index too,
 *  whose rows one refine grows past the saved width while every node stays reachable. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_ram_index_keys(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    using keys_view_t = typename index_type_::keys_view_t;
    std::size_t failures = 0;
    std::size_t const rows = dense_thread_nodes_k, dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto vectors = make_vectors<scalar_t>(rows, dimensions, stream_key(seed, "index_keys"));
    // Rows 2k and 2k + 1 share key k + 1, so every lookup must find a pair.
    std::vector<key_t> keys(rows);
    for (std::size_t row = 0; row != rows; ++row) keys[row] = dense_key<key_t>(1 + row / 2);
    auto rows_of_key = [&](key_t const& key) { return 2 * (dense_key_number(key) - 1); };
    allocation_ledger_t ledger;
    {
        std::unique_ptr<index_type_> const index_ptr = make_index(ledger);
        index_type_& index = *index_ptr;
        typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
        auto assigned = nk::vector<node_id_t>::zeros(rows).value;
        failures += expect_eq(index.capacity(), 0u);
        failures += expect(
            succeeded(index.insert(keys_view_t {keys.data(), rows}, vectors.view(), assigned.span(), context).status));
        failures += expect_eq(index.size(), rows);

        std::size_t wrong = 0;
        for (std::size_t row = 0; row != rows; ++row) {
            wrong += index.key_of(assigned[row]).value != keys[row];
            std::size_t own = 0;
            index.find_each(keys[row], [&](node_id_t node) noexcept { own += node == assigned[row]; });
            wrong += own != 1 || index.count(keys[row]) != 2 || !index.contains(keys[row]);
        }
        failures += expect_eq(wrong, 0u);
        failures += expect_eq(index.key_of(static_cast<node_id_t>(rows)).status, status_t::config_mismatch_k);
        failures += expect(!index.contains(dense_key<key_t>(rows)));

        // A pair exports whole, and a narrower export still reports the true total.
        auto exported = nk::matrix<scalar_t>::zeros({2, dimensions}).value;
        failures += expect_eq(index.get(keys[0], rows_span_t {exported.span()}, 2), 2u);
        failures += expect_eq(index.get(keys[0], rows_span_t {exported.span()}, 1), 2u);
        failures += expect_eq(index.get(dense_key<key_t>(rows), rows_span_t {exported.span()}, 2), 0u);

        // A reshard into two targets carries every key beside its row into one of the parts, and
        // the source forgets it.
        std::unique_ptr<index_type_> const first_ptr = make_index(ledger), second_ptr = make_index(ledger);
        std::array<index_type_*, 2> targets {first_ptr.get(), second_ptr.get()};
        {
            index_type_* const source = &index;
            auto [cutting, cut_status] = index.plan_reshard(std::span<index_type_* const> {&source, 1},
                                                            std::span<index_type_* const> {targets});
            failures += expect(succeeded(cut_status) && succeeded(cutting.reserve(context)));
            if (succeeded(cut_status)) {
                do
                    for (auto const task : cutting.tasks()) failures += expect(succeeded(cutting.run(task, context)));
                while (cutting.advance());
                failures += expect(succeeded(cutting.finish().status));
                // A finished plan advances no further, and finishing it again answers the same.
                failures += expect(!cutting.advance() && succeeded(cutting.finish().status));
            }
        }
        failures += expect_eq(index.size(), 0u);
        failures += expect_eq(index.count(keys[0]), 0u);
        failures += expect_eq(targets[0]->size() + targets[1]->size(), rows);
        wrong = 0;
        for (index_type_ const* part : targets)
            for (node_id_t node = 0; node != part->size(); ++node) {
                key_t const key = part->key_of(node).value;
                std::size_t const row = rows_of_key(key);
                wrong += !ram_index_row_matches(*part, node, vectors, row) &&
                         !ram_index_row_matches(*part, node, vectors, row + 1);
            }
        for (std::size_t row = 0; row != rows; row += 2)
            wrong += targets[0]->count(keys[row]) + targets[1]->count(keys[row]) != 2;
        failures += expect_eq(wrong, 0u);

        // A save and a load through the plans keep every key beside its row.
        index_type_& saved = *targets[0];
        auto const shape = saved.saved_shape();
        auto saved_vectors = nk::matrix<scalar_t>::zeros({shape.nodes, shape.dimensions}).value;
        auto saved_neighbors = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
        failures += expect_eq(saved.save({dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                                         synchronization_t::exclusive_k),
                              status_t::success_k);
        std::vector<key_t> saved_keys(shape.nodes);
        for (node_id_t node = 0; node != shape.nodes; ++node) saved_keys[node] = saved.key_of(node).value;
        std::unique_ptr<index_type_> const restored_ptr = make_index(ledger);
        index_type_& restored = *restored_ptr;
        failures += expect_eq(
            restored.load({dense_matrix_bytes(saved_vectors.view()), dense_matrix_bytes(saved_neighbors.view())},
                          to_span(saved_keys), shape.nodes, {0, static_cast<node_id_t>(shape.nodes)}),
            status_t::success_k);
        failures += expect_eq(restored.size(), shape.nodes);
        wrong = 0;
        for (node_id_t node = 0; node != shape.nodes; ++node) {
            wrong += restored.key_of(node).value != saved_keys[node];
            wrong += restored.count(saved_keys[node]) != saved.count(saved_keys[node]);
            wrong += !ram_index_row_matches(restored, node, vectors, rows_of_key(saved_keys[node])) &&
                     !ram_index_row_matches(restored, node, vectors, rows_of_key(saved_keys[node]) + 1);
        }
        failures += expect_eq(wrong, 0u);

        // A wider index takes the saved rows with empty tails, and a round over them reaches all.
        iota_view<node_id_t> const saved_nodes {0, static_cast<node_id_t>(shape.nodes)};
        std::unique_ptr<index_type_> const widened_ptr = make_index(
            ledger, {shape.dimensions, static_cast<connectivity_t>(2 * shape.connectivity)});
        index_type_& widened = *widened_ptr;
        typename index_type_::columns_view_t const narrow_columns {
            .vectors = dense_matrix_bytes(saved_vectors.view()),
            .neighbors = dense_matrix_bytes(saved_neighbors.view()),
            .connectivity = shape.connectivity,
        };
        failures += expect_eq(widened.load(narrow_columns, to_span(saved_keys), shape.nodes, saved_nodes),
                              status_t::success_k);
        failures += expect(succeeded(widened.refine({}, context).status));
        auto const wide_shape = widened.saved_shape();
        failures += expect_eq(wide_shape.connectivity, 2 * shape.connectivity);
        auto wide_vectors = nk::matrix<scalar_t>::zeros({wide_shape.nodes, wide_shape.dimensions}).value;
        auto wide_neighbors = nk::matrix<node_id_t>::zeros({wide_shape.nodes, wide_shape.connectivity}).value;
        failures += expect_eq(widened.save({dense_matrix_bytes(wide_vectors), dense_matrix_bytes(wide_neighbors)},
                                           synchronization_t::exclusive_k),
                              status_t::success_k);
        auto const [reachability, health_status] = dense_audit_health(widened, context);
        failures += expect(succeeded(health_status));
        failures += expect_eq(reachability.reachable_nodes, reachability.audited_nodes);
        // The wider rows never fit back into the narrower index.
        std::unique_ptr<index_type_> const narrowed_ptr = make_index(ledger);
        typename index_type_::columns_view_t const wide_columns {
            .vectors = dense_matrix_bytes(wide_vectors.view()),
            .neighbors = dense_matrix_bytes(wide_neighbors.view()),
            .connectivity = wide_shape.connectivity,
        };
        failures += expect_eq(narrowed_ptr->load(wide_columns, to_span(saved_keys), shape.nodes, saved_nodes),
                              status_t::config_mismatch_k);
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** A RAM index whose rows hold @p links, one list of targets per entry, its vectors drawn from
 *  @p seed: a graph written by hand rather than built; null when it cannot be loaded. */
template <typename index_type_, typename make_index_type_>
static std::unique_ptr<index_type_> ram_index_of_links(make_index_type_ const& make_index, allocation_ledger_t& ledger,
                                                       std::vector<std::vector<node_id_t>> const& links, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    std::size_t const entries = links.size();
    std::unique_ptr<index_type_> index = make_index(ledger);
    std::size_t const dimensions = index->dimensions(), connectivity = index->connectivity();
    auto vectors = make_vectors<scalar_t>(entries, dimensions, stream_key(seed, "audit_links"));
    auto neighbors = nk::matrix<node_id_t>::zeros({entries, connectivity}).value;
    for (std::size_t entry = 0; entry != entries; ++entry) {
        auto row = neighbors.span().row(entry).as_vector();
        for (std::size_t slot = 0; slot != connectivity; ++slot)
            row[slot] = slot < links[entry].size() ? links[entry][slot] : node_id_unknown_k;
        if (links[entry].empty()) row[0] = static_cast<node_id_t>(entry);
    }
    std::vector<typename index_type_::key_t> const keys = dense_keys<index_type_>(entries);
    if (failed(index->reserve({entries}).status) ||
        failed(index->load({dense_matrix_bytes(vectors.view()), dense_matrix_bytes(neighbors.view())}, to_span(keys),
                           entries, {0, static_cast<node_id_t>(entries)})))
        return nullptr;
    return index;
}

/** The order an audit's driver runs the tasks of each pass in. */
enum class ram_tasks_order_t { first_to_last_k, last_to_first_k };

/** Audits @p index to convergence, the tasks of every pass run in @p order. */
template <typename index_type_>
static typename index_type_::health_plan_t ram_index_audit(index_type_& index, ram_tasks_order_t order) {
    auto [plan, status] = index.plan_health();
    do {
        auto const tasks = plan.tasks();
        for (std::size_t task = 0; task != tasks.size(); ++task)
            plan.sweep(tasks[order == ram_tasks_order_t::last_to_first_k ? tasks.size() - 1 - task : task]);
    } while (plan.advance_depth());
    return std::move(plan);
}

/** The audit on graphs written by hand around the entry the seeded draw roots: a part linked one
 *  way is outside the root's component, an entry nothing links to is an unreached island, a pair
 *  linking only to each other is a stranded one, depths are breadth-first whatever order the
 *  tasks run in, and a root drawn inside a small component gives way to the largest. */
template <typename index_type_, typename make_index_type_>
static std::size_t test_ram_index_audit_components(make_index_type_ const& make_index, u64_t seed) {
    using links_t = std::vector<std::vector<node_id_t>>;
    constexpr std::size_t entries = 200;
    node_id_t const root = static_cast<node_id_t>(seeded_bijection(0, entries, 0));
    // Logical entry 0 is the root wherever the draw lands, which swaps places with it.
    auto at = [&](std::size_t logical) {
        return static_cast<node_id_t>(logical == 0 ? root : logical == root ? 0 : logical);
    };
    auto cycle = [&](links_t& links, std::size_t first, std::size_t bound) {
        for (std::size_t entry = first; entry != bound; ++entry)
            links[at(entry)].push_back(at(entry + 1 == bound ? first : entry + 1));
    };
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    auto audit = [&](links_t const& links, auto&& check) {
        std::unique_ptr<index_type_> const index = ram_index_of_links<index_type_>(make_index, ledger, links, seed);
        failures += expect(index != nullptr);
        if (!index) return;
        auto plan = ram_index_audit(*index, ram_tasks_order_t::first_to_last_k);
        auto const [health, status] = plan.finish();
        failures += expect(succeeded(status));
        failures += expect_eq(health.audited_nodes, entries);
        check(plan, health);
    };

    // Two cycles joined one way: the second is reached but never reaches back.
    links_t joined(entries);
    cycle(joined, 0, 150);
    cycle(joined, 150, entries);
    joined[at(10)].push_back(at(160));
    audit(joined, [&](auto const& plan, reachability_t const& health) {
        failures += expect_eq(health.reachable_nodes, 150u);
        failures += expect_eq(health.unreached_islands, 0u);
        failures += expect_eq(health.stranded_islands, 1u);
        failures += expect_eq(health.components, 1u);
        failures += expect(plan.reached(at(170)) && !plan.reaches_root(at(170)));
    });

    // An entry linking into the cycle, which nothing links to.
    links_t unlinked(entries);
    cycle(unlinked, 0, entries - 1);
    unlinked[at(entries - 1)].push_back(at(5));
    audit(unlinked, [&](auto const& plan, reachability_t const& health) {
        failures += expect_eq(health.reachable_nodes, entries - 1);
        failures += expect_eq(health.unreached_islands, 1u);
        failures += expect_eq(health.stranded_islands, 0u);
        failures += expect_eq(health.unlinked, 1u);
        failures += expect_eq(health.underlinked, 1u);
        failures += expect_eq(plan.in_links(at(entries - 1)), 0u);
        failures += expect(!plan.reached(at(entries - 1)) && plan.reaches_root(at(entries - 1)));
    });

    // A pair the cycle links into, linking only to each other.
    links_t stranded(entries);
    cycle(stranded, 0, entries - 2);
    cycle(stranded, entries - 2, entries);
    stranded[at(20)].push_back(at(entries - 2));
    audit(stranded, [&](auto const& plan, reachability_t const& health) {
        failures += expect_eq(health.reachable_nodes, entries - 2);
        failures += expect_eq(health.unreached_islands, 0u);
        failures += expect_eq(health.stranded_islands, 1u);
        failures += expect(plan.reached(at(entries - 1)) && !plan.reaches_root(at(entries - 1)));
    });

    // A root drawn inside a three-entry cycle linking into the rest gives way to the rest.
    links_t nested(entries);
    cycle(nested, 0, 3);
    cycle(nested, 3, entries);
    nested[at(1)].push_back(at(50));
    audit(nested, [&](auto const& plan, reachability_t const& health) {
        failures += expect_eq(health.reachable_nodes, entries - 3);
        failures += expect(!plan.in_component(root) && plan.in_component(at(100)));
        failures += expect_eq(health.unreached_islands, 1u);
        failures += expect_eq(health.stranded_islands, 0u);
        failures += expect_eq(health.components, 1u);
    });

    // Chains run upward through every word, so a sweep expanding what it reached itself would
    // reach them in one depth; breadth-first depths and in-link counts hold either way round.
    links_t chained(entries);
    cycle(chained, 0, entries);
    for (std::size_t entry = 0; entry != entries; ++entry) {
        std::size_t const jump = (entry * entry + 3) % entries;
        if (jump != entry && jump != (entry + 1) % entries) chained[at(entry)].push_back(at(jump));
    }
    std::vector<u8_t> depths(entries, 255);
    std::vector<std::size_t> in_links(entries, 0), frontier {root};
    depths[root] = 0;
    std::size_t deepest = 0;
    for (std::size_t head = 0; head != frontier.size(); ++head)
        for (node_id_t const target : chained[frontier[head]])
            if (depths[target] == 255) {
                depths[target] = static_cast<u8_t>(depths[frontier[head]] + 1);
                deepest = max_of<std::size_t>(deepest, depths[target]);
                frontier.push_back(target);
            }
    for (links_t::value_type const& row : chained)
        for (node_id_t const target : row) ++in_links[target];
    {
        std::unique_ptr<index_type_> const index = ram_index_of_links<index_type_>(make_index, ledger, chained, seed);
        failures += expect(index != nullptr);
        for (ram_tasks_order_t const order : {ram_tasks_order_t::first_to_last_k, ram_tasks_order_t::last_to_first_k}) {
            if (!index) break;
            auto plan = ram_index_audit(*index, order);
            std::size_t wrong = 0;
            for (node_id_t entry = 0; entry != entries; ++entry)
                wrong += plan.depth_from_root(entry) != depths[entry] || plan.in_links(entry) != in_links[entry];
            failures += expect_eq(wrong, 0u);
            failures += expect_eq(plan.finish().value.depth_from_root, deepest);
        }
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** One round over @p index under @p config on the calling thread, @p at_barrier hearing the plan
 *  past every barrier that leaves another phase to run. */
template <typename index_type_, typename at_barrier_type_>
static expected<refine_report_t> ram_index_round(index_type_& index, typename index_type_::refine_config_t config,
                                                 typename index_type_::traversal_context_t& context,
                                                 at_barrier_type_&& at_barrier) {
    auto [plan, status] = index.plan_refine(config);
    if (failed(status)) return {{}, status};
    if (status_t const sized = index.reserve_traversal(context, {index.non_zero_expansion_factor(config)});
        failed(sized))
        return {{}, sized};
    if (status_t const reserved = plan.reserve(context); failed(reserved)) return {{}, reserved};
    while (true) {
        for (auto const task : plan.tasks())
            if (status_t const ran = plan.run(task, context); failed(ran)) return {{}, ran};
        if (!plan.advance()) break;
        at_barrier(plan);
    }
    return plan.finish();
}

/** The neighbor rows of @p index as saved, one per entry. */
template <typename index_type_, typename make_index_type_>
static nk::matrix<node_id_t> ram_index_rows(index_type_& index, make_index_type_ const& make_index) {
    using scalar_t = typename index_type_::scalar_t;
    auto const shape = index.saved_shape();
    auto vectors = nk::matrix<scalar_t>::zeros({shape.nodes, shape.dimensions}).value;
    auto neighbors = nk::matrix<node_id_t>::zeros({shape.nodes, shape.connectivity}).value;
    static_cast<void>(dense_save(index, make_index, {dense_matrix_bytes(vectors), dense_matrix_bytes(neighbors)},
                                 synchronization_t::exclusive_k));
    return neighbors;
}

/** The round's contract, one behaviour a scenario: planted islands end at zero with no distance
 *  spent healing them, a round stopped in any phase leaves searches no worse, a cache of a few rows
 *  gathers the wide neighborhoods again until each fits its capacity, identical entries over
 *  capacity are cut by ID, renumbering the entries changes recall within noise only, and a graph
 *  refined until converged changes no link set on the next round. */
template <typename index_type_, nk_kernel_kind_t kernel_, typename make_index_type_>
static std::size_t test_ram_index_rounds(make_index_type_ const& make_index, u64_t seed) {
    using scalar_t = typename index_type_::scalar_t;
    using key_t = typename index_type_::key_t;
    using refine_config_t = typename index_type_::refine_config_t;
    using links_t = std::vector<std::vector<node_id_t>>;
    std::size_t failures = 0;
    allocation_ledger_t ledger;
    typename index_type_::traversal_context_t context = dense_make_context<index_type_>(make_index);
    std::size_t const dimensions = dense_dimensions_for<scalar_t>(dense_values_k);
    auto ignore_barriers = [](auto&) noexcept {};
    // Draws of sixteen hubs and a capacity of sixteen entries, so neighborhoods gather again.
    std::size_t const row_bytes = index_type_::padded_row_values({dimensions, connectivity_16_k}) * sizeof(scalar_t);
    std::size_t const narrow_capacity = 16;
    refine_config_t narrow;
    narrow.cache_bytes = 2 * narrow_capacity * row_bytes;

    // An entry linking into a cycle that nothing links to, and a pair the cycle links into that
    // links only to itself: the heal links both through the root's component without a distance.
    {
        constexpr std::size_t entries = 200, cycled = entries - 3;
        links_t links(entries);
        for (std::size_t entry = 0; entry != cycled; ++entry) {
            std::size_t const next = (entry + 1) % cycled, jump = (entry * 7 + 3) % cycled;
            links[entry].push_back(static_cast<node_id_t>(next));
            if (jump != entry && jump != next) links[entry].push_back(static_cast<node_id_t>(jump));
        }
        links[cycled] = {5, 50};
        links[cycled + 1] = {cycled + 2};
        links[cycled + 2] = {cycled + 1};
        links[20].push_back(cycled + 1);
        std::unique_ptr<index_type_> const index = ram_index_of_links<index_type_>(make_index, ledger, links, seed);
        failures += expect(index != nullptr);
        if (index) {
            std::size_t healed_from = 0, healing = 0;
            bool in_heal = false;
            auto const [report, status] = ram_index_round(*index, {}, context, [&](auto const& plan) noexcept {
                if (in_heal) healing += context.computed_distances - healed_from;
                in_heal = plan.phase() == phase_t::heal_k;
                healed_from = context.computed_distances;
            });
            failures += expect(succeeded(status));
            failures += expect_eq(report.opening.unreached_islands, 1u);
            failures += expect_eq(report.opening.stranded_islands, 1u);
            failures += expect_eq(report.reachability.unreached_islands, 0u);
            failures += expect_eq(report.reachability.stranded_islands, 0u);
            failures += expect_eq(report.reachability.reachable_nodes, entries);
            failures += expect_eq(healing, 0u);
        }
    }

    std::size_t const built = 600;
    auto vectors = make_vectors<scalar_t>(built, dimensions, stream_key(seed, "rounds"));
    std::vector<key_t> const keys = dense_keys<index_type_>(built);
    auto saved_vectors = nk::matrix<scalar_t>::zeros({built, dimensions}).value;
    auto saved_neighbors = nk::matrix<node_id_t>::zeros({built, static_cast<std::size_t>(connectivity_16_k)}).value;
    {
        std::unique_ptr<index_type_> const index = make_index(ledger);
        failures += expect(succeeded(index->reserve({built}).status));
        typename index_type_::insert_config_t cheap;
        cheap.expansion_factor = 4;
        failures += expect(succeeded(index->insert(to_span(keys), vectors.view(), {}, context, cheap).status));
        failures += expect_eq(
            dense_save(*index, make_index, {dense_matrix_bytes(saved_vectors), dense_matrix_bytes(saved_neighbors)},
                       synchronization_t::exclusive_k),
            status_t::success_k);
    }
    auto load = [&](nk::matrix_view<scalar_t> rows, nk::matrix_view<node_id_t> links, std::span<key_t const> named) {
        std::unique_ptr<index_type_> index = make_index(ledger);
        failures += expect(succeeded(index->reserve({built}).status));
        failures += expect_eq(dense_load(*index, make_index, {dense_matrix_bytes(rows), dense_matrix_bytes(links)},
                                         named, built, {0, static_cast<node_id_t>(built)}),
                              status_t::success_k);
        return index;
    };
    nk::matrix_view<scalar_t> const rows = vectors.view();
    auto queries = rows(nk::range(0, 64), nk::slice);

    // Stopped at the barrier opening each phase, from the opening health check to the restitch, the
    // round keeps what ran and no search does worse for it.
    if constexpr (kernel_ != nk_kernel_dot_k)
        for (std::size_t phase = 1; phase != 6; ++phase) {
            std::unique_ptr<index_type_> const index = load(saved_vectors.view(), saved_neighbors.view(),
                                                            to_span(keys));
            std::size_t const before = dense_recall_at_ten(*index, make_index, queries);
            bool stopped = false;
            auto const [report, status] = ram_index_round(*index, narrow, context, [&](auto& plan) noexcept {
                if (stopped || plan.ordinal() != phase) return;
                plan.stop();
                stopped = true;
            });
            failures += expect(stopped);
            failures += expect_eq(status, status_t::cancelled_k);
            failures += expect(report.verdict == refine_verdict_t::stopped_k);
            failures += expect(dense_recall_at_ten(*index, make_index, queries) >= before);
        }

    // A cache of a few rows puts the first neighborhoods over capacity, which gather again
    // among their own entries until none is.
    {
        std::unique_ptr<index_type_> const index = load(saved_vectors.view(), saved_neighbors.view(), to_span(keys));
        std::size_t gathers = 0;
        auto const [report, status] = ram_index_round(
            *index, narrow, context, [&](auto const& plan) noexcept { gathers += plan.phase() == phase_t::gather_k; });
        failures += expect(succeeded(status));
        failures += expect(gathers >= 2);
        failures += expect_eq(report.reselected_nodes, built);
        failures += expect(report.neighborhoods > 1 && report.widest_neighborhood != 0);
        if constexpr (!index_type_::has_tiles_k) failures += expect(report.widest_neighborhood <= narrow_capacity);
        failures += expect(report.reachability.reachable_nodes >= report.opening.reachable_nodes);
    }

    // Identical entries all lie nearest to one hub, so no gather splits them; they are cut into
    // pieces by ID instead, and the round ends.
    {
        constexpr std::size_t entries = 10 * narrow_capacity;
        auto copies = nk::matrix<scalar_t>::zeros({entries, dimensions}).value;
        for (std::size_t entry = 0; entry != entries; ++entry) {
            auto const source = vectors.view().row(0).as_vector();
            auto target = copies.row(entry).as_vector();
            for (std::size_t dimension = 0; dimension != dimensions; ++dimension) target[dimension] = source[dimension];
        }
        std::unique_ptr<index_type_> const index = make_index(ledger);
        failures += expect(succeeded(index->reserve({entries}).status));
        failures += expect(succeeded(index->insert(std::span(keys).first(entries), copies.view(), {}, context).status));
        auto const [report, status] = ram_index_round(*index, narrow, context, ignore_barriers);
        failures += expect(succeeded(status));
        failures += expect_eq(report.reselected_nodes, entries);
        if constexpr (!index_type_::has_tiles_k) failures += expect(report.widest_neighborhood <= narrow_capacity);
        failures += expect(report.reachability.reachable_nodes >= report.opening.reachable_nodes);
    }

    // The same graph under shuffled IDs draws other entries but reaches recall within noise.
    if constexpr (kernel_ != nk_kernel_dot_k) {
        auto shuffled_vectors = nk::matrix<scalar_t>::zeros({built, dimensions}).value;
        auto shuffled_neighbors =
            nk::matrix<node_id_t>::zeros({built, static_cast<std::size_t>(connectivity_16_k)}).value;
        std::vector<key_t> shuffled_keys(built);
        nk::matrix_view<node_id_t> const saved_links = saved_neighbors.view();
        nk::matrix_span<node_id_t> const shuffled_links = shuffled_neighbors.span();
        auto moved = [&](node_id_t entry) {
            return static_cast<node_id_t>(seeded_bijection(entry, built, stream_key(seed, "shuffle")));
        };
        for (node_id_t entry = 0; entry != built; ++entry) {
            node_id_t const into = moved(entry);
            shuffled_keys[into] = keys[entry];
            auto const source = saved_vectors.view().row(entry).as_vector();
            auto target = shuffled_vectors.row(into).as_vector();
            for (std::size_t dimension = 0; dimension != dimensions; ++dimension) target[dimension] = source[dimension];
            for (std::size_t slot = 0; slot != connectivity_16_k; ++slot) {
                node_id_t const link = saved_links(entry, slot);
                shuffled_links(into, slot) = link == node_id_unknown_k ? link : moved(link);
            }
        }
        std::unique_ptr<index_type_> const plain = load(saved_vectors.view(), saved_neighbors.view(), to_span(keys));
        std::unique_ptr<index_type_> const shuffled = load(shuffled_vectors.view(), shuffled_neighbors.view(),
                                                           to_span(shuffled_keys));
        failures += expect(succeeded(ram_index_round(*plain, narrow, context, ignore_barriers).status));
        failures += expect(succeeded(ram_index_round(*shuffled, narrow, context, ignore_barriers).status));
        std::size_t const plain_found = dense_recall_at_ten(*plain, make_index, queries);
        std::size_t const shuffled_found = dense_recall_at_ten(*shuffled, make_index, queries);
        std::size_t const noise = queries.extent(0) * 10 / 20;
        failures += expect(plain_found + noise >= shuffled_found && shuffled_found + noise >= plain_found);
    }

    // Seventeen entries in one neighborhood: rounds run until one changes no link set, and the next
    // one writes nothing and answers converged again.
    {
        constexpr std::size_t entries = connectivity_16_k + 1, rounds = 8;
        std::unique_ptr<index_type_> const index = make_index(ledger);
        failures += expect(succeeded(index->reserve({entries}).status));
        failures += expect(succeeded(
            index->insert(std::span(keys).first(entries), rows(nk::range(0, entries), nk::slice), {}, context).status));
        bool converged = false;
        for (std::size_t round = 0; round != rounds && !converged; ++round)
            converged = ram_index_round(*index, {}, context, ignore_barriers).value.verdict ==
                        refine_verdict_t::converged_k;
        failures += expect(converged);
        nk::matrix<node_id_t> const before = ram_index_rows(*index, make_index);
        auto const [report, status] = ram_index_round(*index, {}, context, ignore_barriers);
        nk::matrix<node_id_t> const after = ram_index_rows(*index, make_index);
        nk::matrix_view<node_id_t> const links_before = before.view(), links_after = after.view();
        failures += expect(succeeded(status));
        failures += expect(report.verdict == refine_verdict_t::converged_k);
        failures += expect_eq(report.changed_nodes, 0u);
        failures += expect_eq(report.added_links + report.dropped_links, 0u);
        std::size_t differing = 0;
        for (std::size_t entry = 0; entry != entries; ++entry)
            for (std::size_t slot = 0; slot != connectivity_16_k; ++slot) {
                bool found = false;
                for (std::size_t other = 0; other != connectivity_16_k; ++other)
                    found |= links_before(entry, slot) == links_after(entry, other);
                differing += !found;
            }
        failures += expect_eq(differing, 0u);
    }
    return failures + expect_eq(ledger.live_bytes, 0);
}

/** No-op sibling picked when the cell cannot be compiled on this target: its serial
 *  kernels are not adapted yet, or the requested kit is absent from the build. */
template <                                        //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,  //
    typename waiter_type_ = fu::standard_yield_t, //
    nk_capability_t caps_ = nk_cap_serial_k,      //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
static std::size_t test_ram_index_configuration(char const* name, u64_t) {
    if constexpr (!quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>)
        return report(name, "serial kernels not adapted yet");
    else return report(name, "kit not compiled for this target");
}

template <                                        //
    nk_dtype_t dtype_, nk_kernel_kind_t kernel_,  //
    typename waiter_type_ = fu::standard_yield_t, //
    nk_capability_t caps_ = nk_cap_serial_k,      //
    template <typename> class atomic_ref_ = fu::standard_atomic_ref>
    requires dense_cell_testable<dtype_, kernel_, caps_>
static std::size_t test_ram_index_configuration(char const* name, u64_t seed) {

    using index_t = ram_index<typename quads_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                              typename tiles_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                              typename elementwise_ladder<caps_>::kit_t,
                              typename rmscast_ladder<typename nk::type_for<dtype_>::type, caps_>::kit_t,
                              typename visits_hash_set_ladder<caps_>::kit_t, waiter_type_, atomic_ref_,
                              dummy_cache_hints_t, counting_allocator_t, counting_allocator_t, std::allocator<byte_t>>;

    if (!dense_cell_admitted<caps_, atomic_ref_>()) return report(name, "capabilities missing at runtime");

    using geometry_t = typename index_t::geometry_t;
    using scalar_t = typename index_t::scalar_t;
    auto make_index = [](allocation_ledger_t& ledger,
                         geometry_t geometry = {dense_dimensions_for<scalar_t>(dense_values_k), connectivity_16_k}) {
        return std::unique_ptr<index_t>(
            new (std::nothrow) index_t(kernel_, geometry, counting_allocator_t(ledger), counting_allocator_t(ledger)));
    };

    std::size_t failures_count = 0;
    failures_count += test_ram_index_keys<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_guards<index_t>(make_index, seed);
    failures_count += test_dense_three_vectors<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_growth_and_recall<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_refine_settles<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_duplicate_basin<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_health_audits<index_t, kernel_>(make_index, seed);
    failures_count += test_ram_index_audit_components<index_t>(make_index, seed);
    failures_count += test_ram_index_rounds<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_heal_islands<index_t>(make_index, seed);
    failures_count += test_dense_round_in_place<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_cut<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_predicated_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_scan<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_multiply<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_attend<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_indexer_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_save_load_roundtrip<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_load_hostile<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_load_append<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_build<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_concurrent_search<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_save_race<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_immutable_parity<index_t, kernel_>(make_index, seed);
    using concurrent_cursor_t = ram_cursor<index_t, synchronization_t::concurrent_k, //
                                           counting_allocator_t>;
    using immutable_cursor_t = ram_cursor<index_t, synchronization_t::immutable_k, //
                                          counting_allocator_t>;
    failures_count += test_dense_cursor<index_t, concurrent_cursor_t, immutable_cursor_t>(make_index, seed);
    failures_count += test_dense_allocation_budget<index_t, kernel_>(make_index, seed);
    failures_count += test_dense_frontier_shedding<index_t>(make_index, seed);
    failures_count += test_dense_overflow_race<index_t>(make_index, seed);
    return report(name, failures_count);
}

/** The starved cell: the same index over ledgered traversal scratch, so a refused re-table
 *  has somewhere to come from, walked by the saturation scenario alone. */
template <nk_dtype_t dtype_, nk_kernel_kind_t kernel_>
    requires quads_supported<kernel_, typename nk::type_for<dtype_>::type, nk_cap_serial_k>
static std::size_t test_starved_configuration(char const* name, u64_t seed) {
    using index_t = ram_index<typename quads_ladder<typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t,
                              typename tiles_ladder<typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t,
                              typename elementwise_ladder<nk_cap_serial_k>::kit_t,
                              typename rmscast_ladder<typename nk::type_for<dtype_>::type, nk_cap_serial_k>::kit_t,
                              typename visits_hash_set_ladder<nk_cap_serial_k>::kit_t, fu::standard_yield_t,
                              fu::standard_atomic_ref, dummy_cache_hints_t, counting_allocator_t, counting_allocator_t,
                              counting_allocator_t>;
    using geometry_t = typename index_t::geometry_t;
    using scalar_t = typename index_t::scalar_t;
    auto make_index = [](allocation_ledger_t& ledger,
                         geometry_t geometry = {dense_dimensions_for<scalar_t>(dense_values_k), connectivity_16_k}) {
        return std::unique_ptr<index_t>(
            new (std::nothrow) index_t(kernel_, geometry, counting_allocator_t(ledger), counting_allocator_t(ledger)));
    };
    return report(
        name, test_dense_saturated<index_t>(make_index, seed) + test_dense_refine_atomic<index_t>(make_index, seed));
}

} // namespace unum::usearch::test::dense
