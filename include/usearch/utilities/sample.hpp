/**
 *  @file include/usearch/utilities/sample.hpp
 *  @author Ash Vardanian
 *  @date October 29, 2026
 *  @brief The plan behind every tower's SAMPLE verb: roots, the paths between them, the regions
 *      around them, and three coordinates for every draw, read through a source over the tower's
 *      walk and key map.
 *
 *  A draw names an entry: its key and its ordinal among that key's entries, as @c get orders them.
 *  The plan never scans the index, and never reads how many draws are wanted except to stop, so a
 *  smaller call draws a prefix of a larger one. It draws in passes of a doubling quota. Its roots
 *  are the focus keys' entries, or with none the next entries through @ref seeded_bijection. Greedy
 *  walks attach each new root to its nearest earlier one and run from it toward its nearest shown
 *  point. A region over links around each root, bounded by the distance to its nearest tree
 *  neighbor, draws what ranks lowest by @ref splitmix64_at of the seed and the key, so a narrower
 *  region keeps the draws a wider one made.
 *
 *  The coordinates keep the shown points where the caller drew them. Every draw sits where its
 *  distances to its nearest shown points best match, by @ref sample_gauss_newton. Without shown
 *  points, a dense source projects each entry's values through @ref sample_projection, a source
 *  with fixed coordinates answers its own, and any other source scales its first draws classically
 *  and solves the rest among them. Both helpers are plain loops with contractions off, so a
 *  stand-in copying them verbatim reaches the same bits, even under a build's fast-math flags.
 */
#pragma once

#include <cmath>   // `std::sqrt`, `std::log`, `std::cos`, `std::isfinite`
#include <cstring> // `std::memset`

#include <array>      // `std::array` - a solve's reference points, a walk's staged neighbors
#include <bit>        // `std::bit_ceil` - the probe tables' widths
#include <concepts>   // `std::same_as`, `std::convertible_to`
#include <functional> // `std::hash` - the keys' ranks and probes
#include <limits>     // `std::numeric_limits`
#include <numbers>    // `std::numbers::pi`
#include <span>       // `std::span`

#include "usearch/utilities/types.hpp"     // `seeded_bijection`, `mix_bits`, `shard_node`
#include "usearch/utilities/ram_sorts.hpp" // `partial_sort` - a region's lowest ranks

namespace unum::usearch::utilities {

/** Shown points each draw solves its coordinates against, at most. */
inline constexpr std::size_t sample_nearest_shown_k = 8;

/** Gauss-Newton steps one draw's coordinates take. */
inline constexpr std::size_t sample_gauss_newton_steps_k = 8;

/** Draws a classical scaling spans; later draws are solved among them. */
inline constexpr std::size_t sample_scaling_draws_k = 32;

/** Entries a region visits per draw it may make, when the call names no breadth. */
inline constexpr std::size_t sample_expansion_factor_k = 16;

/** The knobs of one SAMPLE call. */
struct sample_config_t {

    /** Names the roots, the ranks and the projection; equal seeds over an unchanged index draw
     *  equally. */
    u64_t seed = 0;

    /** Entries a region visits per draw it may make; zero means @ref sample_expansion_factor_k. */
    std::size_t expansion_factor = 0;
};

/** What a SAMPLE call reads: the shown points, where they are shown, and where to look. */
template <typename key_type_>
struct sample_inputs {

    /** Shown points: never drawn again, the paths reach toward them, and the coordinates keep
     *  them. */
    std::span<key_type_ const> shown;

    /** @b [shown,3] where each shown point is shown. */
    std::span<f32_t const> shown_coordinates;

    /** Keys to look around; empty means the whole index. */
    std::span<key_type_ const> focus;
};

/** What a SAMPLE call writes, one value per draw; the keys' length is the draws wanted. */
template <typename key_type_>
struct sample_outputs {

    /** The key of each draw. */
    std::span<key_type_> keys;

    /** Which of its key's entries each draw is, as @c get orders them. */
    std::span<std::size_t> entry_ordinals;

    /** The shown point or earlier draw each draw hangs from; the first draw of a call without
     *  shown points hangs from itself. */
    std::span<key_type_> parents;

    /** Each draw's distance to its parent. */
    std::span<f32_t> parent_distances;

    /** @b [keys,3] each draw's coordinates. */
    std::span<f32_t> coordinates;
};

/** What a SAMPLE call reports. */
struct sample_result_t {

    /** Draws written; fewer than wanted once the regions ran dry. */
    std::size_t written = 0;

    /** Entries whose links a region expanded. */
    std::size_t visited_members = 0;

    /** Distances the plan computed, a source's walks excluded. */
    std::size_t computed_distances = 0;
};

/** What a source without fixed coordinates takes, so it solves its draws' coordinates from
 *  distances. */
struct no_fixed_coordinates_t {};

/** Whether @p coordinates_type_ fixes each entry's coordinates, so the plan reads them instead of
 *  solving from distances; a callable of the wrong signature then fails to compile rather than
 *  falling back silently. */
template <typename coordinates_type_>
concept fixed_coordinates = !std::same_as<coordinates_type_, no_fixed_coordinates_t>;

/** SplitMix64's output at @p counter of the stream @p seed starts: what @c splitmix64_t {seed}
 *  answers after discarding @p counter outputs, its state @p counter + 1 steps on, mixed. */
constexpr u64_t splitmix64_at(u64_t seed, u64_t counter) noexcept {
    return mix_bits(seed + (counter + 1) * golden_ratio_u64_k);
}

/** The top 53 bits of @p bits as a double in (0, 1]: ((bits >> 11) + 1) · 2⁻⁵³. */
constexpr f64_t uniform_unit_f64(u64_t bits) noexcept { return static_cast<f64_t>((bits >> 11) + 1) * 0x1p-53; }

/**
 *  @brief Projects one entry's values to three coordinates through a dense Gaussian matrix no one
 *      stores: coordinates[c] = Σ_d values[d] · G[d][c] / sqrt(dimensions).
 *
 *  G[d][c], with n = 3d + c, is the Box-Muller cosine sqrt(−2 ln u₁) · cos(2π u₂) in f64. Its u₁ is
 *  @ref uniform_unit_f64 of @ref splitmix64_at (seed, 2n) and u₂ the same at 2n + 1. The sums run
 *  in f64, d ascending then c, each adding (value · sqrt(−2 ln u₁)) · cos(2π u₂); each divides by
 *  sqrt(dimensions) in f64 and rounds to f32 once. Values of no dimensions project to the origin.
 */
inline void sample_projection(f32_t const* values, std::size_t dimensions, u64_t seed, f32_t* coordinates) noexcept {
#pragma float_control(precise, on)
#pragma clang fp contract(off)
    std::array<f64_t, 3> sums {0, 0, 0};
    for (std::size_t dimension = 0; dimension != dimensions; ++dimension) {
        f64_t const value = values[dimension];
        for (std::size_t column = 0; column != 3; ++column) {
            u64_t const counter = 2 * (3 * static_cast<u64_t>(dimension) + column);
            f64_t const radius = std::sqrt(-2.0 * std::log(uniform_unit_f64(splitmix64_at(seed, counter))));
            f64_t const angle = 2.0 * std::numbers::pi * uniform_unit_f64(splitmix64_at(seed, counter + 1));
            sums[column] += value * radius * std::cos(angle);
        }
    }
    f64_t const root = std::sqrt(static_cast<f64_t>(dimensions));
    for (std::size_t column = 0; column != 3; ++column)
        coordinates[column] = dimensions ? static_cast<f32_t>(sums[column] / root) : 0.0f;
}

/**
 *  @brief Solves one entry's coordinates where its distances to @p references_count reference
 *      points best match the @p distances in relative error, Σ_l (‖p − y_l‖ − d_l)² / d_l², by
 *      @ref sample_gauss_newton_steps_k damped steps of Gauss-Newton.
 *
 *  @param[in] references @b [references_count,3] the reference points' coordinates.
 *  @param[in] distances @b [references_count] the entry's distances to them, in coordinate scale.
 *  @param[in] references_count Reference points, at least one.
 *  @param[in] key_seed The entry's own seed, which nudges the start off the reference points.
 *  @param[out] coordinates The entry's three coordinates.
 *
 *  A distance that is not positive puts the entry on the first such reference point. Otherwise it
 *  runs in f32, reference points ascending:
 *  1. p starts at the mean of the reference points weighted by 1 / d_l.
 *  2. p moves by 0.01 · min_l d_l · (2u_c − 1), u_c = @ref uniform_unit_f64 of @ref splitmix64_at
 *     over @p key_seed and c, rounded to f32.
 *  3. Each step sums H = Σ J_lᵀJ_l and g = Σ J_lᵀ e_l over the reference points p is off, with J_l
 *     = (p − y_l) / (‖p − y_l‖ · d_l) and e_l = (‖p − y_l‖ − d_l) / d_l.
 *  4. It solves (H + λI) s = −g, λ = 0.001 · trace(H), by Cramer's rule, and moves p by s; a
 *     determinant that is zero or not finite stops the steps.
 */
inline void sample_gauss_newton(f32_t const* references, f32_t const* distances, std::size_t references_count,
                                u64_t key_seed, f32_t* coordinates) noexcept {
#pragma float_control(precise, on)
#pragma clang fp contract(off)
    for (std::size_t reference = 0; reference != references_count; ++reference) {
        if (distances[reference] > 0.0f) continue;
        for (std::size_t column = 0; column != 3; ++column) coordinates[column] = references[reference * 3 + column];
        return;
    }
    f32_t weights = 0.0f, nearest = distances[0];
    std::array<f32_t, 3> point {0.0f, 0.0f, 0.0f};
    for (std::size_t reference = 0; reference != references_count; ++reference) {
        f32_t const weight = 1.0f / distances[reference];
        weights += weight;
        for (std::size_t column = 0; column != 3; ++column)
            point[column] += weight * references[reference * 3 + column];
        nearest = min_of(nearest, distances[reference]);
    }
    for (std::size_t column = 0; column != 3; ++column) {
        f32_t const unit = static_cast<f32_t>(uniform_unit_f64(splitmix64_at(key_seed, column)));
        point[column] = point[column] / weights + 0.01f * nearest * (2.0f * unit - 1.0f);
    }

    for (std::size_t step = 0; step != sample_gauss_newton_steps_k; ++step) {
        f32_t hessian_00 = 0.0f, hessian_01 = 0.0f, hessian_02 = 0.0f, hessian_11 = 0.0f, hessian_12 = 0.0f,
              hessian_22 = 0.0f;
        f32_t gradient_0 = 0.0f, gradient_1 = 0.0f, gradient_2 = 0.0f;
        for (std::size_t reference = 0; reference != references_count; ++reference) {
            f32_t const offset_0 = point[0] - references[reference * 3 + 0];
            f32_t const offset_1 = point[1] - references[reference * 3 + 1];
            f32_t const offset_2 = point[2] - references[reference * 3 + 2];
            f32_t const reach = std::sqrt(offset_0 * offset_0 + offset_1 * offset_1 + offset_2 * offset_2);
            if (!(reach > 0.0f)) continue;
            f32_t const wanted = distances[reference];
            f32_t const scale = 1.0f / (reach * wanted);
            f32_t const jacobian_0 = offset_0 * scale, jacobian_1 = offset_1 * scale, jacobian_2 = offset_2 * scale;
            f32_t const error = (reach - wanted) / wanted;
            hessian_00 += jacobian_0 * jacobian_0, hessian_01 += jacobian_0 * jacobian_1,
                hessian_02 += jacobian_0 * jacobian_2, hessian_11 += jacobian_1 * jacobian_1,
                hessian_12 += jacobian_1 * jacobian_2, hessian_22 += jacobian_2 * jacobian_2;
            gradient_0 += jacobian_0 * error, gradient_1 += jacobian_1 * error, gradient_2 += jacobian_2 * error;
        }
        f32_t const damping = 0.001f * (hessian_00 + hessian_11 + hessian_22);
        f32_t const damped_00 = hessian_00 + damping, damped_11 = hessian_11 + damping,
                    damped_22 = hessian_22 + damping;
        f32_t const cofactor_0 = damped_11 * damped_22 - hessian_12 * hessian_12,
                    cofactor_1 = hessian_01 * damped_22 - hessian_12 * hessian_02,
                    cofactor_2 = hessian_01 * hessian_12 - damped_11 * hessian_02;
        f32_t const determinant = damped_00 * cofactor_0 - hessian_01 * cofactor_1 + hessian_02 * cofactor_2;
        if (determinant == 0.0f || !std::isfinite(determinant)) break;
        f32_t const negative_gradient_0 = -gradient_0, negative_gradient_1 = -gradient_1,
                    negative_gradient_2 = -gradient_2;
        f32_t const step_0 = (negative_gradient_0 * cofactor_0 -
                              hessian_01 * (negative_gradient_1 * damped_22 - hessian_12 * negative_gradient_2) +
                              hessian_02 * (negative_gradient_1 * hessian_12 - damped_11 * negative_gradient_2)) /
                             determinant;
        f32_t const step_1 = (damped_00 * (negative_gradient_1 * damped_22 - hessian_12 * negative_gradient_2) -
                              negative_gradient_0 * cofactor_1 +
                              hessian_02 * (hessian_01 * negative_gradient_2 - negative_gradient_1 * hessian_02)) /
                             determinant;
        f32_t const step_2 = (damped_00 * (damped_11 * negative_gradient_2 - negative_gradient_1 * hessian_12) -
                              hessian_01 * (hessian_01 * negative_gradient_2 - negative_gradient_1 * hessian_02) +
                              negative_gradient_0 * cofactor_2) /
                             determinant;
        point[0] += step_0, point[1] += step_1, point[2] += step_2;
    }
    for (std::size_t column = 0; column != 3; ++column) coordinates[column] = point[column];
}

/** A callback that ignores every node, spelling the callbacks @ref sample_source checks. */
struct ignore_nodes_t {
    template <typename node_id_type_>
    void operator()(shard_node<node_id_type_>) const noexcept {}
};

/**
 *  @brief One tower as SAMPLE reads it, below one snapshot of its watermarks.
 *
 *  - @c watermarks() answers one watermark per shard, the entries a draw may name.
 *  - @c key_of(node) answers the key an entry is stored under.
 *  - <tt>nodes_of(key, callback)</tt> calls @c callback(node) for each entry under @p key, as @c get
 *    orders them: shard by shard, insertion order within each.
 *  - <tt>neighbors(node, callback)</tt> calls @c callback(next) for each link of @p node below the
 *    snapshot, holding no lock while it does.
 *  - <tt>distance(from, nodes, distances)</tt> writes the distance from @p from to each of @p nodes.
 *  - <tt>walk(from, toward, chain)</tt> walks greedily from @p from toward @p toward, writing each
 *    entry it steps on into @p chain, and answers the steps.
 *
 *  An optional <tt>ordinals_of(keys, nodes, ordinals) noexcept → status_t</tt> writes each node's
 *  ordinal among its key's entries in one batch, for a source whose @c nodes_of costs a scan; the
 *  plan then calls @c nodes_of only for the shown and focus keys.
 *
 *  Two optional members choose the coordinates without shown points:
 *  <tt>coordinates_of(node, coordinates)</tt> writes fixed coordinates, and <tt>values_of(node)</tt>
 *  answers a dense entry's values as f32 for the projection to read.
 */
template <typename source_type_>
concept sample_source = requires(
    source_type_ const& source, shard_node<typename source_type_::node_id_t> node,
    typename source_type_::key_t const& key, std::span<shard_node<typename source_type_::node_id_t> const> nodes,
    std::span<f32_t> distances, std::span<shard_node<typename source_type_::node_id_t>> chain, ignore_nodes_t sink) {
    { source.watermarks() } noexcept -> std::convertible_to<std::span<typename source_type_::node_id_t const>>;
    { source.key_of(node) } noexcept -> std::convertible_to<typename source_type_::key_t>;
    { source.nodes_of(key, sink) } noexcept;
    { source.neighbors(node, sink) } noexcept;
    { source.distance(node, nodes, distances) } noexcept;
    { source.walk(node, node, chain) } noexcept -> std::same_as<std::size_t>;
};

/** The greedy walk a RAM source answers @c walk with: from @p from, step to the link nearest
 *  @p toward while it comes closer, until @p chain fills or the walk arrives. */
template <typename source_type_>
std::size_t sample_greedy_walk(source_type_ const& source, shard_node<typename source_type_::node_id_t> from,
                               shard_node<typename source_type_::node_id_t> toward,
                               std::span<shard_node<typename source_type_::node_id_t>> chain) noexcept {
    using node_id_t = typename source_type_::node_id_t;
    using shard_node_t = shard_node<node_id_t>;
    std::array<shard_node_t, 64> staged;
    std::array<f32_t, 64> staged_distances;
    f32_t current_distance = 0;
    source.distance(toward, std::span<shard_node_t const>(&from, 1), std::span<f32_t>(&current_distance, 1));
    shard_node_t current = from;
    std::size_t length = 0;
    while (length != chain.size() && current != toward) {
        shard_node_t best = current;
        f32_t best_distance = current_distance;
        std::size_t filled = 0;
        auto const score = [&]() noexcept {
            source.distance(toward, std::span<shard_node_t const>(staged.data(), filled),
                            std::span<f32_t>(staged_distances.data(), filled));
            for (std::size_t lane = 0; lane != filled; ++lane)
                if (staged_distances[lane] < best_distance) best = staged[lane], best_distance = staged_distances[lane];
            filled = 0;
        };
        source.neighbors(current, [&](shard_node_t next) noexcept {
            staged[filled++] = next;
            if (filled == staged.size()) score();
        });
        score();
        if (best == current) break;
        chain[length++] = best;
        current = best;
        current_distance = best_distance;
    }
    return length;
}

/**
 *  @brief One SAMPLE call over a @ref sample_source, its steps run in order: @ref draw, then
 *      @ref assign_ordinals, then @ref assign_coordinates.
 *
 *  A one-call plan: unlike the resumable refine, load and health plans, it lives inside a single
 *  SAMPLE call and has no @c plan_sample opening it. Every buffer is carved once from the
 *  caller's scratch of @ref scratch_bytes, so nothing allocates per draw. Nearest-shown distances
 *  take up to @ref sample_nearest_shown_k per draw, but no more than the shown points; a region's
 *  visits take its breadth per draw; the classical scaling's matrix is carved only for a source
 *  that solves by it.
 */
template <typename source_type_>
    requires sample_source<source_type_>
struct sample_plan {
    using source_t = source_type_;
    using key_t = typename source_t::key_t;
    using node_id_t = typename source_t::node_id_t;
    using shard_node_t = shard_node<node_id_t>;
    using sample_inputs_t = sample_inputs<key_t>;
    using sample_outputs_t = sample_outputs<key_t>;

    /** Whether the source solves coordinates by classical scaling when nothing is shown: it fixes
     *  no coordinates and projects no values. */
    static constexpr bool scales_classically_k = !requires(source_t const& source, std::span<f32_t, 3> coordinates) {
        source.coordinates_of(shard_node_t {}, coordinates);
    } && !requires(source_t const& source) { source.values_of(shard_node_t {}); };

    /** Scratch bytes a call of @p wanted draws around @p shown_count shown points takes. */
    static std::size_t scratch_bytes(std::size_t wanted, std::size_t shown_count, sample_config_t config) noexcept {
        sample_plan measured {nullptr, {}, {}, config, {}, wanted, shown_count};
        measured.carve_all_();
        return measured.carved_bytes_;
    }

    /** A plan over @p source writing @p outputs, carving @p scratch, aligned to
     *  @c default_alignment_k, at its first step. */
    sample_plan(source_t const& source, sample_inputs_t inputs, sample_outputs_t outputs, sample_config_t config,
                bytes_span_t scratch) noexcept
        : sample_plan(&source, inputs, outputs, config, scratch, outputs.keys.size(), inputs.shown.size()) {}

    /**
     *  @brief Resolves the shown points, then draws in passes until the outputs fill or a pass
     *      draws nothing more.
     *
     *  A pass of quota q, one at first and doubling after, but never past one more than the draws
     *  so far:
     *  1. files roots, drawing each: every focus entry in the first pass, a shown focus key once at
     *     its first entry, or without focus the next entries through @ref seeded_bijection that
     *     draw, until q are filed;
     *  2. walks to each new root from its nearest earlier one, then from each new root toward its
     *     nearest shown point, each step drawing and hanging from the step before;
     *  3. draws around every root up to q more of what a draw region over its links reaches within
     *     the distance to its nearest tree neighbor, lowest rank first.
     *
     *  Nothing reads the draws wanted but the stop, so a call wanting fewer draws a prefix of one
     *  that wants more.
     *
     *  @par Statuses:
     *  - @c shape_mismatch_k An output's or the shown coordinates' length disagrees.
     *  - @c bad_alloc_k The scratch is shorter than @ref scratch_bytes.
     */
    status_t draw() noexcept {
        if (outputs_.entry_ordinals.size() != wanted_ || outputs_.parents.size() != wanted_ ||
            outputs_.parent_distances.size() != wanted_ || outputs_.coordinates.size() != wanted_ * 3 ||
            inputs_.shown_coordinates.size() != shown_count_ * 3)
            return status_t::shape_mismatch_k;
        carve_all_();
        if (carved_bytes_ > reserved_scratch_.size()) return status_t::bad_alloc_k;
        std::memset(drawn_slots_.data(), 0, drawn_slots_.size_bytes());
        std::memset(shown_slots_.data(), 0, shown_slots_.size_bytes());
        std::memset(rooted_shown_.data(), 0, rooted_shown_.size_bytes());

        for (std::size_t shown = 0; shown != shown_count_; ++shown) {
            if (!claim_shown_(shown)) continue;
            shard_node_t first;
            std::size_t entries = 0;
            source_->nodes_of(inputs_.shown[shown], [&](shard_node_t node) noexcept {
                if (!entries++) first = node;
            });
            if (!entries) continue;
            shown_nodes_[resolved_shown_count_] = first;
            shown_indices_[resolved_shown_count_] = static_cast<u32_t>(shown);
            ++resolved_shown_count_;
        }
        nearest_shown_count_ = min_of(nearest_shown_stride_, resolved_shown_count_);
        if (!wanted_) return status_t::success_k;

        std::span<node_id_t const> const watermarks = source_->watermarks();
        u64_t total = 0, drawn_so_far = 0, generation = 0;
        for (node_id_t const watermark : watermarks) total += watermark;
        if (!inputs_.focus.empty()) file_focus_();
        for (std::size_t quota = 1, attached_roots = 0;;) {
            std::size_t const before = written_;
            if (inputs_.focus.empty())
                while (roots_count_ != quota && drawn_so_far != total && written_ != wanted_) {
                    u64_t entry = seeded_bijection(drawn_so_far++, total, config_.seed);
                    u32_t shard = 0;
                    while (entry >= watermarks[shard]) entry -= watermarks[shard++];
                    shard_node_t const node {shard, static_cast<node_id_t>(entry)};
                    if (u32_t const draw = draw_nearest_(node); draw != none_k) file_root_(node, draw);
                }
            std::size_t const first_new = attached_roots;
            for (; attached_roots != roots_count_; ++attached_roots) attach_root_(attached_roots);
            if (nearest_shown_count_)
                for (std::size_t root = first_new; root != roots_count_; ++root)
                    if (root_draws_[root] != none_k)
                        walk_between_(roots_[root],
                                      shown_nodes_[nearest_shown_indices_[root_draws_[root] * nearest_shown_stride_]]);
            for (std::size_t root = 0; root != roots_count_ && written_ != wanted_; ++root)
                draw_region_(roots_[root], root_radii_[root], quota, ++generation);
            if (written_ == wanted_) break;
            // A quota that cannot grow over a pass that drew nothing would repeat that pass.
            std::size_t const next = min_of(2 * quota, written_ + 1);
            if (written_ == before && next == quota) break;
            quota = next;
        }
        return status_t::success_k;
    }

    /** Writes which of its key's entries each draw is, as @c get orders them: in one batch where
     *  the source counts them through @c ordinals_of, else through @c nodes_of draw by draw. */
    status_t assign_ordinals() noexcept {
        std::span<key_t const> const keys = outputs_.keys.first(written_);
        std::span<shard_node_t const> const drawn = drawn_.first(written_);
        std::span<std::size_t> const ordinals = outputs_.entry_ordinals.first(written_);
        if constexpr (requires { source_->ordinals_of(keys, drawn, ordinals); })
            return source_->ordinals_of(keys, drawn, ordinals);
        else {
            for (std::size_t draw = 0; draw != written_; ++draw) {
                std::size_t ordinal = 0, entries = 0;
                source_->nodes_of(keys[draw], [&](shard_node_t entry) noexcept {
                    if (entry == drawn[draw]) ordinal = entries;
                    ++entries;
                });
                ordinals[draw] = ordinal;
            }
            return status_t::success_k;
        }
    }

    /** Writes every draw's coordinates: its fixed ones, or those solved among its nearest shown
     *  points, or without any its projection or its solve in a classical scaling. */
    status_t assign_coordinates() noexcept {
        if constexpr (requires(std::span<f32_t, 3> coordinates) {
                          source_->coordinates_of(shard_node_t {}, coordinates);
                      })
            for (std::size_t draw = 0; draw != written_; ++draw)
                source_->coordinates_of(drawn_[draw], coordinates_(draw));
        else {
            if (nearest_shown_count_) {
                f32_t const scale = shown_scale_();
                for (std::size_t draw = 0; draw != written_; ++draw) place_among_shown_(draw, scale);
            }
            else if constexpr (requires { source_->values_of(shard_node_t {}); })
                for (std::size_t draw = 0; draw != written_; ++draw) {
                    std::span<f32_t const> const values = source_->values_of(drawn_[draw]);
                    sample_projection(values.data(), values.size(), config_.seed, coordinates_(draw).data());
                }
            else scale_classically_();
        }
        return status_t::success_k;
    }

    /** What the steps so far drew and spent. */
    sample_result_t result() const noexcept { return {written_, visited_members_, computed_distances_}; }

  private:

    /** One region candidate: its rank, then its entry, which breaks ties. */
    struct candidate_t {
        u64_t rank;
        shard_node_t node;
    };
    static_assert(sizeof(candidate_t) == 16);

    /** No draw, no root, no tree parent. */
    inline static constexpr u32_t none_k = std::numeric_limits<u32_t>::max();

    sample_plan(source_t const* source, sample_inputs_t inputs, sample_outputs_t outputs, sample_config_t config,
                bytes_span_t scratch, std::size_t wanted, std::size_t shown_count) noexcept
        : source_(source), inputs_(inputs), outputs_(outputs), config_(config), reserved_scratch_(scratch),
          wanted_(wanted), shown_count_(shown_count),
          nearest_shown_stride_(min_of(sample_nearest_shown_k, shown_count)),
          scaling_width_(scales_classically_k ? min_of(sample_scaling_draws_k, wanted) : 0),
          visited_capacity_(wanted * (config.expansion_factor ? config.expansion_factor : sample_expansion_factor_k)) {}

    /** Probe slots for @p count members: a power of two at least twice as many. */
    static std::size_t slots_for(std::size_t count) noexcept { return std::bit_ceil(2 * count + 2); }

    template <typename element_type_>
    std::span<element_type_> carve_(std::size_t count) noexcept {
        std::size_t const first = carved_bytes_;
        carved_bytes_ += round_up_to_multiple<std::size_t>(count * sizeof(element_type_), default_alignment_k);
        if (carved_bytes_ > reserved_scratch_.size()) return {};
        return {reinterpret_cast<element_type_*>(reserved_scratch_.data() + first), count};
    }

    /** Lays every buffer out over the scratch, or only measures them without one. */
    void carve_all_() noexcept {
        carved_bytes_ = 0;
        drawn_ = carve_<shard_node_t>(wanted_);
        drawn_slots_ = carve_<u32_t>(slots_for(wanted_));
        shown_nodes_ = carve_<shard_node_t>(shown_count_);
        shown_indices_ = carve_<u32_t>(shown_count_);
        shown_slots_ = carve_<u32_t>(slots_for(shown_count_));
        nearest_shown_indices_ = carve_<u32_t>(wanted_ * nearest_shown_stride_);
        nearest_shown_distances_ = carve_<f32_t>(wanted_ * nearest_shown_stride_);
        // Every root draws but a shown focus key's, which roots once.
        rooted_shown_ = carve_<u32_t>(shown_count_);
        roots_ = carve_<shard_node_t>(wanted_ + shown_count_);
        root_draws_ = carve_<u32_t>(wanted_ + shown_count_);
        root_radii_ = carve_<f32_t>(wanted_ + shown_count_);
        visited_ = carve_<shard_node_t>(visited_capacity_);
        visited_distances_ = carve_<f32_t>(visited_capacity_);
        visited_slots_ = carve_<u64_t>(slots_for(visited_capacity_));
        candidates_ = carve_<candidate_t>(visited_capacity_);
        distances_ = carve_<f32_t>(wanted_ + shown_count_);
        scaling_ = carve_<f32_t>(scaling_width_ * (scaling_width_ + 2));
    }

    static u64_t key_hash_(key_t const& key) noexcept { return static_cast<u64_t>(std::hash<key_t> {}(key)); }

    std::span<f32_t, 3> coordinates_(std::size_t draw) const noexcept {
        return std::span<f32_t, 3>(outputs_.coordinates.data() + draw * 3, 3);
    }

    void distance_(shard_node_t from, std::span<shard_node_t const> nodes, std::span<f32_t> distances) noexcept {
        if (nodes.empty()) return;
        computed_distances_ += nodes.size();
        source_->distance(from, nodes, distances);
    }

    /** Files shown point @p shown, answering whether its key is new among the shown. */
    bool claim_shown_(std::size_t shown) noexcept {
        key_t const& key = inputs_.shown[shown];
        std::size_t const mask = shown_slots_.size() - 1;
        for (std::size_t slot = mix_bits(key_hash_(key)) & mask;; slot = (slot + 1) & mask) {
            if (!shown_slots_[slot]) return shown_slots_[slot] = static_cast<u32_t>(shown + 1), true;
            if (inputs_.shown[shown_slots_[slot] - 1] == key) return false;
        }
    }

    /** The first shown point holding @p key, or @c none_k when it is not shown. */
    u32_t shown_index_(key_t const& key) const noexcept {
        std::size_t const mask = shown_slots_.size() - 1;
        for (std::size_t slot = mix_bits(key_hash_(key)) & mask;; slot = (slot + 1) & mask) {
            if (!shown_slots_[slot]) return none_k;
            if (inputs_.shown[shown_slots_[slot] - 1] == key) return shown_slots_[slot] - 1;
        }
    }

    static u64_t node_hash_(shard_node_t node) noexcept { return mix_bits(u64_t {node.shard} << 32 ^ node.node); }

    /** Files @p node as the next draw, answering false when it was drawn already. */
    bool claim_draw_(shard_node_t node) noexcept {
        std::size_t const mask = drawn_slots_.size() - 1;
        for (std::size_t slot = node_hash_(node) & mask;; slot = (slot + 1) & mask) {
            if (!drawn_slots_[slot]) return drawn_slots_[slot] = static_cast<u32_t>(written_ + 1), true;
            if (drawn_[drawn_slots_[slot] - 1] == node) return false;
        }
    }

    bool was_drawn_(shard_node_t node) const noexcept {
        std::size_t const mask = drawn_slots_.size() - 1;
        for (std::size_t slot = node_hash_(node) & mask;; slot = (slot + 1) & mask) {
            if (!drawn_slots_[slot]) return false;
            if (drawn_[drawn_slots_[slot] - 1] == node) return true;
        }
    }

    /** Draws @p node unless it is shown, drawn already or the outputs are full, keeping its nearest
     *  shown points; answers the draw or @c none_k. */
    u32_t draw_(shard_node_t node) noexcept {
        if (written_ == wanted_) return none_k;
        key_t const key = source_->key_of(node);
        if (shown_index_(key) != none_k || !claim_draw_(node)) return none_k;
        std::size_t const draw = written_++;
        drawn_[draw] = node;
        outputs_.keys[draw] = key;
        if (!nearest_shown_count_) return static_cast<u32_t>(draw);

        distance_(node, shown_nodes_.first(resolved_shown_count_), distances_.first(resolved_shown_count_));
        f32_t* const kept_distances = nearest_shown_distances_.data() + draw * nearest_shown_stride_;
        u32_t* const kept_shown = nearest_shown_indices_.data() + draw * nearest_shown_stride_;
        std::size_t kept = 0;
        for (std::size_t shown = 0; shown != resolved_shown_count_; ++shown) {
            f32_t const distance = distances_[shown];
            if (kept == nearest_shown_count_ && !(distance < kept_distances[kept - 1])) continue;
            std::size_t slot = kept == nearest_shown_count_ ? kept - 1 : kept++;
            for (; slot && distance < kept_distances[slot - 1]; --slot)
                kept_distances[slot] = kept_distances[slot - 1], kept_shown[slot] = kept_shown[slot - 1];
            kept_distances[slot] = distance, kept_shown[slot] = static_cast<u32_t>(shown);
        }
        return static_cast<u32_t>(draw);
    }

    /** Draws @p node hanging from the nearest shown point or earlier draw. */
    u32_t draw_nearest_(shard_node_t node) noexcept {
        u32_t const draw = draw_(node);
        if (draw == none_k) return none_k;
        key_t parent = outputs_.keys[draw];
        f32_t parent_distance = 0, best = std::numeric_limits<f32_t>::infinity();
        if (nearest_shown_count_) {
            best = parent_distance = nearest_shown_distances_[draw * nearest_shown_stride_];
            parent = inputs_.shown[shown_indices_[nearest_shown_indices_[draw * nearest_shown_stride_]]];
        }
        distance_(node, drawn_.first(draw), distances_.first(draw));
        for (std::size_t earlier = 0; earlier != draw; ++earlier)
            if (distances_[earlier] < best)
                best = parent_distance = distances_[earlier], parent = outputs_.keys[earlier];
        outputs_.parents[draw] = parent;
        outputs_.parent_distances[draw] = parent_distance;
        return draw;
    }

    /** Draws @p node hanging from @p previous, a shown point or an earlier draw. */
    void draw_after_(shard_node_t node, shard_node_t previous) noexcept {
        u32_t const draw = draw_(node);
        if (draw == none_k) return;
        outputs_.parents[draw] = source_->key_of(previous);
        distance_(previous, std::span<shard_node_t const>(&node, 1),
                  std::span<f32_t>(&outputs_.parent_distances[draw], 1));
    }

    /** Walks from @p from until it arrives at @p toward, drawing each step; a full chain resumes
     *  from its last step, so the steps never depend on the chain's length. */
    void walk_between_(shard_node_t from, shard_node_t toward) noexcept {
        std::array<shard_node_t, 64> chain;
        shard_node_t previous = from;
        for (std::size_t steps = chain.size(); steps == chain.size() && written_ != wanted_;) {
            steps = source_->walk(previous, toward, chain);
            for (std::size_t step = 0; step != steps && written_ != wanted_; ++step)
                draw_after_(chain[step], previous), previous = chain[step];
        }
    }

    /** Files @p node as the next root, @p draw naming its draw or @c none_k for a shown one. */
    void file_root_(shard_node_t node, u32_t draw) noexcept {
        roots_[roots_count_] = node, root_draws_[roots_count_] = draw;
        root_radii_[roots_count_++] = std::numeric_limits<f32_t>::infinity();
    }

    /** Files every focus entry that draws as a root, and a shown focus key once, at its first
     *  entry. */
    void file_focus_() noexcept {
        for (key_t const& key : inputs_.focus) {
            u32_t const shown = shown_index_(key);
            std::size_t entries = 0;
            source_->nodes_of(key, [&](shard_node_t node) noexcept {
                if (shown == none_k) {
                    if (u32_t const draw = draw_nearest_(node); draw != none_k) file_root_(node, draw);
                }
                else if (!entries++ && !rooted_shown_[shown]) rooted_shown_[shown] = 1, file_root_(node, none_k);
            });
        }
    }

    /** Attaches root @p root to its nearest earlier one, lowest first on ties, by a walk from
     *  there, each of the two reaching no farther than the other. */
    void attach_root_(std::size_t root) noexcept {
        if (!root) return;
        distance_(roots_[root], roots_.first(root), distances_.first(root));
        std::size_t nearest = 0;
        for (std::size_t earlier = 1; earlier != root; ++earlier)
            if (distances_[earlier] < distances_[nearest]) nearest = earlier;
        root_radii_[root] = min_of(root_radii_[root], distances_[nearest]);
        root_radii_[nearest] = min_of(root_radii_[nearest], distances_[nearest]);
        walk_between_(roots_[nearest], roots_[root]);
    }

    /** Files @p node among the visits of @p generation in the first @p mask + 1 slots, answering
     *  false when it is there. */
    bool visit_(shard_node_t node, u64_t generation, std::size_t mask, std::size_t& count) noexcept {
        for (std::size_t slot = node_hash_(node) & mask;; slot = (slot + 1) & mask) {
            u64_t const filed = visited_slots_[slot];
            if (filed >> 32 != generation) {
                visited_slots_[slot] = generation << 32 | (count + 1);
                visited_[count++] = node;
                return true;
            }
            if (visited_[static_cast<u32_t>(filed) - 1] == node) return false;
        }
    }

    /** Visits breadth-first from @p root through the entries within @p radius of it, at most
     *  @p quota times the breadth, and draws the @p quota lowest-ranked of them not drawn yet. */
    void draw_region_(shard_node_t root, f32_t radius, std::size_t quota, u64_t generation) noexcept {
        std::size_t const limit = min_of(visited_.size(), quota * (visited_capacity_ / wanted_));
        // Earlier regions' slots stay filed under older generations, so only the slots never used
        // before get zeroed.
        std::size_t const slots = slots_for(limit);
        if (slots > cleared_visit_slots_) {
            std::memset(visited_slots_.data() + cleared_visit_slots_, 0,
                        (slots - cleared_visit_slots_) * sizeof(u64_t));
            cleared_visit_slots_ = slots;
        }
        std::size_t const mask = slots - 1;
        std::size_t count = 0;
        visit_(root, generation, mask, count);
        visited_distances_[0] = 0;
        for (std::size_t head = 0; head != count; ++head) {
            if (visited_distances_[head] > radius) continue;
            ++visited_members_;
            std::size_t const first = count;
            source_->neighbors(visited_[head], [&](shard_node_t next) noexcept {
                if (count != limit) visit_(next, generation, mask, count);
            });
            distance_(root, visited_.subspan(first, count - first), visited_distances_.subspan(first, count - first));
        }
        std::size_t candidates = 0;
        for (std::size_t visit = 1; visit != count; ++visit) {
            shard_node_t const node = visited_[visit];
            if (!(visited_distances_[visit] <= radius) || was_drawn_(node)) continue;
            key_t const key = source_->key_of(node);
            if (shown_index_(key) == none_k)
                candidates_[candidates++] = {splitmix64_at(config_.seed, key_hash_(key)), node};
        }
        std::size_t const kept = min_of(candidates, quota);
        partial_sort(candidates_.first(candidates), kept, [](candidate_t const& left, candidate_t const& right) {
            if (left.rank != right.rank) return left.rank < right.rank;
            if (left.node.shard != right.node.shard) return left.node.shard < right.node.shard;
            return left.node.node < right.node.node;
        });
        for (std::size_t candidate = 0; candidate != kept; ++candidate) draw_nearest_(candidates_[candidate].node);
    }

    /** The seed nudging a draw's solved coordinates: its key's rank, mixed with its ordinal. */
    u64_t coordinates_seed_(std::size_t draw) const noexcept {
        return splitmix64_at(splitmix64_at(config_.seed, key_hash_(outputs_.keys[draw])),
                             outputs_.entry_ordinals[draw]);
    }

    /** Shown coordinates per unit of distance, over consecutive pairs of the first shown points,
     *  so distances meet the caller's layout in its own scale; one when they cannot tell. */
    f32_t shown_scale_() noexcept {
        std::size_t const pairs = min_of(resolved_shown_count_, sample_scaling_draws_k + 1) - 1;
        f64_t shown = 0, measured = 0;
        for (std::size_t pair = 0; pair != pairs; ++pair) {
            f32_t distance = 0;
            distance_(shown_nodes_[pair], std::span<shard_node_t const>(&shown_nodes_[pair + 1], 1),
                      std::span<f32_t>(&distance, 1));
            f32_t const* first = inputs_.shown_coordinates.data() + shown_indices_[pair] * 3;
            f32_t const* second = inputs_.shown_coordinates.data() + shown_indices_[pair + 1] * 3;
            f64_t squares = 0;
            for (std::size_t column = 0; column != 3; ++column) {
                f64_t const offset = f64_t {first[column]} - f64_t {second[column]};
                squares += offset * offset;
            }
            shown += std::sqrt(squares), measured += distance;
        }
        return shown > 0 && measured > 0 ? static_cast<f32_t>(shown / measured) : 1.0f;
    }

    void place_among_shown_(std::size_t draw, f32_t scale) noexcept {
        std::array<f32_t, sample_nearest_shown_k * 3> references;
        std::array<f32_t, sample_nearest_shown_k> distances;
        for (std::size_t nearest = 0; nearest != nearest_shown_count_; ++nearest) {
            std::size_t const kept = draw * nearest_shown_stride_ + nearest;
            f32_t const* shown = inputs_.shown_coordinates.data() + shown_indices_[nearest_shown_indices_[kept]] * 3;
            for (std::size_t axis = 0; axis != 3; ++axis) references[nearest * 3 + axis] = shown[axis];
            distances[nearest] = nearest_shown_distances_[kept] * scale;
        }
        sample_gauss_newton(references.data(), distances.data(), nearest_shown_count_, coordinates_seed_(draw),
                            coordinates_(draw).data());
    }

    /** Solves the first @ref sample_scaling_draws_k draws by classical scaling among themselves,
     *  three power iterations deflating the double-centered squared distances, and every later draw
     *  by Gauss-Newton among its nearest of them. */
    void scale_classically_() noexcept {
        std::size_t const anchors = min_of(written_, scaling_width_);
        if (!anchors) return;
        f32_t* const centered = scaling_.data();
        f32_t* const eigenvector = centered + anchors * anchors;
        f32_t* const product = eigenvector + anchors;
        for (std::size_t anchor = 0; anchor != anchors; ++anchor) {
            distance_(drawn_[anchor], drawn_.first(anchors), distances_.first(anchors));
            for (std::size_t other = 0; other != anchors; ++other)
                centered[anchor * anchors + other] = distances_[other] * distances_[other];
        }
        // Symmetric squares, then each anchor's mean in `product` and the total in `total`.
        f32_t total = 0;
        for (std::size_t anchor = 0; anchor != anchors; ++anchor) {
            f32_t sum = 0;
            for (std::size_t other = 0; other != anchors; ++other) {
                f32_t const square = 0.5f * (centered[anchor * anchors + other] + centered[other * anchors + anchor]);
                sum += square;
            }
            product[anchor] = sum / static_cast<f32_t>(anchors);
            total += product[anchor];
        }
        total /= static_cast<f32_t>(anchors);
        for (std::size_t anchor = 0; anchor != anchors; ++anchor)
            for (std::size_t other = anchor; other != anchors; ++other) {
                f32_t const square = 0.5f * (centered[anchor * anchors + other] + centered[other * anchors + anchor]);
                f32_t const entry = -0.5f * (square - product[anchor] - product[other] + total);
                centered[anchor * anchors + other] = centered[other * anchors + anchor] = entry;
            }
        for (std::size_t axis = 0; axis != 3; ++axis) {
            for (std::size_t anchor = 0; anchor != anchors; ++anchor)
                eigenvector[anchor] =
                    2.0f * static_cast<f32_t>(uniform_unit_f64(splitmix64_at(config_.seed, axis * anchors + anchor))) -
                    1.0f;
            auto const multiply = [&]() noexcept {
                f32_t squares = 0;
                for (std::size_t anchor = 0; anchor != anchors; ++anchor) {
                    f32_t sum = 0;
                    for (std::size_t other = 0; other != anchors; ++other)
                        sum += centered[anchor * anchors + other] * eigenvector[other];
                    product[anchor] = sum, squares += sum * sum;
                }
                return std::sqrt(squares);
            };
            for (std::size_t iteration = 0; iteration != 64; ++iteration) {
                f32_t const norm = multiply();
                if (!(norm > 0)) break;
                for (std::size_t anchor = 0; anchor != anchors; ++anchor) eigenvector[anchor] = product[anchor] / norm;
            }
            // The Rayleigh quotient keeps the sign a norm would lose, so deflation subtracts the
            // pair with the right sign.
            multiply();
            f32_t eigenvalue = 0;
            for (std::size_t anchor = 0; anchor != anchors; ++anchor)
                eigenvalue += eigenvector[anchor] * product[anchor];
            f32_t const spread = std::sqrt(max_of(eigenvalue, 0.0f));
            for (std::size_t anchor = 0; anchor != anchors; ++anchor) {
                outputs_.coordinates[anchor * 3 + axis] = eigenvector[anchor] * spread;
                for (std::size_t other = 0; other != anchors; ++other)
                    centered[anchor * anchors + other] -= eigenvalue * eigenvector[anchor] * eigenvector[other];
            }
        }

        std::array<f32_t, sample_nearest_shown_k * 3> references;
        std::array<f32_t, sample_nearest_shown_k> kept_distances;
        std::size_t const nearest = min_of(sample_nearest_shown_k, anchors);
        for (std::size_t draw = anchors; draw != written_; ++draw) {
            distance_(drawn_[draw], drawn_.first(anchors), distances_.first(anchors));
            std::array<u32_t, sample_nearest_shown_k> kept_anchors;
            std::size_t kept = 0;
            for (std::size_t anchor = 0; anchor != anchors; ++anchor) {
                f32_t const distance = distances_[anchor];
                if (kept == nearest && !(distance < kept_distances[kept - 1])) continue;
                std::size_t slot = kept == nearest ? kept - 1 : kept++;
                for (; slot && distance < kept_distances[slot - 1]; --slot)
                    kept_distances[slot] = kept_distances[slot - 1], kept_anchors[slot] = kept_anchors[slot - 1];
                kept_distances[slot] = distance, kept_anchors[slot] = static_cast<u32_t>(anchor);
            }
            for (std::size_t reference = 0; reference != kept; ++reference)
                for (std::size_t axis = 0; axis != 3; ++axis)
                    references[reference * 3 + axis] = outputs_.coordinates[kept_anchors[reference] * 3 + axis];
            sample_gauss_newton(references.data(), kept_distances.data(), kept, coordinates_seed_(draw),
                                coordinates_(draw).data());
        }
    }

    source_t const* source_;
    sample_inputs_t inputs_;
    sample_outputs_t outputs_;
    sample_config_t config_;
    bytes_span_t reserved_scratch_;
    std::size_t wanted_;
    std::size_t shown_count_;
    std::size_t nearest_shown_stride_;
    std::size_t scaling_width_;
    std::size_t visited_capacity_;
    std::size_t carved_bytes_ = 0;
    std::size_t cleared_visit_slots_ = 0;

    std::size_t written_ = 0;
    std::size_t resolved_shown_count_ = 0;
    std::size_t nearest_shown_count_ = 0;
    std::size_t roots_count_ = 0;
    std::size_t visited_members_ = 0;
    std::size_t computed_distances_ = 0;

    std::span<shard_node_t> drawn_;
    std::span<u32_t> drawn_slots_;
    std::span<shard_node_t> shown_nodes_;
    std::span<u32_t> shown_indices_;
    std::span<u32_t> shown_slots_;
    std::span<u32_t> nearest_shown_indices_;
    std::span<f32_t> nearest_shown_distances_;
    std::span<u32_t> rooted_shown_;
    std::span<shard_node_t> roots_;
    std::span<u32_t> root_draws_;
    std::span<f32_t> root_radii_;
    std::span<shard_node_t> visited_;
    std::span<f32_t> visited_distances_;
    std::span<u64_t> visited_slots_;
    std::span<candidate_t> candidates_;
    std::span<f32_t> distances_;
    std::span<f32_t> scaling_;
};

} // namespace unum::usearch::utilities
