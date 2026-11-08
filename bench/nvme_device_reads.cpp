/**
 *  @file bench/nvme_device_reads.cpp
 *  @author Ash Vardanian
 *  @date October 31, 2026
 *  @brief The I/O engines alone, beside @c fio : each engine reads the covering rows of a live
 *      monolith at the depth its drive admits, one table row per engine and column.
 *
 *  Every number derives from the device, the monolith and the caller. The address is a slot holding
 *  a live monolith, so reads land on written blocks, which a drive serves from NAND rather than
 *  answering from its mapping table. A read is the covering read of one random row of the key,
 *  neighbors or value column, as a walk issues it, and each column takes as many reads as the
 *  monolith has nodes. Each thread keeps the depth @c nvme_queue_depth derives from the disk's tags
 *  and hardware queues in flight, or its share of @c --depth. Engines run in turn: blocking reads,
 *  AIO and io_uring over the block node, and io_uring passthrough over its generic character node,
 *  <tt>/dev/ngXnY</tt>.
 *
 *  An @c spdk:// monolith reads through the SPDK engine alone, its depth derived from the queues
 *  the controller granted, and each of its rows prints an @c spdk_nvme_perf line for a fio one.
 *
 *  @code{.sh}
 *  systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 -- \
 *      usearch_bench_nvme_device_reads --monolith /dev/nvme1n1@64GiB --threads 64
 *  usearch_bench_nvme_device_reads --monolith spdk://0000:c6:00.0@64GiB --threads 64 --depth 4096
 *  @endcode
 */
#include <algorithm> // `std::stable_sort`
#include <array>     // `std::array`
#include <charconv>  // `std::from_chars` - the cgroup's memory cap
#include <chrono>    // `std::chrono::steady_clock` - the rates and latencies
#include <memory>    // `std::unique_ptr` - the topology
#include <string>    // `std::string`
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "usearch/utilities/nvme_walks.hpp" // `nvme_begin_call`, `nvme_covering_read`, `nvme_read_header`
#include "usearch/utilities/sample.hpp"     // `splitmix64_at`

#include "abi.hpp" // `read_settings`, `percentile`, `cgroup_text`, `character_sibling`

namespace unum::usearch::bench {
using namespace unum::usearch::tools;

/** The executor over every engine the build admits, yielding between polls as the library does. */
using nvme_executor_t = nvme_executor<default_io_engine<fu::standard_yield_t>>;

/** What one table row reads: the column's rows on the medium, the block reads round to, and the
 *  depth and count of reads the threads share. */
struct reads_plan_t {

    /** Where the column's first row sits on the medium, its byte base included. */
    u64_t column_begin = 0;

    /** Bytes one row of the column holds. */
    u64_t row_bytes = 0;

    /** Rows the column holds, which random reads pick among. */
    u64_t rows = 0;

    /** The medium's logical block. */
    u64_t block_bytes = 0;

    /** Bytes the widest covering read of a row takes, which sizes every staging buffer. */
    u64_t buffer_bytes = 0;

    /** Reads each thread keeps in flight. */
    std::size_t depth = 0;

    /** Reads all threads complete together. */
    std::size_t reads = 0;

    /** The stream the random rows draw from. */
    u64_t seed = 0;
};

/** Nanoseconds on the steady clock, which a read's context carries from issue to completion. */
inline u64_t steady_nanoseconds() noexcept {
    return static_cast<u64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

/** The first read of thread @p thread 's share of @p reads over @p threads, the remainder going to
 *  the first threads. */
inline std::size_t share_begin(std::size_t reads, std::size_t threads, std::size_t thread) noexcept {
    return thread * (reads / threads) + min_of(thread, reads % threads);
}

/** One engine's row: each thread keeps @p plan.depth covering reads of random rows in flight until
 *  its share of @p plan.reads completes, writing each read's nanoseconds into @p latencies and the
 *  bytes it moved into @p moved, one slot per thread, and its failure into @p statuses. */
inline status_t engine_reads(nvme_executor_t& executor, std::span<nvme_executor_t::medium_t const> media,
                             reads_plan_t const& plan, std::span<u64_t> latencies, std::span<u64_t> moved,
                             std::span<status_t> statuses) noexcept {
    std::size_t const scratch_bytes = plan.depth * sizeof(io_completion_t) + alignof(io_completion_t);
    auto [call, call_status] = nvme_begin_call(executor, media, plan.depth, scratch_bytes, plan.depth,
                                               plan.buffer_bytes);
    if (failed(call_status)) return call_status;
    std::size_t const threads = executor.size();
    return executor.for_threads([&](std::size_t thread, std::size_t) noexcept {
        nvme_executor_t::engine_t& engine = executor.engine(thread);
        std::span<io_completion_t> const polled_into {
            reinterpret_cast<io_completion_t*>(
                executor.arena(thread).allocate(plan.depth * sizeof(io_completion_t), alignof(io_completion_t))),
            plan.depth};
        std::size_t const first = share_begin(plan.reads, threads, thread);
        std::span<u64_t> const timings = latencies.subspan(first, share_begin(plan.reads, threads, thread + 1) - first);
        u64_t const stream = plan.seed + thread;
        status_t& status = statuses[thread];
        std::size_t issued = 0, completed = 0;
        auto random_read = [&](io_buffer_id_t buffer) noexcept {
            u64_t const row = splitmix64_at(stream, issued) % plan.rows;
            return nvme_covering_read(plan.column_begin + row * plan.row_bytes, plan.row_bytes, plan.block_bytes, 0,
                                      buffer, steady_nanoseconds());
        };
        for (; issued != min_of(plan.depth, timings.size()) && succeeded(status); ++issued)
            status = engine.enqueue_read(random_read(static_cast<io_buffer_id_t>(issued)));
        while (completed != issued && succeeded(status)) {
            if (failed(status = engine.submit(io_wait_t::blocking_k))) return;
            std::size_t const polled = engine.poll_completions(polled_into);
            u64_t const now = steady_nanoseconds();
            for (io_completion_t const& completion : polled_into.first(polled)) {
                if (completion.result <= 0) status = status_t::io_k;
                moved[thread] += static_cast<u64_t>(max_of<i32_t>(completion.result, 0));
                timings[completed++] = now - completion.context;
                if (issued != timings.size() && succeeded(status))
                    status = engine.enqueue_read(random_read(completion.buffer)), ++issued;
            }
        }
    });
}

/** One column a row reads: its name and its header entry. */
struct column_reads_t {
    std::string_view name;
    safegraphs_column_t column;
    u64_t read_bytes = 0;
};

/** One engine a row reads through: its name, its transport, and the address it opens. */
struct engine_row_t {
    std::string_view name;
    io_transport_t transport = io_transport_t::automatic_k;
    std::string address;
};

/** The @c fio line reading what @p plan reads through @p engine on @p threads threads. */
inline std::string fio_line(engine_row_t const& engine, reads_plan_t const& plan, u64_t read_bytes, std::size_t threads,
                            std::string_view location, linux_uring_reaping_t reaping) {
    std::string_view const ioengine = engine.transport == io_transport_t::blocking_k    ? "psync"
                                      : engine.transport == io_transport_t::linux_aio_k ? "libaio"
                                      : engine.transport == io_transport_t::linux_uring_nvme_k
                                          ? "io_uring_cmd --cmd_type=nvme"
                                          : "io_uring --fixedbufs --registerfiles";
    u64_t const offset = plan.column_begin - plan.column_begin % plan.block_bytes;
    u64_t const size = round_up_to_multiple<u64_t>(plan.column_begin + plan.rows * plan.row_bytes, plan.block_bytes) -
                       offset;
    // fio refuses direct=1 on a generic char node, which has no page cache to bypass.
    std::string_view const direct = engine.transport == io_transport_t::linux_uring_nvme_k ? "" : " --direct=1";
    return fmt::format("fio --name={} --filename={} --offset={} --size={}{} --ioengine={} --rw=randread " //
                       "--bs={} --iodepth={} --numjobs={} --number_ios={} --group_reporting{}",
                       engine.name, location, offset, size, direct, ioengine, read_bytes,
                       engine.transport == io_transport_t::blocking_k ? 1 : plan.depth, threads, plan.reads / threads,
                       reaping == linux_uring_reaping_t::polled_k ? " --hipri" : "");
}

/** The @c spdk_nvme_perf line reading what @p plan reads on @p threads cores of the namespace
 *  @p address names, though over the whole namespace, as it takes no offset range. */
inline std::string spdk_perf_line(reads_plan_t const& plan, u64_t read_bytes, std::size_t threads,
                                  std::string_view address) {
    file_address_t const parsed = parse_file_address(address).value;
    // An spdk:// address holds a bare PCIe address, which a transport ID spells with its keys.
    std::string_view const pcie_keys = names_transport_id(parsed.location) ? "" : "trtype:PCIe traddr:";
    std::string const name_space = parsed.namespace_id ? fmt::format(" ns:{}", parsed.namespace_id) : std::string {};
    return fmt::format("spdk_nvme_perf -q {} -o {} -w randread -t 30 -c '[0-{}]' -r '{}{}{}'", plan.depth, read_bytes,
                       threads - 1, pcie_keys, parsed.location, name_space);
}

/** Runs the engine reads benchmark over the flags in @p argv. */
int bench_nvme_device_reads(int argc, char const* const* argv) {
    std::string_view monolith;
    count_t total_depth {.minimum = 0};
    std::array<flag_t, 2> const flags {{
        {"--monolith", "address", "required: a live monolith on a drive, like /dev/nvme1n1@64GiB", &monolith},
        {"--depth", "reads", "reads in flight over all threads; derived from the disk's tags and queues by default",
         &total_depth},
    }};
    settings_t const settings = read_settings(
        argc, argv,
        "usearch_bench_nvme_device_reads --monolith <address> [--threads <count>] " //
        "[--depth <reads>] [--filter <engines>]",
        flags);
    if (monolith.empty()) return fmt::println(stderr, "--monolith is required"), 1;
    growing_buffer<byte_t> header_bytes {global_allocator};
    status_t const header_read = nvme_read_header<fu::standard_yield_t>(monolith, header_bytes);
    auto const [header, header_status] = safegraphs_read_header(to_span(header_bytes));
    if (failed(header_read) || failed(header_status))
        return fmt::println(stderr, "--monolith=\"{}\" starts no SafeGraphs monolith: {}", monolith,
                            status_to_string(failed(header_read) ? header_read : header_status)),
               1;
    std::string_view const location = parse_file_address(monolith).value.location;
    std::string const character_address = character_sibling(monolith);
    std::vector<engine_row_t> const engines =
        names_transport_id(monolith)
            ? std::vector<engine_row_t> {{"spdk", io_transport_t::linux_spdk_k, std::string {monolith}}}
            : std::vector<engine_row_t> {{
                  {"blocking", io_transport_t::blocking_k, std::string {monolith}},
                  {"aio", io_transport_t::linux_aio_k, std::string {monolith}},
                  {"uring", io_transport_t::linux_uring_k, std::string {monolith}},
                  {"passthrough", io_transport_t::linux_uring_nvme_k, character_address},
              }};

    fu_topology_t topology = nullptr;
    if (fu_topology_new(&topology) != fu_success_k) return fmt::println(stderr, "Topology discovery failed"), 1;
    std::unique_ptr<void, void (*)(fu_topology_t)> const topology_owner {topology, &fu_topology_delete};
    std::optional<std::string> const memory_cap = cgroup_text("memory.max");
    u64_t memory_limit = 0;
    if (memory_cap) std::from_chars(memory_cap->data(), memory_cap->data() + memory_cap->size(), memory_limit);

    for (engine_row_t const& engine : engines) {
        if (!settings.filter.selects(engine.name)) continue;
        if (engine.address.empty()) {
            fmt::println("- {}: skipped, {} has no generic character node", engine.name, location);
            continue;
        }
        std::array<nvme_executor_t::medium_t, 1> media;
        nvme_executor_t executor;
        // SPDK holds go back explicitly, so every way out of this engine's row releases them.
        struct holds_t {
            std::array<nvme_executor_t::medium_t, 1>& media;
            nvme_executor_t& executor;
            ~holds_t() noexcept {
                executor.release();
                release_medium<fu::standard_yield_t>(media[0]);
            }
        } const holds {media, executor};
        io_engine_config_t config;
        config.transport = engine.transport;
        status_t open_status = status_t::success_k;
#if USEARCH_WITH_LINUX_SPDK
        if (engine.transport == io_transport_t::linux_spdk_k) {
            auto [name_space, spdk_status] = spdk_medium_t::open<fu::standard_yield_t>(engine.address);
            media[0] = nvme_executor_t::medium_t {std::move(name_space)}, open_status = spdk_status;
        }
        else
#endif
        {
            auto [file, file_status] = file_medium_t::open(engine.address);
            media[0] = nvme_executor_t::medium_t {std::move(file)}, open_status = file_status;
        }
        if (failed(open_status)) {
            fmt::println("- {}: skipped, {} does not open: {}", engine.name, engine.address,
                         status_to_string(open_status));
            continue;
        }
        auto [spawned, spawn_status] = nvme_executor_t::spawn(topology, settings.threads.value, config);
        if (failed(spawn_status)) {
            fmt::println("- {}: skipped, the engine does not open: {}", engine.name, status_to_string(spawn_status));
            continue;
        }
        executor = std::move(spawned);
        std::size_t const threads = executor.size();
        std::string_view const engine_location = parse_file_address(engine.address).value.location;
        // A pinned total splits evenly over the threads, as fio's iodepth times numjobs does.
        std::size_t const depth = total_depth.value ? total_depth.value / threads
                                                    : nvme_queue_depth<nvme_executor_t::medium_t>(media, threads);
        io_device_queues_t const queues = device_queues_of(media[0]);
        if (!depth)
            return fmt::println(stderr, "{} has no block queue to derive a depth from, and no --depth of {} or more",
                                engine.address, threads),
                   1;
        fmt::println("- {}: {} threads · {} reads deep each, {} tags over {} hardware queues · {}", engine.name,
                     threads, depth, queues.tags, queues.hardware_queues,
                     reads_directly(media[0]) ? "direct" : "buffered");

        u64_t const block_bytes = block_bytes_of(media[0]);
        u64_t const column_base = byte_base_of(media[0]) + header.data_offset;
        std::array<column_reads_t, 3> columns {{
            {"key", header.key, 0},
            {"neighbors", header.neighbors, 0},
            {"value", header.value, 0},
        }};
        // A row tiling the block reads one aligned block, and the table opens with the narrowest.
        for (column_reads_t& column : columns)
            column.read_bytes = nvme_covering_read(column_base + column.column.data.first, column.column.row_bytes(),
                                                   block_bytes, 0, 0, 0)
                                    .length;
        std::stable_sort(columns.begin(), columns.end(), [](column_reads_t const& left, column_reads_t const& right) {
            return left.read_bytes < right.read_bytes;
        });

        for (column_reads_t const& column : columns) {
            reads_plan_t const plan {
                .column_begin = column_base + column.column.data.first,
                .row_bytes = column.column.row_bytes(),
                .rows = header.nodes(),
                .block_bytes = block_bytes,
                .buffer_bytes = nvme_covering_bytes(column.column.row_bytes(), block_bytes),
                .depth = depth,
                .reads = header.nodes(),
                .seed = settings.seed,
            };
            u64_t const stride = round_up_to_multiple<u64_t>(plan.buffer_bytes, io_alignment_k);
            u64_t const needed = plan.reads * sizeof(u64_t) + threads * plan.depth * stride;
            if (memory_limit && needed > memory_limit) {
                fmt::println(stderr, "{} {} reads need {} bytes past the cgroup's memory.max of {}", engine.name,
                             column.name, needed, memory_limit);
                return 1;
            }
            std::vector<u64_t> latencies(plan.reads), moved(threads);
            std::vector<status_t> statuses(threads, status_t::success_k);
            std::chrono::steady_clock::time_point const started = std::chrono::steady_clock::now();
            status_t const ran = engine_reads(executor, media, plan, latencies, moved, statuses);
            double const seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            for (status_t const status : statuses)
                if (failed(status))
                    return fmt::println(stderr, "{} reads failed: {}", engine.name, status_to_string(status)), 1;
            if (failed(ran)) return fmt::println(stderr, "{} reads failed: {}", engine.name, status_to_string(ran)), 1;
            u64_t bytes = 0;
            for (u64_t const thread_bytes : moved) bytes += thread_bytes;
            fmt::println("  {} rows of {} B: {} B reads · {:.0f} IOPS · {:.2f} GB/s · p50 {:.0f} µs · p99 {:.0f} µs",
                         column.name, plan.row_bytes, column.read_bytes, static_cast<double>(plan.reads) / seconds,
                         static_cast<double>(bytes) / seconds / 1e9, percentile(latencies, 0.50) / 1e3,
                         percentile(latencies, 0.99) / 1e3);
            fmt::println("  {}", engine.transport == io_transport_t::linux_spdk_k
                                     ? spdk_perf_line(plan, column.read_bytes, threads, engine.address)
                                     : fio_line(engine, plan, column.read_bytes, threads, engine_location,
                                                nvme_reaping<nvme_executor_t::medium_t>(media)));
        }
    }
    return 0;
}

} // namespace unum::usearch::bench

namespace us = unum::usearch;

int main(int argc, char** argv) { return us::bench::bench_nvme_device_reads(argc, argv); }
