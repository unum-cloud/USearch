/**
 * @brief A benchmark for the construction speed of the USearch index
 * and the resulting accuracy (recall) of the Approximate Nearest Neighbors
 * Search queries.
 */
#include <execinfo.h>
#include <fcntl.h>    // `open`
#include <stdlib.h>   // `getenv`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `stat`
#include <unistd.h>

#include <csignal>
#include <cstdio>
#include <iostream>  // `std::cerr`
#include <numeric>   // `std::iota`
#include <stdexcept> // `std::invalid_argument`
#include <thread>    // `std::thread::hardware_concurrency()`
#include <variant>   // `std::monostate`

#include <clipp.h> // Command Line Interface
#include <omp.h>   // `omp_get_num_threads()`

#include "advanced.hpp"

using namespace unum::usearch;
using namespace unum;

template <typename element_at>
std::size_t offset_of(element_at const* begin, element_at const* end, element_at v) noexcept {
    auto iterator = begin;
    for (; iterator != end; ++iterator)
        if (*iterator == v)
            break;
    return iterator - begin;
}

template <typename element_at> bool contains(element_at const* begin, element_at const* end, element_at v) noexcept {
    return offset_of(begin, end, v) != static_cast<std::size_t>(end - begin);
}

template <typename scalar_at> //
struct alignas(32) persisted_matrix_gt {
    using scalar_t = scalar_at;
    std::uint8_t const* raw_handle{};
    std::size_t raw_length{};
    std::uint32_t rows{};
    std::uint32_t cols{};
    scalar_t const* scalars{};

    persisted_matrix_gt(char const* path) noexcept(false) {
        if (!path || !std::strlen(path))
            throw std::invalid_argument("The file path is empty");
        auto file_descriptor = open(path, O_RDONLY | O_CLOEXEC);
        if (file_descriptor == -1)
            throw std::invalid_argument("Couldn't open provided file path");
        struct stat stat_vectors;
        if (fstat(file_descriptor, &stat_vectors) == -1)
            throw std::invalid_argument("Couldn't obtain file stats");
        raw_length = stat_vectors.st_size;
        raw_handle = (std::uint8_t*)mmap(NULL, raw_length, PROT_READ, MAP_PRIVATE, file_descriptor, 0);
        if (raw_handle == nullptr)
            throw std::invalid_argument("Couldn't memory-map the file");
        std::memcpy(&rows, raw_handle, sizeof(rows));
        std::memcpy(&cols, raw_handle + sizeof(rows), sizeof(cols));
        scalars = (scalar_t*)(raw_handle + sizeof(rows) + sizeof(cols));
    }

    ~persisted_matrix_gt() {
        if (raw_handle != nullptr)
            munmap((void*)raw_handle, raw_length);
    }

    scalar_t const* row(std::size_t i) const noexcept { return scalars + i * cols; }
    std::size_t row_size_bytes() const noexcept { return cols * sizeof(scalar_t); }
    std::size_t size_bytes() const noexcept { return rows * row_size_bytes(); }
};

template <typename scalar_at> //
struct vectors_view_gt {
    using scalar_t = scalar_at;

    scalar_t const* begin_{};
    std::size_t count_{};
    std::size_t stride_{};

    std::size_t size() const noexcept { return count_; }
    scalar_t const* at(std::size_t i) const noexcept { return begin_ + i * stride_; }
};

template <typename scalar_at, typename vector_id_at> //
struct persisted_dataset_gt {
    using scalar_t = scalar_at;
    using vector_id_t = vector_id_at;
    persisted_matrix_gt<scalar_t> vectors_;
    persisted_matrix_gt<scalar_t> queries_;
    persisted_matrix_gt<vector_id_t> neighborhoods_;

    persisted_dataset_gt(char const* path_vectors, char const* path_queries, char const* path_neighbors) noexcept(false)
        : vectors_(path_vectors), queries_(path_queries), neighborhoods_(path_neighbors) {
        if (vectors_.cols != queries_.cols)
            throw std::invalid_argument("Contents and queries have different dimensionality");
        if (queries_.rows != neighborhoods_.rows)
            throw std::invalid_argument("Number of ground-truth neighborhoods doesn't match number of queries");
    }

    std::size_t dimensions() const noexcept { return vectors_.cols; }
    std::size_t vectors_count() const noexcept { return vectors_.rows; }
    std::size_t queries_count() const noexcept { return queries_.rows; }
    std::size_t neighborhood_size() const noexcept { return neighborhoods_.cols; }
    scalar_t const* vector(std::size_t i) const noexcept { return vectors_.row(i); }
    scalar_t const* query(std::size_t i) const noexcept { return queries_.row(i); }
    vector_id_t const* neighborhood(std::size_t i) const noexcept { return neighborhoods_.row(i); }

    vectors_view_gt<scalar_t> vectors_view() const noexcept { return {vector(0), vectors_count(), dimensions()}; }
};

template <typename scalar_at, typename vector_id_at> //
struct in_memory_dataset_gt {
    using scalar_t = scalar_at;
    using vector_id_t = vector_id_at;

    std::vector<scalar_t> vectors_{};
    std::vector<scalar_t> queries_{};
    std::vector<vector_id_t> neighborhoods_{};
    std::size_t dimensions_{};
    std::size_t vectors_count_{};
    std::size_t neighborhood_size_{};
    std::size_t queries_count_{};

    in_memory_dataset_gt( //
        std::size_t dimensions, std::size_t vectors_count, std::size_t queries_count,
        std::size_t neighborhood_size) noexcept(false)
        : vectors_(vectors_count * dimensions), queries_(queries_count * dimensions),
          neighborhoods_(queries_count * neighborhood_size), dimensions_(dimensions), vectors_count_(vectors_count),
          queries_count_(queries_count), neighborhood_size_(neighborhood_size) {}

    std::size_t dimensions() const noexcept { return dimensions_; }
    std::size_t vectors_count() const noexcept { return vectors_count_; }
    std::size_t queries_count() const noexcept { return vectors_count(); }
    std::size_t neighborhood_size() const noexcept { return 1; }
    scalar_t const* vector(std::size_t i) const noexcept { return vectors_.data() + i * dimensions_; }
    scalar_t const* query(std::size_t i) const noexcept { return queries_.data() + i * dimensions_; }
    vector_id_t const* neighborhood(std::size_t i) const noexcept {
        return neighborhoods_.data() + i * neighborhood_size_;
    }

    scalar_t* vector(std::size_t i) noexcept { return vectors_.data() + i * dimensions_; }
    scalar_t* query(std::size_t i) noexcept { return queries_.data() + i * dimensions_; }
    vector_id_t* neighborhood(std::size_t i) noexcept { return neighborhoods_.data() + i * neighborhood_size_; }

    vectors_view_gt<scalar_t> vectors_view() const noexcept { return {vector(0), vectors_count(), dimensions()}; }
};

char const* getenv_or(char const* name, char const* default_) { return getenv(name) ? getenv(name) : default_; }

template <typename index_at, typename vector_id_at, typename real_at>
void index_many(index_at& native, std::size_t n, vector_id_at const* ids, real_at const* vectors, std::size_t dims) {

    using span_t = span_gt<float const>;

#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        native.add(ids[i], span_t{vectors + dims * i, dims}, omp_get_thread_num(), false);
        if (((i + 1) % 100000) == 0)
            std::printf("-- added point # %zu\n", i + 1);
    }
}

template <typename index_at, typename vector_id_at, typename real_at>
void search_many(index_at& native, std::size_t n, real_at const* vectors, std::size_t dims, std::size_t wanted,
                 vector_id_at* ids, real_at* distances) {

    using span_t = span_gt<float const>;

#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t found = native.search(            //
            span_t{vectors + dims * i, dims}, wanted, //
            ids + wanted * i, distances + wanted * i, //
            omp_get_thread_num());
    }
}

template <typename callback_at> std::size_t nanoseconds(callback_at&& callback) {
    auto t_start = std::chrono::high_resolution_clock::now();
    callback();
    auto t_end = std::chrono::high_resolution_clock::now();
    auto dt_add = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start).count();
    return dt_add;
}

template <typename dataset_at, typename index_at> //
static void single_shot(dataset_at& dataset, index_at& index, bool construct = true) {
    using label_t = typename index_at::label_t;
    using distance_t = typename index_at::distance_t;

    if (construct) {
        // Perform insertions, evaluate speed
        std::vector<label_t> ids(dataset.vectors_count());
        std::iota(ids.begin(), ids.end(), 0);
        auto dt_add = nanoseconds(
            [&] { index_many(index, dataset.vectors_count(), ids.data(), dataset.vector(0), dataset.dimensions()); });
        std::printf("- Added %.2f vectors/sec\n", dataset.vectors_count() * 1e9 / dt_add);
    }

    // Perform search, evaluate speed
    std::vector<label_t> found_neighbors(dataset.queries_count() * dataset.neighborhood_size());
    std::vector<distance_t> found_distances(dataset.queries_count() * dataset.neighborhood_size());
    std::size_t executed_search_queries = 0;

    auto dt_search = nanoseconds([&] {
        while (executed_search_queries < dataset.vectors_count()) {
            search_many(index, dataset.queries_count(), dataset.query(0), dataset.dimensions(),
                        dataset.neighborhood_size(), found_neighbors.data(), found_distances.data());
            executed_search_queries += dataset.queries_count();
            std::printf("- Searched %zu vectors\n", executed_search_queries);
        }
    });
    std::printf("- Searched %.2f vectors/sec\n", executed_search_queries * 1e9 / dt_search);

    // Evaluate quality
    std::size_t recall_at_1 = 0, recall_full = 0;
    for (std::size_t i = 0; i != dataset.queries_count(); ++i) {
        auto expected = dataset.neighborhood(i);
        auto received = found_neighbors.data() + i * dataset.neighborhood_size();
        recall_at_1 += expected[0] == received[0];
        recall_full += contains(received, received + dataset.neighborhood_size(), label_t{expected[0]});
    }
    std::printf("- Recall@1 %.2f %%\n", recall_at_1 * 100.f / dataset.queries_count());
    std::printf("- Recall %.2f %%\n", recall_full * 100.f / dataset.queries_count());
}

void handler(int sig) {
    void* array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

bool ends_with(std::string const& value, std::string const& ending) {
    if (ending.size() > value.size())
        return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

struct args_t {
    std::string path_vectors;
    std::string path_queries;
    std::string path_neighbors;

    std::size_t connectivity = config_t::connectivity_default_k;
    std::size_t expansion_add = config_t::expansion_construction_default_k;
    std::size_t expansion_search = config_t::expansion_search_default_k;
    std::size_t threads = std::thread::hardware_concurrency();

    bool help = false;

    bool big = false;
    bool native = false;
    bool quantize_f16 = false;
    bool quantize_i8 = false;

    bool metric_ip = true;
    bool metric_l2 = false;
    bool metric_cos = false;
    bool metric_haversine = false;
};

template <typename index_at, typename dataset_at> //
void run_type_punned(dataset_at& dataset, args_t const& args, config_t config) {

    accuracy_t accuracy = accuracy_t::f32_k;
    if (args.quantize_f16)
        accuracy = accuracy_t::f16_k;
    if (args.quantize_i8)
        accuracy = accuracy_t::i8q100_k;

    index_at index;
    if (args.metric_ip)
        index = index_at::ip(dataset.dimensions(), accuracy, config);
    if (args.metric_l2)
        index = index_at::l2(dataset.dimensions(), accuracy, config);
    if (args.metric_cos)
        index = index_at::cos(dataset.dimensions(), accuracy, config);
    if (args.metric_haversine)
        index = index_at::haversine(accuracy, config);

    std::printf("- Hardware acceleration: %s\n", isa(index.acceleration()));
    std::printf("-- Will benchmark in-memory\n");

    single_shot(dataset, index, true);
    index.save("tmp.usearch");

    std::printf("-------\n");
    std::printf("-- Will benchmark an on-disk view\n");

    index_at index_view = index.fork();
    index_view.view("tmp.usearch");
    single_shot(dataset, index_view, false);
}

template <typename index_at, typename dataset_at> //
void run_typed(dataset_at& dataset, config_t config) {

    index_at index(config);
    std::printf("-- Will benchmark in-memory\n");

    single_shot(dataset, index, true);
    index.save("tmp.usearch");

    std::printf("-------\n");
    std::printf("-- Will benchmark an on-disk view\n");

    auto_index_t index_view;
    index_view.view("tmp.usearch");
    single_shot(dataset, index_view, false);
}

template <typename neighbor_id_at, typename dataset_at> //
void run_big_or_small(dataset_at& dataset, args_t const& args, config_t config) {
    if (args.native) {
        if (args.metric_cos)
            run_typed<index_gt<ip_gt<float>, std::size_t, neighbor_id_at>>(dataset, config);
        else if (args.metric_l2)
            run_typed<index_gt<l2_squared_gt<float>, std::size_t, neighbor_id_at>>(dataset, config);
        else if (args.metric_haversine)
            run_typed<index_gt<haversine_gt<float>, std::size_t, neighbor_id_at>>(dataset, config);
        else
            run_typed<index_gt<ip_gt<float>, std::size_t, neighbor_id_at>>(dataset, config);
    } else
        run_type_punned<auto_index_gt<std::size_t, neighbor_id_at>>(dataset, args, config);
}

int main(int argc, char** argv) {

    // Print backtrace if something goes wrong.
    signal(SIGSEGV, handler);

    using namespace clipp;

    auto args = args_t{};
    auto cli = ( //
        (option("--vectors") & value("path", args.path_vectors)).doc(".fbin file path to construct the index"),
        (option("--queries") & value("path", args.path_queries)).doc(".fbin file path to query the index"),
        (option("--neighbors") & value("path", args.path_neighbors)).doc(".ibin file path with ground truth"),
        (option("-b", "--big").set(args.big)).doc("Will switch to uint40_t for neighbors lists with over 4B entries"),
        (option("-j", "--threads") & value("integer", args.threads)).doc("Uses all available cores by default"),
        (option("-c", "--connectivity") & value("integer", args.connectivity)).doc("Index granularity"),
        (option("--expansion-add") & value("integer", args.expansion_add)).doc("Affects indexing depth"),
        (option("--expansion-search") & value("integer", args.expansion_search)).doc("Affects search depth"),
        ( //
            option("--native").set(args.native).doc("Use raw templates instead of type-punned classes") |
            option("--f16quant").set(args.quantize_f16).doc("Enable `f16_t` quantization") |
            option("--i8quant").set(args.quantize_i8).doc("Enable `int8_t` quantization")),
        ( //
            option("--ip").set(args.metric_ip).doc("Choose Inner Product metric") |
            option("--l2").set(args.metric_l2).doc("Choose L2 Euclidean metric") |
            option("--cos").set(args.metric_cos).doc("Choose Angular metric") |
            option("--haversine").set(args.metric_haversine).doc("Choose Haversine metric")),
        option("-h", "--help").set(args.help).doc("Print this help information on this tool and exit"));

    if (!parse(argc, argv, cli)) {
        std::cerr << make_man_page(cli, argv[0]);
        exit(1);
    }
    if (args.help) {
        std::cout << make_man_page(cli, argv[0]);
        exit(0);
    }

    // Instead of relying on `multithreaded` from "advanced.hpp" we will use OpenMP
    // to better estimate statistics between tasks batches, without having to recreate
    // the threads.
    omp_set_dynamic(true);
    omp_set_num_threads(args.threads);
    std::printf("- OpenMP threads: %d\n", omp_get_max_threads());

    std::printf("- Dataset: \n");
    std::printf("-- Base vectors path: %s\n", args.path_vectors.c_str());
    std::printf("-- Query vectors path: %s\n", args.path_queries.c_str());
    std::printf("-- Ground truth neighbors path: %s\n", args.path_neighbors.c_str());

    using vector_id_t = std::uint32_t;

    persisted_dataset_gt<float, vector_id_t> dataset{
        args.path_vectors.c_str(),
        args.path_queries.c_str(),
        args.path_neighbors.c_str(),
    };
    std::printf("-- Dimensions: %zu\n", dataset.dimensions());
    std::printf("-- Vectors count: %zu\n", dataset.vectors_count());
    std::printf("-- Queries count: %zu\n", dataset.queries_count());
    std::printf("-- Neighbors per query: %zu\n", dataset.neighborhood_size());

    config_t config;
    config.connectivity = args.connectivity;
    config.expansion_add = args.expansion_add;
    config.expansion_search = args.expansion_search;
    config.max_threads_add = config.max_threads_search = args.threads;
    config.max_elements = dataset.vectors_count();

    if (args.big)
        run_big_or_small<uint40_t>(dataset, args, config);
    else
        run_big_or_small<std::uint32_t>(dataset, args, config);

    return 0;
}