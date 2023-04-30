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
void index_many(index_at& native, std::size_t n, vector_id_at const* ids, real_at const* vectors) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        native.add(ids[i], vectors + native.dimensions() * i, omp_get_thread_num(), false);
        if (((i + 1) % 100000) == 0)
            std::printf("- added point # %zu\n", i + 1);
    }
}

template <typename index_at, typename vector_id_at, typename real_at>
void search_many(                                         //
    index_at& native,                                     //
    std::size_t n, real_at const* vectors, std::size_t k, //
    vector_id_at* ids, real_at* distances) {

#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t found = native.search(        //
            vectors + native.dimensions() * i, k, //
            ids + k * i, distances + k * i,       //
            omp_get_thread_num());
        std::reverse(ids + k * i, ids + k * i + found);
        std::reverse(distances + k * i, distances + k * i + found);
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
        auto dt_add = nanoseconds([&] { index_many(index, dataset.vectors_count(), ids.data(), dataset.vector(0)); });
        std::printf("- Added %.2f vectors/sec\n", dataset.vectors_count() * 1e9 / dt_add);
    }

    // Perform search, evaluate speed
    std::vector<label_t> found_neighbors(dataset.queries_count() * dataset.neighborhood_size());
    std::vector<distance_t> found_distances(dataset.queries_count() * dataset.neighborhood_size());
    std::size_t executed_search_queries = 0;

    auto dt_search = nanoseconds([&] {
        while (executed_search_queries < dataset.vectors_count()) {
            search_many(index, dataset.queries_count(), dataset.query(0), dataset.neighborhood_size(),
                        found_neighbors.data(), found_distances.data());
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

int main(int argc, char** argv) {

    // Print backtrace if something goes wrong.
    signal(SIGSEGV, handler);

    using namespace clipp;

    std::string path_vectors;
    std::string path_queries;
    std::string path_neighbors;
    std::size_t threads = std::thread::hardware_concurrency();
    std::size_t connectivity = config_t{}.connectivity;
    bool help = false;
    bool quantize_f16 = false;
    bool quantize_i8 = false;
    bool metric_ip = true;
    bool metric_l2 = false;
    bool metric_cos = false;
    bool metric_haversine = false;

    auto cli = ( //
        (option("--vectors") & value("path", path_vectors)).doc(".fbin file path to construct the index"),
        (option("--queries") & value("path", path_queries)).doc(".fbin file path to query the index"),
        (option("--neighbors") & value("path", path_neighbors)).doc(".ibin file path with ground truth"),
        (option("-j", "--threads") & value("threads", threads)).doc("Uses all available cores by default"),
        (option("-c", "--connectivity") & value("connectivity", connectivity)).doc("Index granularity"),
        ( //
            option("--f16quant").set(quantize_f16).doc("Enable half-precision quantization") |
            option("--i8quant").set(quantize_i8).doc("Enable int8_t quantization")),
        ( //
            option("--ip").set(metric_ip).doc("Choose Inner Product metric") |
            option("--l2").set(metric_l2).doc("Choose L2 Euclidean metric") |
            option("--cos").set(metric_cos).doc("Choose Angular metric") |
            option("--haversine").set(metric_haversine).doc("Choose Haversine metric")),
        option("-h", "--help").set(help).doc("Print this help information on this tool and exit"));

    if (!parse(argc, argv, cli)) {
        std::cerr << make_man_page(cli, argv[0]);
        exit(1);
    }
    if (help) {
        std::cout << make_man_page(cli, argv[0]);
        exit(0);
    }

    // Instead of relying on `multithreaded` from "advanced.hpp" we will use OpenMP
    // to better estimate statistics between tasks batches, without having to recreate
    // the threads.
    omp_set_dynamic(true);
    omp_set_num_threads(threads);
    std::printf("- OpenMP threads: %d\n", omp_get_max_threads());

    std::printf("- Dataset: \n");
    std::printf("-- Base vectors path: %s\n", path_vectors.c_str());
    std::printf("-- Query vectors path: %s\n", path_queries.c_str());
    std::printf("-- Ground truth neighbors path: %s\n", path_neighbors.c_str());

    using vector_id_t = std::uint32_t;
    using real_t = float;

    persisted_dataset_gt<float, vector_id_t> dataset{
        path_vectors.c_str(),
        path_queries.c_str(),
        path_neighbors.c_str(),
    };
    std::printf("-- Dimensions: %zu\n", dataset.dimensions());
    std::printf("-- Vectors count: %zu\n", dataset.vectors_count());
    std::printf("-- Queries count: %zu\n", dataset.queries_count());
    std::printf("-- Neighbors per query: %zu\n", dataset.neighborhood_size());

    config_t config;
    config.connectivity = connectivity;
    config.max_threads_add = config.max_threads_search = threads;
    config.max_elements = dataset.vectors_count();

    accuracy_t accuracy = accuracy_t::f32_k;
    if (quantize_f16)
        accuracy = accuracy_t::f16_k;
    if (quantize_i8)
        accuracy = accuracy_t::i8q100_k;

    auto_index_t index;
    if (metric_ip)
        index = auto_index_t::ip(dataset.dimensions(), accuracy, config);
    if (metric_l2)
        index = auto_index_t::l2(dataset.dimensions(), accuracy, config);
    if (metric_cos)
        index = auto_index_t::cos(dataset.dimensions(), accuracy, config);
    if (metric_haversine)
        index = auto_index_t::haversine(accuracy, config);

    single_shot(dataset, index, true);
    index.save("tmp/index.usearch");

    auto_index_t index_copy;
    index_copy.load("tmp/index.usearch");
    single_shot(dataset, index_copy, false);

    auto_index_t index_view;
    index_view.view("tmp/index.usearch");
    single_shot(dataset, index_view, false);

    // Test compilation of more obscure index types.
    {
        using index_t = index_gt<ip_gt<float>, std::size_t, uint40_t>;
        index_t index;
        index.reserve(2);
        float vec[2]{4, 5};
        index.add(10, &vec[0], 2);
        index.add(11, &vec[0], 2);
        index.search(&vec[0], 2, 10, [](std::size_t, float) {});
    }

    return 0;
}