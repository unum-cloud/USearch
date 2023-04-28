/**
 * @brief A benchmark for the construction speed of the USearch index
 * and the resulting accuracy (recall) of the Approximate Nearest Neighbors
 * Search queries.
 */
#include <fcntl.h>    // `open
#include <omp.h>      // `omp_get_num_threads()`
#include <pthread.h>  // `pthread_setaffinity_np`
#include <stdlib.h>   // `getenv`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `stat`

#include <csignal>
#include <cstdio>
#include <execinfo.h>
#include <unistd.h>

#include <algorithm>     // `std::generate`
#include <barrier>       // `std::barrier`
#include <cstring>       // `std::strlen`
#include <mutex>         // `std::mutex`
#include <numeric>       // `std::iota`
#include <random>        // `std::random_device`
#include <stdexcept>     // `std::invalid_argument`
#include <thread>        // `std::thread::hardware_concurrency()`
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <variant>       // `std::monostate`

#include <usearch/usearch.hpp>

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

    in_memory_dataset_gt(std::size_t dimensions, std::size_t vectors_count, std::size_t queries_count,
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
void index_many(index_at& native, std::size_t n, vector_id_at const* ids, real_at const* vecs) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        native.add(ids[i], vecs + native.dim() * i, native.dim(), omp_get_thread_num(), false);
        if (((i + 1) % 100000) == 0)
            std::printf("- added point # %zu\n", i + 1);
    }
}

template <typename index_at, typename vector_id_at, typename real_at>
void search_many(index_at& native, std::size_t n, real_at const* vecs, std::size_t k, vector_id_at* ids,
                 real_at* distances) {
#pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t j = 0;
        native.search(vecs + native.dim() * i, native.dim(), k, omp_get_thread_num(),
                      [&](vector_id_at id, real_at distance) {
                          ids[k * i + j] = id;
                          distances[k * i + j] = distance;
                          j++;
                      });
        std::reverse(ids + k * i, ids + k * i + j);
        std::reverse(distances + k * i, distances + k * i + j);
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

template <typename distance_function_at>
static float type_punned_distance_function(void const* a, void const* b, dim_t a_dim, dim_t b_dim) noexcept {
    using scalar_t = typename distance_function_at::scalar_t;
    return distance_function_at{}((scalar_t const*)a, (scalar_t const*)b, a_dim, b_dim);
}

int main(int, char**) {

    signal(SIGSEGV, handler); // install our handler

    omp_set_dynamic(true);
    omp_set_num_threads(std::thread::hardware_concurrency());
    std::printf("- OpenMP threads: %d\n", omp_get_max_threads());

    auto path_vectors = getenv_or("path_vectors", "");
    auto path_queries = getenv_or("path_queries", path_vectors);
    auto path_neighbors = getenv_or("path_neighbors", "");
    std::printf("- Dataset: \n");
    std::printf("-- Base vectors path: %s\n", path_vectors);
    std::printf("-- Query vectors path: %s\n", path_queries);
    std::printf("-- Ground truth neighbors path: %s\n", path_neighbors);
    persisted_dataset_gt<float, unsigned> dataset{path_vectors, path_queries, path_neighbors};

    std::printf("-- Dimensions: %zu\n", dataset.dimensions());
    std::printf("-- Vectors count: %zu\n", dataset.vectors_count());
    std::printf("-- Queries count: %zu\n", dataset.queries_count());
    std::printf("-- Neighbors per query: %zu\n", dataset.neighborhood_size());

    // We can forward SimSIMD functions:
    // struct simsimd_f32_t {
    //     inline real_t operator()(real_t const* a, real_t const* b, dim_t d, dim_t) const noexcept {
    //         return 1 - simsimd_dot_f32sve(a, b, d);
    //     }
    // };

    {
        using vector_id_t = unsigned;
        using real_t = float;
        using index_t = index_gt<ip_gt<real_t>, vector_id_t, std::uint32_t, real_t, std::allocator<char>>;

        config_t config;
        config.connectivity = 16;
        config.max_threads_add = config.max_threads_search = omp_get_max_threads();
        config.max_elements = dataset.vectors_count();
        config.dim = dataset.dimensions();
        index_t index(config);
        single_shot(dataset, index, true);
        index.save("index.usearch");

        index_t index_copy;
        index_copy.load("index.usearch");
        single_shot(dataset, index_copy, false);

        index_t index_view;
        index_view.view("index.usearch");
        single_shot(dataset, index_view, false);
    }

    {
        using index_t = index_gt<ip_gt<float>, std::size_t, uint40_t>;

        index_t index;
        index.reserve(2);

        float vec[2]{4, 5};
        index.add(10, &vec[0], 2, 0, true);
        index.add(11, &vec[0], 2, 0, false);
        index.save("index40.usearch");

        index_t index_copy;
        index_copy.load("index40.usearch");
        single_shot(dataset, index_copy, false);

        index_t index_view;
        index_view.view("index40.usearch");
        single_shot(dataset, index_view, false);
    }

    return 0;
}