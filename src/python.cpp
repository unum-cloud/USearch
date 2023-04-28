/**
 *  @file python.cpp
 *  @author Ashot Vardanian
 *  @brief Python bindings for Unum USearch.
 *  @date 2023-04-26
 *
 *
 *  https://pythoncapi.readthedocs.io/type_object.html
 *  https://numpy.org/doc/stable/reference/c-api/types-and-structures.html
 *  https://pythonextensionpatterns.readthedocs.io/en/latest/refcount.html
 *  https://docs.python.org/3/extending/newtypes_tutorial.html#adding-data-and-methods-to-the-basic-example
 *
 *  @copyright Copyright (c) 2023
 */
#define PY_SSIZE_T_CLEAN
#include <thread>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <usearch/usearch.hpp>

#include "advanced.hpp"

using namespace unum::usearch;
using namespace unum;

namespace py = pybind11;

class py_search_api_t {
  public:
    virtual ~py_search_api_t() {}

    virtual void add(py::buffer labels, py::buffer vectors, bool copy) = 0;
    virtual py::tuple search(py::buffer vectors, ssize_t count) = 0;

    virtual void set_dimensions(ssize_t) = 0;
    virtual void set_distance(py::function) = 0;
    virtual void set_distance(std::string const& name) = 0;

    virtual void save(std::string const& path) const = 0;
    virtual void load(std::string const& path) = 0;
    virtual void view(std::string const& path) = 0;
};

template <typename distance_function_at>
static float type_punned_distance_function(void const* a, void const* b, dim_t a_dim, dim_t b_dim) noexcept {
    using scalar_t = typename distance_function_at::scalar_t;
    return distance_function_at{}((scalar_t const*)a, (scalar_t const*)b, a_dim, b_dim);
}

template <typename scalar_at = float, typename neighbor_at = std::uint32_t> //
class py_index_gt final : public py_search_api_t {

    using scalar_t = scalar_at;
    using label_t = Py_ssize_t;
    using neighbor_t = neighbor_at;

    using distance_t = float;
    using distance_function_t = distance_t (*)(void const*, void const*, dim_t, dim_t);
    using index_t = index_gt<distance_function_t, label_t, neighbor_t, scalar_t, aligned_allocator_gt<char>>;

    index_t native_;
    std::string distance_name_;

  public:
    ~py_index_gt() override {}

    py_index_gt(config_t config = {}) : native_(config) { set_distance("ip"); }

    void set_dimensions(ssize_t n) override {
        native_.adjust_dimensions(n);
        if (!distance_name_.empty())
            set_distance(distance_name_);
    }

    void set_distance(py::function) override { distance_name_.clear(); }

    void set_distance(std::string const& name) override {
        distance_name_ = name;
        if (name == "l2_sq" || name == "euclidean_sq") {
            distance_function_t dist = &type_punned_distance_function<l2_squared_gt<scalar_t>>;
            native_.adjust_metric(dist);
        } else if (name == "ip" || name == "inner" || name == "dot") {
            distance_function_t dist = &type_punned_distance_function<ip_gt<scalar_t>>;
            native_.adjust_metric(dist);
        } else if (name == "cos" || name == "angular") {
            distance_function_t dist = &type_punned_distance_function<cos_gt<scalar_t>>;
            native_.adjust_metric(dist);
        } else if (name == "hamming") {
            using allowed_t = typename std::conditional<std::is_unsigned<scalar_t>::value, scalar_t, unsigned>::type;
            distance_function_t dist = &type_punned_distance_function<bit_hamming_gt<allowed_t>>;
            native_.adjust_metric(dist);
        } else if (name == "jaccard") {
            using allowed_t = typename std::conditional<std::is_integral<scalar_t>::value, scalar_t, unsigned>::type;
            distance_function_t dist = &type_punned_distance_function<jaccard_gt<allowed_t>>;
            native_.adjust_metric(dist);
        } else if (name == "haversine") {
            using allowed_t = typename std::conditional<std::is_floating_point<scalar_t>::value, scalar_t, float>::type;
            distance_function_t dist = &type_punned_distance_function<haversine_gt<allowed_t>>;
            native_.adjust_metric(dist);
        } else
            throw std::runtime_error("Unknown distance! Supported: l2_sq, ip, cos, hamming, jaccard");
    }

    void add(py::buffer labels, py::buffer vectors, bool copy) override {

        py::buffer_info labels_info = labels.request();
        py::buffer_info vectors_info = vectors.request();

        if (labels_info.format != py::format_descriptor<label_t>::format())
            throw std::runtime_error("Incompatible label type!");

        if (labels_info.ndim != 1)
            throw std::runtime_error("Labels must be placed in a single-dimensional array!");

        if (vectors_info.format != py::format_descriptor<scalar_t>::format())
            throw std::runtime_error("Incompatible scalars in the vectors matrix!");

        if (vectors_info.ndim != 2)
            throw std::runtime_error("Expects a matrix of vectors to add!");

        ssize_t labels_count = labels_info.shape[0];
        ssize_t vectors_count = vectors_info.shape[0];
        ssize_t vectors_dimensions = vectors_info.shape[1];

        if (labels_count != vectors_count)
            throw std::runtime_error("Number of labels and vectors must match!");

        if (native_.size() + vectors_count >= native_.capacity()) {
            std::size_t next_capacity = ceil2(native_.size() + vectors_count);
            native_.reserve(next_capacity);
        }

        char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
        char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

        multithreaded(native_.max_threads_add(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            scalar_t const* vector =
                reinterpret_cast<scalar_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            native_.add(label, vector, vectors_dimensions, thread_idx, copy);
        });
    }

    /**
     *  @param vectors Matrix of vectors to search for..
     *  @param matches_wanted Number of matches per request.
     *
     *  @return Tuple with:
     *      1. matrix of neighbors,
     *      2. matrix of distances,
     *      3. array with match counts.
     */
    py::tuple search(py::buffer vectors, ssize_t matches_wanted) override {

        py::buffer_info vectors_info = vectors.request();

        if (vectors_info.format != py::format_descriptor<scalar_t>::format())
            throw std::runtime_error("Incompatible scalars in the vectors matrix!");

        if (vectors_info.ndim != 2)
            throw std::runtime_error("Expects a matrix of vectors to add!");

        ssize_t vectors_count = vectors_info.shape[0];
        ssize_t vectors_dimensions = vectors_info.shape[1];

        char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);

        py::array_t<label_t> labels_py({vectors_count, matches_wanted});
        py::array_t<distance_t> distances_py({vectors_count, matches_wanted});
        py::array_t<Py_ssize_t> counts_py(vectors_count);
        auto labels_py2d = labels_py.mutable_unchecked<2>();
        auto distances_py2d = distances_py.mutable_unchecked<2>();
        auto counts_py1d = counts_py.mutable_unchecked<1>();

        multithreaded(native_.max_threads_add(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            std::size_t matches_found = 0;
            scalar_t const* vector =
                reinterpret_cast<scalar_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            auto callback = [&](label_t id, distance_t distance) {
                labels_py2d(task_idx, matches_found) = id;
                distances_py2d(task_idx, matches_found) = distance;
                matches_found++;
            };
            native_.search(vector, vectors_dimensions, matches_wanted, callback, thread_idx);
            std::reverse(&labels_py2d(task_idx, 0), &labels_py2d(task_idx, matches_found));
            std::reverse(&distances_py2d(task_idx, 0), &distances_py2d(task_idx, matches_found));
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(matches_found);
        });

        py::tuple results(3);
        results[0] = std::move(labels_py);
        results[1] = std::move(distances_py);
        results[2] = std::move(counts_py);
        return results;
    }

    void save(std::string const& path) const override { native_.save(path.c_str()); }
    void load(std::string const& path) override { native_.load(path.c_str()); }
    void view(std::string const& path) override { native_.view(path.c_str()); }
};

using py_index_f16u32_t = py_index_gt<f16_converted_t, std::uint32_t>;
using py_index_f32u32_t = py_index_gt<float, std::uint32_t>;
using py_index_f64u32_t = py_index_gt<double, std::uint32_t>;
using py_index_i32u32_t = py_index_gt<std::int32_t, std::uint32_t>;
using py_index_i8u32_t = py_index_gt<std::int8_t, std::uint32_t>;

using py_index_f16u40_t = py_index_gt<f16_converted_t, uint40_t>;
using py_index_f32u40_t = py_index_gt<float, uint40_t>;
using py_index_f64u40_t = py_index_gt<double, uint40_t>;
using py_index_i32u40_t = py_index_gt<std::int32_t, uint40_t>;
using py_index_i8u40_t = py_index_gt<std::int8_t, uint40_t>;

static std::shared_ptr<py_search_api_t> make_index( //
    std::string const& scalar_type,                 //
    std::size_t expansion_construction,             //
    std::size_t expansion_search,                   //
    std::size_t connectivity,                       //
    dim_t dim,                                      //
    std::size_t capacity,                           //
    bool can_exceed_four_billion) {

    config_t config;
    config.expansion_construction = expansion_construction;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;
    config.max_elements = capacity;
    config.dim = dim;
    config.max_threads_add = std::thread::hardware_concurrency();
    config.max_threads_search = std::thread::hardware_concurrency();

    if (!can_exceed_four_billion) {
        if (scalar_type == "f32")
            return std::make_shared<py_index_f32u32_t>(config);
        if (scalar_type == "f64")
            return std::make_shared<py_index_f64u32_t>(config);
        if (scalar_type == "i32")
            return std::make_shared<py_index_i32u32_t>(config);
        if (scalar_type == "i8")
            return std::make_shared<py_index_i8u32_t>(config);
        // if (scalar_type == "f16")
        //     return std::make_shared<py_index_f16u32_t>(config);
    } else {
        if (scalar_type == "f32")
            return std::make_shared<py_index_f32u40_t>(config);
        if (scalar_type == "f64")
            return std::make_shared<py_index_f64u40_t>(config);
        if (scalar_type == "i32")
            return std::make_shared<py_index_i32u40_t>(config);
        if (scalar_type == "i8")
            return std::make_shared<py_index_i8u40_t>(config);
        // if (scalar_type == "f16")
        //     return std::shared_ptr<py_search_api_t>(new py_index_f16u40_t(config));
    }

    return {};
}

PYBIND11_MODULE(usearch, m) {
    m.doc() = "Unum USearch Python Bindings";

    auto i = py::class_<py_search_api_t>(m, "Index");

    i.def(                            //
        "add", &py_search_api_t::add, //
        py::arg("labels"),            //
        py::arg("vectors"),           //
        py::kw_only(),                //
        py::arg("copy") = true        //
    );

    i.def(                                  //
        "search", &py_search_api_t::search, //
        py::arg("vectors"),                 //
        py::arg("count") = 10               //
    );

    i.def("save", &py_search_api_t::save, py::arg("path"));
    i.def("load", &py_search_api_t::load, py::arg("path"));
    i.def("view", &py_search_api_t::view, py::arg("path"));

    m.def(                                       //
        "make_index", &make_index,               //
        py::kw_only(),                           //
        py::arg("dtype") = std::string("f32"),   //
        py::arg("expansion_construction") = 200, //
        py::arg("expansion_search") = 100,       //
        py::arg("connectivity") = 16,            //
        py::arg("dim") = 0,                      //
        py::arg("capacity") = 0,                 //
        py::arg("big") = false                   //
    );
}
