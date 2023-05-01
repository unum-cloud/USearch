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

#include "advanced.hpp"

using namespace unum::usearch;
using namespace unum;

namespace py = pybind11;

using label_t = Py_ssize_t;
using distance_t = punned_distance_t;
using index_t = auto_index_gt<label_t>;

static index_t make_index(          //
    std::size_t dimensions,         //
    std::size_t capacity,           //
    std::string const& scalar_type, //
    std::string const& metric,      //
    std::size_t connectivity,       //
    std::size_t expansion_add,      //
    std::size_t expansion_search    //
) {

    config_t config;
    config.expansion_add = expansion_add;
    config.expansion_search = expansion_search;
    config.connectivity = connectivity;
    config.max_elements = capacity;
    config.max_threads_add = std::thread::hardware_concurrency();
    config.max_threads_search = std::thread::hardware_concurrency();

    accuracy_t accuracy;
    if (scalar_type == "f32")
        accuracy = accuracy_t::f32_k;
    else if (scalar_type == "f64")
        accuracy = accuracy_t::f64_k;
    else if (scalar_type == "f16")
        accuracy = accuracy_t::f16_k;
    else if (scalar_type == "i8q100")
        accuracy = accuracy_t::i8q100_k;
    else
        throw std::runtime_error("Unknown type, choose: f32, f16, f64, i8q100");

    if (metric == "l2_sq" || metric == "euclidean_sq")
        return index_t::l2(dimensions, accuracy, config);
    else if (metric == "ip" || metric == "inner" || metric == "dot")
        return index_t::ip(dimensions, accuracy, config);
    else if (metric == "cos" || metric == "angular")
        return index_t::cos(dimensions, accuracy, config);
    else if (metric == "haversine")
        return index_t::haversine(accuracy, config);
    else
        throw std::runtime_error("Unknown distance, choose: l2_sq, ip, cos, hamming, jaccard");

    return {};
}

static void add_to_index(index_t& index, py::buffer labels, py::buffer vectors, bool copy) {

    py::buffer_info labels_info = labels.request();
    py::buffer_info vectors_info = vectors.request();

    if (labels_info.format != py::format_descriptor<label_t>::format())
        throw std::runtime_error("Incompatible label type!");

    if (labels_info.ndim != 1)
        throw std::runtime_error("Labels must be placed in a single-dimensional array!");

    if (vectors_info.ndim != 2)
        throw std::runtime_error("Expects a matrix of vectors to add!");

    ssize_t labels_count = labels_info.shape[0];
    ssize_t vectors_count = vectors_info.shape[0];
    ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != index.dimensions())
        throw std::runtime_error("The number of vector dimensions doesn't match!");

    if (labels_count != vectors_count)
        throw std::runtime_error("Number of labels and vectors must match!");

    if (index.size() + vectors_count >= index.capacity()) {
        std::size_t next_capacity = ceil2(index.size() + vectors_count);
        index.reserve(next_capacity);
    }

    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);
    char const* labels_data = reinterpret_cast<char const*>(labels_info.ptr);

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "e")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            f16_converted_t const* vector =
                reinterpret_cast<f16_converted_t const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, thread_idx, copy);
        });
    else if (vectors_info.format == "f")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            float const* vector = reinterpret_cast<float const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, thread_idx, copy);
        });
    else if (vectors_info.format == "d")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            label_t label = *reinterpret_cast<label_t const*>(labels_data + task_idx * labels_info.strides[0]);
            double const* vector = reinterpret_cast<double const*>(vectors_data + task_idx * vectors_info.strides[0]);
            index.add(label, vector, thread_idx, copy);
        });
    else
        throw std::runtime_error("Incompatible scalars in the vectors matrix!");
}

/**
 *  @param vectors Matrix of vectors to search for..
 *  @param wanted Number of matches per request.
 *
 *  @return Tuple with:
 *      1. matrix of neighbors,
 *      2. matrix of distances,
 *      3. array with match counts.
 */
static py::tuple search_in_index(index_t& index, py::buffer vectors, ssize_t wanted) {

    py::buffer_info vectors_info = vectors.request();
    if (vectors_info.ndim != 2)
        throw std::runtime_error("Expects a matrix of vectors to add!");

    ssize_t vectors_count = vectors_info.shape[0];
    ssize_t vectors_dimensions = vectors_info.shape[1];
    if (vectors_dimensions != index.dimensions())
        throw std::runtime_error("The number of vector dimensions doesn't match!");

    py::array_t<label_t> labels_py({vectors_count, wanted});
    py::array_t<distance_t> distances_py({vectors_count, wanted});
    py::array_t<Py_ssize_t> counts_py(vectors_count);
    auto labels_py2d = labels_py.mutable_unchecked<2>();
    auto distances_py2d = distances_py.mutable_unchecked<2>();
    auto counts_py1d = counts_py.mutable_unchecked<1>();

    char const* vectors_data = reinterpret_cast<char const*>(vectors_info.ptr);

    // https://docs.python.org/3/library/struct.html#format-characters
    if (vectors_info.format == "e")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            f16_converted_t const* vector = (f16_converted_t const*)(vectors_data + task_idx * vectors_info.strides[0]);
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(
                index.search(vector, wanted, &labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0), thread_idx));
        });
    else if (vectors_info.format == "f")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            float const* vector = (float const*)(vectors_data + task_idx * vectors_info.strides[0]);
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(
                index.search(vector, wanted, &labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0), thread_idx));
        });
    else if (vectors_info.format == "d")
        multithreaded(index.concurrency(), vectors_count, [&](std::size_t thread_idx, std::size_t task_idx) {
            double const* vector = (double const*)(vectors_data + task_idx * vectors_info.strides[0]);
            counts_py1d(task_idx) = static_cast<Py_ssize_t>(
                index.search(vector, wanted, &labels_py2d(task_idx, 0), &distances_py2d(task_idx, 0), thread_idx));
        });
    else
        throw std::runtime_error("Incompatible scalars in the query matrix!");

    py::tuple results(3);
    results[0] = labels_py;
    results[1] = distances_py;
    results[2] = counts_py;
    return results;
}

static void save_index(index_t const& index, std::string const& path) { index.save(path.c_str()); }
static void load_index(index_t& index, std::string const& path) { index.load(path.c_str()); }
static void view_index(index_t& index, std::string const& path) { index.view(path.c_str()); }

PYBIND11_MODULE(usearch, m) {
    m.doc() = "Unum USearch Python bindings";

    auto i = py::class_<index_t>(m, "Index");

    i.def(py::init(&make_index),                                                 //
          py::kw_only(),                                                         //
          py::arg("dim"),                                                        //
          py::arg("capacity") = 0,                                               //
          py::arg("dtype") = std::string("f32"),                                 //
          py::arg("metric") = std::string("ip"),                                 //
          py::arg("connectivity") = config_t::connectivity_default_k,            //
          py::arg("expansion_add") = config_t::expansion_construction_default_k, //
          py::arg("expansion_search") = config_t::expansion_search_default_k     //
    );

    i.def(                     //
        "add", &add_to_index,  //
        py::arg("labels"),     //
        py::arg("vectors"),    //
        py::kw_only(),         //
        py::arg("copy") = true //
    );

    i.def(                          //
        "search", &search_in_index, //
        py::arg("vectors"),         //
        py::arg("count") = 10       //
    );

    i.def("save", &save_index, py::arg("path"));
    i.def("load", &load_index, py::arg("path"));
    i.def("view", &view_index, py::arg("path"));
}
