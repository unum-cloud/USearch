#include "lib.hpp"
#include "usearch/rust/lib.rs.h"

using namespace unum::usearch;
using namespace unum;

using index_t = index_dense_t;
using add_result_t = typename index_t::add_result_t;
using search_result_t = typename index_t::search_result_t;
using labeling_result_t = typename index_t::labeling_result_t;
using vector_key_t = typename index_dense_t::vector_key_t;

char const* hardware_acceleration_compiled() { return unum::usearch::hardware_acceleration_compiled(); }
char const* hardware_acceleration_available() { return unum::usearch::hardware_acceleration_available(); }

metric_kind_t rust_to_cpp_metric(MetricKind value) {
    switch (value) {
    case MetricKind::IP: return metric_kind_t::ip_k;
    case MetricKind::L2sq: return metric_kind_t::l2sq_k;
    case MetricKind::Cos: return metric_kind_t::cos_k;
    case MetricKind::Pearson: return metric_kind_t::pearson_k;
    case MetricKind::Haversine: return metric_kind_t::haversine_k;
    case MetricKind::Divergence: return metric_kind_t::divergence_k;
    case MetricKind::Hamming: return metric_kind_t::hamming_k;
    case MetricKind::Tanimoto: return metric_kind_t::tanimoto_k;
    case MetricKind::Sorensen: return metric_kind_t::sorensen_k;
    default: return metric_kind_t::unknown_k;
    }
}

scalar_kind_t rust_to_cpp_scalar(ScalarKind value) {
    switch (value) {
    case ScalarKind::F64: return scalar_kind_t::f64_k;
    case ScalarKind::F32: return scalar_kind_t::f32_k;
    case ScalarKind::BF16: return scalar_kind_t::bf16_k;
    case ScalarKind::F16: return scalar_kind_t::f16_k;
    case ScalarKind::E5M2: return scalar_kind_t::e5m2_k;
    case ScalarKind::E4M3: return scalar_kind_t::e4m3_k;
    case ScalarKind::E3M2: return scalar_kind_t::e3m2_k;
    case ScalarKind::E2M3: return scalar_kind_t::e2m3_k;
    case ScalarKind::I8: return scalar_kind_t::i8_k;
    case ScalarKind::U8: return scalar_kind_t::u8_k;
    case ScalarKind::B1: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

MetricKind cpp_to_rust_metric(metric_kind_t value) {
    switch (value) {
    case metric_kind_t::ip_k: return MetricKind::IP;
    case metric_kind_t::l2sq_k: return MetricKind::L2sq;
    case metric_kind_t::cos_k: return MetricKind::Cos;
    case metric_kind_t::pearson_k: return MetricKind::Pearson;
    case metric_kind_t::haversine_k: return MetricKind::Haversine;
    case metric_kind_t::divergence_k: return MetricKind::Divergence;
    case metric_kind_t::hamming_k: return MetricKind::Hamming;
    case metric_kind_t::tanimoto_k: return MetricKind::Tanimoto;
    case metric_kind_t::sorensen_k: return MetricKind::Sorensen;
    default: return MetricKind::Unknown;
    }
}

ScalarKind cpp_to_rust_scalar(scalar_kind_t value) {
    switch (value) {
    case scalar_kind_t::f64_k: return ScalarKind::F64;
    case scalar_kind_t::f32_k: return ScalarKind::F32;
    case scalar_kind_t::bf16_k: return ScalarKind::BF16;
    case scalar_kind_t::f16_k: return ScalarKind::F16;
    case scalar_kind_t::e5m2_k: return ScalarKind::E5M2;
    case scalar_kind_t::e4m3_k: return ScalarKind::E4M3;
    case scalar_kind_t::e3m2_k: return ScalarKind::E3M2;
    case scalar_kind_t::e2m3_k: return ScalarKind::E2M3;
    case scalar_kind_t::i8_k: return ScalarKind::I8;
    case scalar_kind_t::u8_k: return ScalarKind::U8;
    case scalar_kind_t::b1x8_k: return ScalarKind::B1;
    default: return ScalarKind::Unknown;
    }
}

template <typename scalar_at>
Matches search_(index_dense_t& index, scalar_at const* vec, size_t vec_dims, size_t count, bool exact = false) {
    if (vec_dims != index.scalar_words())
        throw std::invalid_argument("Vector length must match index dimensionality");
    Matches matches;
    matches.keys.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.keys.push_back(0), matches.distances.push_back(0);

    search_result_t result = index.search(vec, count, index_dense_t::any_thread(), exact);
    result.error.raise();
    count = result.dump_to(matches.keys.data(), matches.distances.data(), count);
    matches.keys.truncate(count);
    matches.distances.truncate(count);
    return matches;
}

template <typename scalar_at, typename predicate_at>
Matches filtered_search_(index_dense_t& index, scalar_at const* vec, size_t vec_dims, size_t count,
                         predicate_at&& predicate) {
    if (vec_dims != index.scalar_words())
        throw std::invalid_argument("Vector length must match index dimensionality");
    Matches matches;
    matches.keys.reserve(count);
    matches.distances.reserve(count);
    for (size_t i = 0; i != count; ++i)
        matches.keys.push_back(0), matches.distances.push_back(0);

    search_result_t result = index.filtered_search(vec, count, std::forward<predicate_at>(predicate));
    result.error.raise();
    count = result.dump_to(matches.keys.data(), matches.distances.data(), count);
    matches.keys.truncate(count);
    matches.distances.truncate(count);
    return matches;
}

template <typename scalar_at> void add_(index_dense_t& index, vector_key_t key, scalar_at const* vec, size_t vec_dims) {
    if (vec_dims != index.scalar_words())
        throw std::invalid_argument("Vector length must match index dimensionality");
    index.add(key, vec).error.raise();
}

NativeIndex::NativeIndex(std::unique_ptr<index_t> index) : index_(std::move(index)) {}

auto make_predicate(uptr_t metric, uptr_t metric_state) {
    return [=](vector_key_t key) {
        auto func = reinterpret_cast<bool (*)(uptr_t, vector_key_t)>(metric);
        auto state = static_cast<uptr_t>(metric_state);
        return func(key, state);
    };
}

// clang-format off
void NativeIndex::add_f64(vector_key_t key, rust::Slice<double const> vec) const { add_(*index_, key, vec.data(), vec.size()); }
void NativeIndex::add_f32(vector_key_t key, rust::Slice<float const> vec) const { add_(*index_, key, vec.data(), vec.size()); }
void NativeIndex::add_f16(vector_key_t key, rust::Slice<int16_t const> vec) const { add_(*index_, key, (f16_t const*)vec.data(), vec.size()); }
void NativeIndex::add_i8(vector_key_t key, rust::Slice<int8_t const> vec) const { add_(*index_, key, vec.data(), vec.size()); }
void NativeIndex::add_u8(vector_key_t key, rust::Slice<uint8_t const> vec) const { add_(*index_, key, (u8_t const*)vec.data(), vec.size()); }
void NativeIndex::add_b1x8(vector_key_t key, rust::Slice<uint8_t const> vec) const { add_(*index_, key, (b1x8_t const*)vec.data(), vec.size()); }

// Regular approximate search
Matches NativeIndex::search_f64(rust::Slice<double const> vec, size_t count) const { return search_(*index_, vec.data(), vec.size(), count, false); }
Matches NativeIndex::search_f32(rust::Slice<float const> vec, size_t count) const { return search_(*index_, vec.data(), vec.size(), count, false); }
Matches NativeIndex::search_f16(rust::Slice<int16_t const> vec, size_t count) const { return search_(*index_, (f16_t const*)vec.data(), vec.size(), count, false); }
Matches NativeIndex::search_i8(rust::Slice<int8_t const> vec, size_t count) const { return search_(*index_, vec.data(), vec.size(), count, false); }
Matches NativeIndex::search_u8(rust::Slice<uint8_t const> vec, size_t count) const { return search_(*index_, (u8_t const*)vec.data(), vec.size(), count, false); }
Matches NativeIndex::search_b1x8(rust::Slice<uint8_t const> vec, size_t count) const { return search_(*index_, (b1x8_t const*)vec.data(), vec.size(), count, false); }

// Exact (brute force) search
Matches NativeIndex::exact_search_f64(rust::Slice<double const> vec, size_t count) const { return search_(*index_, vec.data(), vec.size(), count, true); }
Matches NativeIndex::exact_search_f32(rust::Slice<float const> vec, size_t count) const { return search_(*index_, vec.data(), vec.size(), count, true); }
Matches NativeIndex::exact_search_f16(rust::Slice<int16_t const> vec, size_t count) const { return search_(*index_, (f16_t const*)vec.data(), vec.size(), count, true); }
Matches NativeIndex::exact_search_i8(rust::Slice<int8_t const> vec, size_t count) const { return search_(*index_, vec.data(), vec.size(), count, true); }
Matches NativeIndex::exact_search_u8(rust::Slice<uint8_t const> vec, size_t count) const { return search_(*index_, (u8_t const*)vec.data(), vec.size(), count, true); }
Matches NativeIndex::exact_search_b1x8(rust::Slice<uint8_t const> vec, size_t count) const { return search_(*index_, (b1x8_t const*)vec.data(), vec.size(), count, true); }

// Filtered search (always approximate)
Matches NativeIndex::filtered_search_f64(rust::Slice<double const> vec, size_t count, uptr_t metric, uptr_t metric_state) const { return filtered_search_(*index_, vec.data(), vec.size(), count, make_predicate(metric, metric_state)); }
Matches NativeIndex::filtered_search_f32(rust::Slice<float const> vec, size_t count, uptr_t metric, uptr_t metric_state) const { return filtered_search_(*index_, vec.data(), vec.size(), count, make_predicate(metric, metric_state)); }
Matches NativeIndex::filtered_search_f16(rust::Slice<int16_t const> vec, size_t count, uptr_t metric, uptr_t metric_state) const { return filtered_search_(*index_, (f16_t const*)vec.data(), vec.size(), count, make_predicate(metric, metric_state)); }
Matches NativeIndex::filtered_search_i8(rust::Slice<int8_t const> vec, size_t count, uptr_t metric, uptr_t metric_state) const { return filtered_search_(*index_, vec.data(), vec.size(), count, make_predicate(metric, metric_state)); }
Matches NativeIndex::filtered_search_u8(rust::Slice<uint8_t const> vec, size_t count, uptr_t metric, uptr_t metric_state) const { return filtered_search_(*index_, (u8_t const*)vec.data(), vec.size(), count, make_predicate(metric, metric_state)); }
Matches NativeIndex::filtered_search_b1x8(rust::Slice<uint8_t const> vec, size_t count, uptr_t metric, uptr_t metric_state) const { return filtered_search_(*index_, (b1x8_t const*)vec.data(), vec.size(), count, make_predicate(metric, metric_state)); }

size_t NativeIndex::get_f64(vector_key_t key, rust::Slice<double> vec) const { if (vec.size() % dimensions()) throw std::invalid_argument("Vector length must match index dimensionality"); return index_->get(key, vec.data(), vec.size() / dimensions()); }
size_t NativeIndex::get_f32(vector_key_t key, rust::Slice<float> vec) const { if (vec.size() % dimensions()) throw std::invalid_argument("Vector length must match index dimensionality"); return index_->get(key, vec.data(), vec.size() / dimensions()); }
size_t NativeIndex::get_f16(vector_key_t key, rust::Slice<int16_t> vec) const { if (vec.size() % dimensions()) throw std::invalid_argument("Vector length must match index dimensionality"); return index_->get(key, (f16_t*)vec.data(), vec.size() / dimensions()); }
size_t NativeIndex::get_i8(vector_key_t key, rust::Slice<int8_t> vec) const { if (vec.size() % dimensions()) throw std::invalid_argument("Vector length must match index dimensionality"); return index_->get(key, vec.data(), vec.size() / dimensions()); }
size_t NativeIndex::get_u8(vector_key_t key, rust::Slice<uint8_t> vec) const { if (vec.size() % dimensions()) throw std::invalid_argument("Vector length must match index dimensionality"); return index_->get(key, (u8_t*)vec.data(), vec.size() / dimensions()); }
size_t NativeIndex::get_b1x8(vector_key_t key, rust::Slice<uint8_t> vec) const { if (vec.size() % dimensions()) throw std::invalid_argument("Vector length must match index dimensionality"); return index_->get(key, (b1x8_t*)vec.data(), vec.size() / dimensions()); }
// clang-format on

void NativeIndex::reserve(size_t capacity) const { index_->reserve(capacity); }
void NativeIndex::reserve_capacity_and_threads(size_t capacity, size_t threads) const {
    index_->reserve({capacity, threads});
}

size_t NativeIndex::expansion_add() const { return index_->expansion_add(); }
size_t NativeIndex::expansion_search() const { return index_->expansion_search(); }
void NativeIndex::change_expansion_add(size_t n) const { index_->change_expansion_add(n); }
void NativeIndex::change_expansion_search(size_t n) const { index_->change_expansion_search(n); }

MetricKind NativeIndex::metric_kind() const { return cpp_to_rust_metric(index_->metric_kind()); }

void NativeIndex::change_metric_kind(MetricKind metric) const {
    index_->change_metric(metric_punned_t::builtin( //
        index_->dimensions(),                       //
        rust_to_cpp_metric(metric),                 //
        index_->scalar_kind()));
}

void NativeIndex::change_metric(uptr_t metric, uptr_t state) const {
    index_->change_metric(metric_punned_t::stateful( //
        index_->dimensions(),                        //
        static_cast<std::uintptr_t>(metric),         //
        static_cast<std::uintptr_t>(state),          //
        index_->metric().metric_kind(),              //
        index_->scalar_kind()));
}

size_t NativeIndex::dimensions() const { return index_->dimensions(); }
size_t NativeIndex::connectivity() const { return index_->connectivity(); }
ScalarKind NativeIndex::scalar_kind() const { return cpp_to_rust_scalar(index_->scalar_kind()); }
bool NativeIndex::multi() const { return index_->multi(); }
size_t NativeIndex::size() const { return index_->size(); }
size_t NativeIndex::capacity() const { return index_->capacity(); }
size_t NativeIndex::serialized_length() const { return index_->serialized_length(); }

size_t NativeIndex::count(vector_key_t key) const { return index_->count(key); }
bool NativeIndex::contains(vector_key_t key) const { return index_->contains(key); }

size_t NativeIndex::level_of_key(vector_key_t key) const { return index_->level_of(key); }

std::size_t NeighborsCursor::size() const noexcept { return view_.size(); }
std::size_t NeighborsCursor::remaining() const noexcept { return view_.size() - position_; }
bool NeighborsCursor::has_next() const noexcept { return position_ < view_.size(); }

NeighborsCursor::vector_key_t NeighborsCursor::next_key() noexcept {
    auto key = static_cast<vector_key_t>(view_[position_].key);
    ++position_;
    return key;
}

std::size_t NeighborsCursor::drain_into(rust::Slice<vector_key_t> output) noexcept {
    std::size_t available = view_.size() - position_;
    std::size_t copied = (std::min)(available, output.size());
    for (std::size_t offset = 0; offset != copied; ++offset)
        output[offset] = static_cast<vector_key_t>(view_[position_ + offset].key);
    position_ += copied;
    return copied;
}

std::unique_ptr<NeighborsCursor> NativeIndex::neighbors(vector_key_t key, size_t level) const {
    return std::unique_ptr<NeighborsCursor>(new NeighborsCursor(index_->neighbors(key, level)));
}

size_t NativeIndex::remove(vector_key_t key) const {
    labeling_result_t result = index_->remove(key);
    result.error.raise();
    return result.completed;
}

size_t NativeIndex::rename(vector_key_t from, vector_key_t to) const {
    labeling_result_t result = index_->rename(from, to);
    result.error.raise();
    return result.completed;
}

void NativeIndex::save(rust::Str path) const { index_->save(output_file_t(std::string(path).c_str())).error.raise(); }
void NativeIndex::load(rust::Str path) const { index_->load(input_file_t(std::string(path).c_str())).error.raise(); }
void NativeIndex::view(rust::Str path) const {
    index_->view(memory_mapped_file_t(std::string(path).c_str())).error.raise();
}

void NativeIndex::reset() const { index_->reset(); }
size_t NativeIndex::memory_usage() const { return index_->memory_usage(); }

MemoryStats NativeIndex::memory_stats() const {
    auto stats = index_->memory_stats();
    MemoryStats result;
    result.graph_allocated = stats.graph_allocated;
    result.graph_wasted = stats.graph_wasted;
    result.graph_reserved = stats.graph_reserved;
    result.vectors_allocated = stats.vectors_allocated;
    result.vectors_wasted = stats.vectors_wasted;
    result.vectors_reserved = stats.vectors_reserved;
    return result;
}

char const* NativeIndex::hardware_acceleration() const { return index_->metric().isa_name(); }

void NativeIndex::save_to_buffer(rust::Slice<uint8_t> buffer) const {
    index_->save(memory_mapped_file_t((byte_t*)buffer.data(), buffer.size())).error.raise();
}

void NativeIndex::load_from_buffer(rust::Slice<uint8_t const> buffer) const {
    index_->load(memory_mapped_file_t((byte_t*)buffer.data(), buffer.size())).error.raise();
}

void NativeIndex::view_from_buffer(rust::Slice<uint8_t const> buffer) const {
    index_->view(memory_mapped_file_t((byte_t*)buffer.data(), buffer.size())).error.raise();
}

std::unique_ptr<NativeIndex> wrap(index_t&& index) {
    std::unique_ptr<index_t> punned_ptr;
    punned_ptr.reset(new index_t(std::move(index)));
    std::unique_ptr<NativeIndex> result;
    result.reset(new NativeIndex(std::move(punned_ptr)));
    return result;
}

std::unique_ptr<NativeIndex> new_native_index(IndexOptions const& options) {
    metric_kind_t metric_kind = rust_to_cpp_metric(options.metric);
    scalar_kind_t scalar_kind = rust_to_cpp_scalar(options.quantization);
    metric_punned_t metric(options.dimensions, metric_kind, scalar_kind);
    if (metric.missing())
        throw std::invalid_argument("Unsupported metric or scalar type");
    index_dense_config_t config(options.connectivity, options.expansion_add, options.expansion_search);
    config.multi = options.multi;
    index_t index = index_t::make(metric, config);

    // Preserve constructor pre-allocation semantics (`index_limits_t{}`), but execute
    // reserve after heap allocation to avoid move-induced pointer invalidation.
    std::unique_ptr<NativeIndex> native = wrap(std::move(index));
    native->reserve_capacity_and_threads(0, std::thread::hardware_concurrency());
    return native;
}

IndexMetadata head_to_metadata(index_dense_head_t const& head) {
    IndexMetadata meta;
    meta.dimensions = static_cast<std::uint64_t>(head.dimensions);
    meta.metric = cpp_to_rust_metric(static_cast<metric_kind_t>(head.kind_metric));
    meta.quantization = cpp_to_rust_scalar(static_cast<scalar_kind_t>(head.kind_scalar));
    meta.multi = static_cast<bool>(head.multi);
    meta.count_present = static_cast<std::uint64_t>(head.count_present);
    meta.count_deleted = static_cast<std::uint64_t>(head.count_deleted);
    meta.version_major = static_cast<std::uint16_t>(head.version_major);
    meta.version_minor = static_cast<std::uint16_t>(head.version_minor);
    meta.version_patch = static_cast<std::uint16_t>(head.version_patch);
    return meta;
}

IndexMetadata read_metadata(rust::Str path) {
    index_dense_metadata_result_t result = index_dense_metadata_from_path(std::string(path).c_str());
    if (!result)
        result.error.raise();
    return head_to_metadata(result.head);
}

IndexMetadata read_metadata_from_buffer(rust::Slice<uint8_t const> buffer) {
    index_dense_metadata_result_t result =
        index_dense_metadata_from_buffer(memory_mapped_file_t((byte_t*)buffer.data(), buffer.size()));
    if (!result)
        result.error.raise();
    return head_to_metadata(result.head);
}
