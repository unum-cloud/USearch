/**
 *  @brief JNI boundary of the Java SDK, translating between the JVM and the flat C interface
 *      of @c include/usearch/dense_index.h.
 *
 *  @author Ash Vardanian
 *  @file java/lib.cpp
 *
 *  This translation unit instantiates no index template. It links the pre-instantiated library
 *  and does nothing but marshal: pin a slab, lower an enumerator to its code, hand the call
 *  across, lift the outcome back. Every verb the Java side declares as @c native appears here
 *  exactly once, and the two agree by symbol name rather than by a generated header.
 *
 *  Java's signed 32-bit @c int wraps four billion times sooner than a @c size_t, so every extent
 *  a caller supplies is checked on the Java side before it reaches an allocation, and checked
 *  again here for sign and for fit before it narrows into a @c size_t.
 *
 *  Slabs are addressed through JNI's critical region, which forbids any other JNI call while it
 *  is open. Every call below therefore resolves each slab's kind first - a direct buffer already
 *  carries an address, a primitive array does not - and only then pins them back to back.
 */
#include <cstddef> // `std::size_t`
#include <cstdint> // `std::int32_t`

#include <vector> // `std::vector`

#include <jni.h>

#include <usearch/dense_index.h>

#pragma region Cached Class Handles

namespace {

/** @brief @c java.nio.ByteBuffer, distinguishing a direct buffer from a primitive array. */
jclass byte_buffer_class = nullptr;
/** @brief @c cloud.unum.usearch.IndexException, the one failure type this boundary raises. */
jclass exception_class = nullptr;
/** @brief The static factory building an exception from a status code and a message. */
jmethodID exception_factory = nullptr;

} // namespace

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* machine, void*) {
    JNIEnv* environment = nullptr;
    if (machine->GetEnv((void**)&environment, JNI_VERSION_10) != JNI_OK) return JNI_ERR;

    jclass const buffer = environment->FindClass("java/nio/ByteBuffer");
    jclass const failure = environment->FindClass("cloud/unum/usearch/IndexException");
    if (!buffer || !failure) return JNI_ERR;

    byte_buffer_class = (jclass)environment->NewGlobalRef(buffer);
    exception_class = (jclass)environment->NewGlobalRef(failure);
    exception_factory = environment->GetStaticMethodID( //
        exception_class, "of", "(ILjava/lang/String;)Lcloud/unum/usearch/IndexException;");
    if (!byte_buffer_class || !exception_class || !exception_factory) return JNI_ERR;
    return JNI_VERSION_10;
}

extern "C" JNIEXPORT void JNICALL JNI_OnUnload(JavaVM* machine, void*) {
    JNIEnv* environment = nullptr;
    if (machine->GetEnv((void**)&environment, JNI_VERSION_10) != JNI_OK) return;
    if (byte_buffer_class) environment->DeleteGlobalRef(byte_buffer_class);
    if (exception_class) environment->DeleteGlobalRef(exception_class);
    byte_buffer_class = nullptr;
    exception_class = nullptr;
    exception_factory = nullptr;
}

#pragma endregion Cached Class Handles

#pragma region Failure Reporting

namespace {

/** @brief Raises the Java failure a status code and its message describe. */
void raise(JNIEnv* environment, usearch_status_t status, char const* message) {
    if (environment->ExceptionCheck()) return;
    jstring const text = environment->NewStringUTF(message ? message : "");
    jobject const failure = environment->CallStaticObjectMethod(exception_class, exception_factory, (jint)status, text);
    if (failure) environment->Throw((jthrowable)failure);
}

/** @brief Raises the failure a populated error slot carries; reports whether it raised. */
bool raised(JNIEnv* environment, usearch_error_t const& error) {
    if (error.status == usearch_success_k) return false;
    raise(environment, error.status, error.message);
    return true;
}

/** @brief Narrows a Java extent into a @c size_t, raising rather than wrapping on a negative. */
bool narrowed(JNIEnv* environment, jlong extent, char const* what, std::size_t& into) {
    if (extent < 0) {
        raise(environment, usearch_error_shape_mismatch_k, what);
        return false;
    }
    into = (std::size_t)extent;
    return true;
}

/**
 *  @brief Multiplies two extents, raising @c usearch_error_overflow_k rather than wrapping.
 *      The Java side sized its slabs against the same product, so a wrap here would let the
 *      engine write past a buffer that looks long enough.
 */
bool multiplied(JNIEnv* environment, std::size_t left, std::size_t right, char const* what, std::size_t& into) {
    if (left && right > (std::size_t)-1 / left) {
        raise(environment, usearch_error_overflow_k, what);
        return false;
    }
    into = left * right;
    return true;
}

} // namespace

#pragma endregion Failure Reporting

#pragma region Slab Pinning

namespace {

/**
 *  @brief A caller's slab with its kind already resolved: a direct buffer carries an address, a
 *      primitive array carries the handle a later pin turns into one. Resolving costs JNI calls,
 *      which is exactly what a critical region forbids, so it happens before any slab is pinned.
 */
struct slab_plan_t {
    jarray array = nullptr;
    void* direct = nullptr;
    jlong offset = 0;

    /** @brief Resolves how a slab will be addressed; JNI calls are still allowed here. */
    slab_plan_t(JNIEnv* environment, jobject storage, jlong byte_offset) : offset(byte_offset) {
        if (!storage) return;
        if (environment->IsInstanceOf(storage, byte_buffer_class))
            direct = environment->GetDirectBufferAddress(storage);
        else array = (jarray)storage;
    }

    /** @brief Whether the slab names storage at all - a null argument or a heap buffer does not. */
    bool addressable() const noexcept { return array != nullptr || direct != nullptr; }
};

/**
 *  @brief The critical region every batch call opens: the planned slabs pinned back to back with
 *      no other JNI call between them, and released in reverse when the region closes.
 */
struct pinned_slabs_t {
    static constexpr std::size_t capacity_k = 4;

    JNIEnv* environment = nullptr;
    jarray arrays[capacity_k] = {};
    void* bases[capacity_k] = {};
    std::size_t count = 0;
    bool complete = true;

    explicit pinned_slabs_t(JNIEnv* environment) noexcept : environment(environment) {}

    ~pinned_slabs_t() {
        while (count) {
            --count;
            if (arrays[count] && bases[count])
                environment->ReleasePrimitiveArrayCritical(arrays[count], bases[count], 0);
        }
    }

    pinned_slabs_t(pinned_slabs_t const&) = delete;
    pinned_slabs_t& operator=(pinned_slabs_t const&) = delete;

    /** @brief Pins one planned slab and reports the address its first row starts at. */
    void* address_of(slab_plan_t const& plan) {
        void* base = plan.direct;
        if (plan.array) base = environment->GetPrimitiveArrayCritical(plan.array, nullptr);
        arrays[count] = plan.array;
        bases[count] = base;
        ++count;
        if (!base) complete = false;
        return base ? (char*)base + plan.offset : nullptr;
    }

    /** @brief Whether every slab pinned so far became addressable. */
    bool addressable() const noexcept { return complete; }
};

/** @brief Views a path's bytes as the borrowed run the C interface takes, never terminated. */
usearch_string_t borrow(void const* address, jsize length) {
    return usearch_string_t {(char const*)address, (std::size_t)length};
}

} // namespace

#pragma endregion Slab Pinning

#pragma region Options and Telemetry

namespace {

/** @brief The word offsets a flat options layout uses, matching @c OptionsWords on the Java side. */
enum options_word_t {
    options_metric_k = 0,
    options_quantization_k = 1,
    options_dimensions_k = 2,
    options_dimension_padding_k = 3,
    options_connectivity_k = 4,
    options_residency_k = 5,
    options_residency_first_k = 6,
    options_residency_second_k = 7,
    options_residency_third_k = 8,
    options_word_count_k = 9,
};

/** @brief The scalar quantities a handle can be probed for, matching @c Measure on the Java side. */
enum measure_t {
    measure_size_k = 0,
    measure_capacity_k = 1,
    measure_dimensions_k = 2,
    measure_connectivity_k = 3,
    measure_memory_usage_k = 4,
    measure_storage_requirements_k = 5,
};

/** @brief How many traversal counters one call reports. */
constexpr jsize statistics_count_k = 3;

/** @brief Lifts a flat options layout into the C structure, tagging the residency arm it names. */
bool lower_options(JNIEnv* environment, jlongArray words, usearch_dense_options_t& options) {
    if (!words || environment->GetArrayLength(words) < options_word_count_k) {
        raise(environment, usearch_error_shape_mismatch_k, "the options layout is short");
        return false;
    }
    jlong buffer[options_word_count_k];
    environment->GetLongArrayRegion(words, 0, options_word_count_k, buffer);

    options = usearch_dense_options_t {};
    options.metric = (usearch_dense_metric_t)buffer[options_metric_k];
    options.quantization = (usearch_scalar_t)buffer[options_quantization_k];
    options.pad_dimensions = buffer[options_dimension_padding_k] != 0;
    if (!narrowed(environment, buffer[options_dimensions_k], "the dimensionality is negative", options.dimensions))
        return false;
    if (!narrowed(environment, buffer[options_connectivity_k], "the connectivity is negative", options.connectivity))
        return false;

    options.residency = (usearch_residency_t)buffer[options_residency_k];
    switch (options.residency) {
    case usearch_residency_ram_k:
        if (!narrowed(environment, buffer[options_residency_first_k], "the thread count is negative",
                      options.ram.threads))
            return false;
        options.ram.huge_pages = buffer[options_residency_second_k] != 0;
        return true;
    case usearch_residency_vram_k:
        options.vram.device = (std::int32_t)buffer[options_residency_first_k];
        options.vram.unified_addressing = buffer[options_residency_second_k] != 0;
        return true;
    case usearch_residency_nvme_k:
        options.nvme.sector_bytes = (std::uint32_t)buffer[options_residency_first_k];
        options.nvme.queue_depth = (std::uint32_t)buffer[options_residency_second_k];
        options.nvme.direct_io = buffer[options_residency_third_k] == 0;
        return true;
    default: raise(environment, usearch_error_unknown_kind_k, "the residency names neither RAM, VRAM, nor NVMe");
    }
    return false;
}

/** @brief Lowers the C options structure into the flat layout the Java side reads. */
void lift_options(JNIEnv* environment, usearch_dense_options_t const& options, jlongArray words) {
    if (!words || environment->GetArrayLength(words) < options_word_count_k)
        return raise(environment, usearch_error_shape_mismatch_k, "the options layout is short");

    jlong buffer[options_word_count_k] = {};
    buffer[options_metric_k] = (jlong)options.metric;
    buffer[options_quantization_k] = (jlong)options.quantization;
    buffer[options_dimensions_k] = (jlong)options.dimensions;
    buffer[options_dimension_padding_k] = options.pad_dimensions ? 1 : 0;
    buffer[options_connectivity_k] = (jlong)options.connectivity;
    buffer[options_residency_k] = (jlong)options.residency;
    switch (options.residency) {
    case usearch_residency_ram_k:
        buffer[options_residency_first_k] = (jlong)options.ram.threads;
        buffer[options_residency_second_k] = options.ram.huge_pages ? 1 : 0;
        break;
    case usearch_residency_vram_k:
        buffer[options_residency_first_k] = (jlong)options.vram.device;
        buffer[options_residency_second_k] = options.vram.unified_addressing ? 1 : 0;
        break;
    case usearch_residency_nvme_k:
        buffer[options_residency_first_k] = (jlong)options.nvme.sector_bytes;
        buffer[options_residency_second_k] = (jlong)options.nvme.queue_depth;
        buffer[options_residency_third_k] = options.nvme.direct_io ? 0 : 1;
        break;
    default: break;
    }
    environment->SetLongArrayRegion(words, 0, options_word_count_k, buffer);
}

/** @brief Copies the traversal telemetry out, when the caller asked for any. */
void report(JNIEnv* environment, jlongArray sink, usearch_dense_traversal_stats_t const& stats) {
    if (!sink || environment->GetArrayLength(sink) < statistics_count_k) return;
    jlong const buffer[statistics_count_k] = {(jlong)stats.visited_members, (jlong)stats.computed_distances,
                                              (jlong)stats.restarts};
    environment->SetLongArrayRegion(sink, 0, statistics_count_k, buffer);
}

/** @brief Reads the sixteen bytes of a key out of the Java slab it arrived in. */
bool decoded(JNIEnv* environment, jbyteArray source, usearch_key_t& key) {
    if (!source || environment->GetArrayLength(source) != (jsize)sizeof(key)) {
        raise(environment, usearch_error_malformed_input_k, "the key is not sixteen bytes");
        return false;
    }
    environment->GetByteArrayRegion(source, 0, (jsize)sizeof(key), (jbyte*)key.bytes);
    return true;
}

} // namespace

#pragma endregion Options and Telemetry

#pragma region Introspection

extern "C" JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Usearch_nativeVersion(JNIEnv* environment, jclass) {
    return environment->NewStringUTF(usearch_version());
}

extern "C" JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Usearch_nativeCapabilitiesCompiled(JNIEnv* environment,
                                                                                                jclass) {
    return environment->NewStringUTF(usearch_capabilities_compiled());
}

extern "C" JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Usearch_nativeCapabilitiesAvailable(JNIEnv* environment,
                                                                                                 jclass) {
    return environment->NewStringUTF(usearch_capabilities_available());
}

#pragma endregion Introspection

#pragma region Lifetime and Probes

extern "C" JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeInitialize( //
    JNIEnv* environment, jclass, jlongArray words) {
    usearch_dense_options_t options;
    if (!lower_options(environment, words, options)) return 0;
    usearch_error_t error {};
    usearch_dense_index_t const index = usearch_dense_init(&options, &error);
    if (raised(environment, error)) return 0;
    return (jlong)index;
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeFree( //
    JNIEnv*, jclass, jlong handle) {
    usearch_dense_free((usearch_dense_index_t)handle);
}

extern "C" JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeProbe( //
    JNIEnv* environment, jclass, jlong handle, jint measure) {
    usearch_dense_index_t const index = (usearch_dense_index_t)handle;
    usearch_error_t error {};
    std::size_t reading = 0;
    switch (measure) {
    case measure_size_k: reading = usearch_dense_size(index, &error); break;
    case measure_capacity_k: reading = usearch_dense_capacity(index, &error); break;
    case measure_dimensions_k: reading = usearch_dense_dimensions(index, &error); break;
    case measure_connectivity_k: reading = usearch_dense_connectivity(index, &error); break;
    case measure_memory_usage_k: reading = usearch_dense_memory_usage(index, &error); break;
    case measure_storage_requirements_k: reading = usearch_dense_storage_requirements(index, &error); break;
    default: raise(environment, usearch_error_unknown_kind_k, "the probe names no measure"); return 0;
    }
    if (raised(environment, error)) return 0;
    return (jlong)reading;
}

extern "C" JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_DenseIndex_nativeCapabilities( //
    JNIEnv* environment, jclass, jlong handle) {
    usearch_error_t error {};
    char const* const named = usearch_dense_capabilities((usearch_dense_index_t)handle, &error);
    if (raised(environment, error)) return nullptr;
    if (!named) {
        raise(environment, usearch_error_internal_k, "the library reported no capabilities and no error");
        return nullptr;
    }
    return environment->NewStringUTF(named);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeReserve( //
    JNIEnv* environment, jclass, jlong handle, jlong capacity) {
    std::size_t wanted = 0;
    if (!narrowed(environment, capacity, "the capacity is negative", wanted)) return;
    usearch_error_t error {};
    usearch_dense_reserve((usearch_dense_index_t)handle, wanted, &error);
    raised(environment, error);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeClear( //
    JNIEnv* environment, jclass, jlong handle) {
    usearch_error_t error {};
    usearch_dense_clear((usearch_dense_index_t)handle, &error);
    raised(environment, error);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeMerge( //
    JNIEnv* environment, jclass, jlong handle, jlong donor) {
    usearch_error_t error {};
    usearch_dense_merge((usearch_dense_index_t)handle, (usearch_dense_index_t)donor, &error);
    raised(environment, error);
}

#pragma endregion Lifetime and Probes

#pragma region Insertion and Search

extern "C" JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeAdd( //
    JNIEnv* environment, jclass, jlong handle, jbyteArray keys, jobject vectors, jlong byte_offset, jlong rows,
    jlong stride, jint scalar_kind, jlong expansion_factor, jlongArray statistics) {

    std::size_t count = 0, row_stride = 0, expansion = 0, key_bytes = 0;
    if (!narrowed(environment, rows, "the row count is negative", count)) return 0;
    if (!narrowed(environment, stride, "the stride is negative", row_stride)) return 0;
    if (!narrowed(environment, expansion_factor, "the expansion factor is negative", expansion)) return 0;
    if (!multiplied(environment, count, sizeof(usearch_key_t), "the key matrix overflows", key_bytes)) return 0;
    if (!keys || (std::size_t)environment->GetArrayLength(keys) != key_bytes) {
        raise(environment, usearch_error_shape_mismatch_k, "the key matrix is not one key per row");
        return 0;
    }

    slab_plan_t const rows_plan(environment, vectors, byte_offset);
    slab_plan_t const keys_plan(environment, keys, 0);
    if (!rows_plan.addressable() || !keys_plan.addressable()) {
        raise(environment, usearch_error_malformed_input_k, "the matrix borrows no addressable storage");
        return 0;
    }

    usearch_dense_expansion_config_t const config {expansion};
    usearch_dense_traversal_stats_t stats {};
    usearch_error_t error {};
    std::size_t added = 0;
    {
        pinned_slabs_t pinned(environment);
        void* const rows_address = pinned.address_of(rows_plan);
        void* const keys_address = pinned.address_of(keys_plan);
        if (!pinned.addressable()) {
            raise(environment, usearch_error_out_of_memory_k, "the slabs could not be pinned");
            return 0;
        }
        added = usearch_dense_add((usearch_dense_index_t)handle, (usearch_key_t const*)keys_address, rows_address,
                                  count, row_stride, (usearch_scalar_t)scalar_kind, &config, &stats, &error);
    }
    if (raised(environment, error)) return 0;
    report(environment, statistics, stats);
    return (jlong)added;
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeSearch( //
    JNIEnv* environment, jclass, jlong handle, jobject queries, jlong byte_offset, jlong rows, jlong stride,
    jint scalar_kind, jlong wanted, jlong expansion_factor, jlongArray statistics, jbyteArray keys,
    jdoubleArray distances, jlongArray counts) {

    std::size_t count = 0, row_stride = 0, matches = 0, expansion = 0;
    if (!narrowed(environment, rows, "the query count is negative", count)) return;
    if (!narrowed(environment, stride, "the stride is negative", row_stride)) return;
    if (!narrowed(environment, wanted, "the wanted count is negative", matches)) return;
    if (!narrowed(environment, expansion_factor, "the expansion factor is negative", expansion)) return;
    if (!matches)
        return raise(environment, usearch_error_shape_mismatch_k, "the outputs reserve no room for a single match");

    // The Java side sized its slabs against the same products; re-deriving them here is what keeps
    // an extent that wrapped on the way in from becoming a write past the end of a short slab.
    std::size_t slots = 0, key_bytes = 0, keys_stride = 0, distances_stride = 0;
    if (!multiplied(environment, count, matches, "the match matrix overflows", slots)) return;
    if (!multiplied(environment, slots, sizeof(usearch_key_t), "the key matrix overflows", key_bytes)) return;
    if (!multiplied(environment, matches, sizeof(usearch_key_t), "the key row overflows", keys_stride)) return;
    if (!multiplied(environment, matches, sizeof(usearch_distance_t), "the distance row overflows", distances_stride))
        return;
    if (!keys || !distances || !counts || (std::size_t)environment->GetArrayLength(keys) < key_bytes ||
        (std::size_t)environment->GetArrayLength(distances) < slots ||
        (std::size_t)environment->GetArrayLength(counts) < count)
        return raise(environment, usearch_error_shape_mismatch_k, "the outputs are narrower than the batch");

    // Counts cross as `size_t` and Java holds them as `long`, so they travel through a staging
    // vector rather than being aliased across the two spellings.
    std::vector<std::size_t> reported;
    std::vector<jlong> widened;
    try {
        reported.resize(count);
        widened.resize(count);
    }
    catch (...) {
        return raise(environment, usearch_error_out_of_memory_k, "the per-query counts could not be staged");
    }

    slab_plan_t const queries_plan(environment, queries, byte_offset);
    slab_plan_t const keys_plan(environment, keys, 0);
    slab_plan_t const distances_plan(environment, distances, 0);
    if (!queries_plan.addressable())
        return raise(environment, usearch_error_malformed_input_k, "the matrix borrows no addressable storage");

    usearch_dense_expansion_config_t const config {expansion};
    usearch_dense_traversal_stats_t stats {};
    usearch_error_t error {};
    {
        pinned_slabs_t pinned(environment);
        void* const queries_address = pinned.address_of(queries_plan);
        void* const keys_address = pinned.address_of(keys_plan);
        void* const distances_address = pinned.address_of(distances_plan);
        if (!pinned.addressable())
            return raise(environment, usearch_error_out_of_memory_k, "the slabs could not be pinned");
        usearch_dense_search((usearch_dense_index_t)handle, queries_address, count, row_stride,
                             (usearch_scalar_t)scalar_kind, matches, &config, &stats, (usearch_key_t*)keys_address,
                             keys_stride, (usearch_distance_t*)distances_address, distances_stride, reported.data(),
                             &error);
    }
    if (raised(environment, error)) return;

    for (std::size_t query = 0; query != count; ++query) widened[query] = (jlong)reported[query];
    environment->SetLongArrayRegion(counts, 0, (jsize)count, widened.data());
    report(environment, statistics, stats);
}

extern "C" JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeCount( //
    JNIEnv* environment, jclass, jlong handle, jbyteArray key) {
    usearch_key_t identity {};
    if (!decoded(environment, key, identity)) return 0;
    usearch_error_t error {};
    std::size_t const counted = usearch_dense_count((usearch_dense_index_t)handle, identity, &error);
    if (raised(environment, error)) return 0;
    return (jlong)counted;
}

extern "C" JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeGet( //
    JNIEnv* environment, jclass, jlong handle, jbyteArray key, jobject vectors, jlong byte_offset, jlong wanted,
    jlong stride, jint scalar_kind) {

    usearch_key_t identity {};
    if (!decoded(environment, key, identity)) return 0;

    std::size_t rows = 0, row_stride = 0;
    if (!narrowed(environment, wanted, "the row capacity is negative", rows)) return 0;
    if (!narrowed(environment, stride, "the stride is negative", row_stride)) return 0;

    slab_plan_t const rows_plan(environment, vectors, byte_offset);
    if (!rows_plan.addressable()) {
        raise(environment, usearch_error_malformed_input_k, "the matrix borrows no addressable storage");
        return 0;
    }

    usearch_error_t error {};
    std::size_t entries = 0;
    {
        pinned_slabs_t pinned(environment);
        void* const rows_address = pinned.address_of(rows_plan);
        if (!pinned.addressable()) {
            raise(environment, usearch_error_out_of_memory_k, "the slab could not be pinned");
            return 0;
        }
        entries = usearch_dense_get((usearch_dense_index_t)handle, identity, rows_address, rows, row_stride,
                                    (usearch_scalar_t)scalar_kind, &error);
    }
    if (raised(environment, error)) return 0;
    return (jlong)entries;
}

#pragma endregion Insertion and Search

#pragma region Serialization

namespace {

/** @brief The shape @c save and @c load share: pin a path, hand it across, lift the outcome. */
enum class checkpoint_verb_t { save_k, load_k };

void traverse_checkpoint(JNIEnv* environment, jlong handle, jbyteArray path, checkpoint_verb_t verb) {
    slab_plan_t const path_plan(environment, path, 0);
    if (!path_plan.addressable())
        return raise(environment, usearch_error_malformed_input_k, "the path borrows no storage");
    jsize const length = environment->GetArrayLength(path);

    usearch_error_t error {};
    {
        pinned_slabs_t pinned(environment);
        void* const address = pinned.address_of(path_plan);
        if (!pinned.addressable())
            return raise(environment, usearch_error_out_of_memory_k, "the path could not be pinned");
        if (verb == checkpoint_verb_t::save_k)
            usearch_dense_save((usearch_dense_index_t)handle, borrow(address, length), &error);
        else usearch_dense_load((usearch_dense_index_t)handle, borrow(address, length), &error);
    }
    raised(environment, error);
}

} // namespace

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeSave( //
    JNIEnv* environment, jclass, jlong handle, jbyteArray path) {
    traverse_checkpoint(environment, handle, path, checkpoint_verb_t::save_k);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeLoad( //
    JNIEnv* environment, jclass, jlong handle, jbyteArray path) {
    traverse_checkpoint(environment, handle, path, checkpoint_verb_t::load_k);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeSaveBuffer( //
    JNIEnv* environment, jclass, jlong handle, jobject buffer, jlong byte_offset, jlong length) {
    std::size_t bytes = 0;
    if (!narrowed(environment, length, "the buffer length is negative", bytes)) return;
    slab_plan_t const buffer_plan(environment, buffer, byte_offset);
    if (!buffer_plan.addressable())
        return raise(environment, usearch_error_malformed_input_k, "the buffer borrows no addressable storage");

    usearch_error_t error {};
    {
        pinned_slabs_t pinned(environment);
        void* const address = pinned.address_of(buffer_plan);
        if (!pinned.addressable())
            return raise(environment, usearch_error_out_of_memory_k, "the buffer could not be pinned");
        usearch_dense_save_buffer((usearch_dense_index_t)handle, address, bytes, &error);
    }
    raised(environment, error);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeLoadBuffer( //
    JNIEnv* environment, jclass, jlong handle, jobject buffer, jlong byte_offset, jlong length) {
    std::size_t bytes = 0;
    if (!narrowed(environment, length, "the buffer length is negative", bytes)) return;
    slab_plan_t const buffer_plan(environment, buffer, byte_offset);
    if (!buffer_plan.addressable())
        return raise(environment, usearch_error_malformed_input_k, "the buffer borrows no addressable storage");

    usearch_error_t error {};
    {
        pinned_slabs_t pinned(environment);
        void* const address = pinned.address_of(buffer_plan);
        if (!pinned.addressable())
            return raise(environment, usearch_error_out_of_memory_k, "the buffer could not be pinned");
        usearch_dense_load_buffer((usearch_dense_index_t)handle, address, bytes, &error);
    }
    raised(environment, error);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeMetadata( //
    JNIEnv* environment, jclass, jbyteArray path, jlongArray words) {
    slab_plan_t const path_plan(environment, path, 0);
    if (!path_plan.addressable())
        return raise(environment, usearch_error_malformed_input_k, "the path borrows no storage");
    jsize const length = environment->GetArrayLength(path);

    usearch_dense_options_t options {};
    usearch_error_t error {};
    {
        pinned_slabs_t pinned(environment);
        void* const address = pinned.address_of(path_plan);
        if (!pinned.addressable())
            return raise(environment, usearch_error_out_of_memory_k, "the path could not be pinned");
        usearch_dense_metadata(borrow(address, length), &options, &error);
    }
    if (raised(environment, error)) return;
    lift_options(environment, options, words);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeMetadataBuffer( //
    JNIEnv* environment, jclass, jobject buffer, jlong byte_offset, jlong length, jlongArray words) {
    std::size_t bytes = 0;
    if (!narrowed(environment, length, "the buffer length is negative", bytes)) return;
    slab_plan_t const buffer_plan(environment, buffer, byte_offset);
    if (!buffer_plan.addressable())
        return raise(environment, usearch_error_malformed_input_k, "the buffer borrows no addressable storage");

    usearch_dense_options_t options {};
    usearch_error_t error {};
    {
        pinned_slabs_t pinned(environment);
        void* const address = pinned.address_of(buffer_plan);
        if (!pinned.addressable())
            return raise(environment, usearch_error_out_of_memory_k, "the buffer could not be pinned");
        usearch_dense_metadata_buffer(address, bytes, &options, &error);
    }
    if (raised(environment, error)) return;
    lift_options(environment, options, words);
}

extern "C" JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeReloadOptions( //
    JNIEnv* environment, jclass, jlongArray input, jlongArray output) {
    usearch_dense_options_t options;
    if (!lower_options(environment, input, options)) return;
    lift_options(environment, options, output);
}

#pragma endregion Serialization
