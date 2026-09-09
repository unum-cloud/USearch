/**
 *  @brief JNI boundary of the Java SDK, translating between the JVM and the flat C interface
 *      of @c include/usearch/types.h and @c include/usearch/dense_index.h.
 *
 *  @author Ash Vardanian
 *  @file java/lib.c
 *
 *  This translation unit instantiates no index template. It links the pre-instantiated library
 *  and does nothing but marshal: borrow a slab, lower an enumerator to its code, hand the call
 *  across, lift the outcome back. Every verb the Java side declares as @c native appears here
 *  exactly once, and the two agree by symbol name rather than by a generated header.
 *
 *  Java's signed 32-bit @c int wraps four billion times sooner than a @c size_t, so every extent
 *  a caller supplies is checked on the Java side before it reaches an allocation, and checked
 *  again here for sign and for fit before it narrows into a @c size_t.
 *
 *  Slabs are borrowed through the typed element accessors rather than through a critical region,
 *  since an engine call runs for as long as its batch does and a critical region stalls every
 *  collector thread for that whole span. A direct buffer is addressed in place either way, so a
 *  caller who cannot afford the copy already has the argument that avoids it. The Java side names
 *  which accessor each slab needs, which is what lets one @c jobject parameter carry five slab
 *  shapes without this file asking the JVM what it was handed.
 *
 *  Progress handlers and admission predicates stay unbound. Both are called from the engine's own
 *  worker threads, which are not attached to this JVM, so every report or candidate would pay an
 *  attach and an upcall, and one that threw would unwind through native frames that cannot handle
 *  it.
 */
#include <stdbool.h> // `bool`
#include <stddef.h>  // `size_t`
#include <stdint.h>  // `int32_t`
#include <stdlib.h>  // `calloc`, `free`

#include <jni.h>

#include <usearch/dense_index.h>
#include <usearch/types.h>

/**
 *  @brief Declares the JNI level this boundary calls at, and nothing more: it caches no class, no
 *      method, and no state of any kind between calls, so there is nothing to tear down either.
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* machine, void* reserved) {
    JNIEnv* environment = NULL;
    (void)reserved;
    if ((*machine)->GetEnv(machine, (void**)&environment, JNI_VERSION_21) != JNI_OK) return JNI_ERR;
    return JNI_VERSION_21;
}

#pragma region Failure Reporting

/**
 *  @brief Raises the Java failure a status code and its message describe.
 *
 *  The class and its factory are resolved per raise rather than cached in file scope: this is the
 *  error path, where one class lookup costs nothing, and the alternative is process-lifetime
 *  mutable state that every call would have to trust was initialized.
 */
static void raise(JNIEnv* environment, usearch_status_t status, char const* message) {
    if ((*environment)->ExceptionCheck(environment)) return;

    jclass const failure_class = (*environment)->FindClass(environment, "cloud/unum/usearch/Types$USearchException");
    if (!failure_class) return;
    jmethodID const factory = (*environment)
                                  ->GetStaticMethodID( //
                                      environment, failure_class, "of",
                                      "(ILjava/lang/String;)Lcloud/unum/usearch/Types$USearchException;");
    if (!factory) return;

    jstring const text = (*environment)->NewStringUTF(environment, message ? message : "");
    jobject const failure = (*environment)
                                ->CallStaticObjectMethod( //
                                    environment, failure_class, factory, (jint)status, text);
    if (failure) (*environment)->Throw(environment, (jthrowable)failure);
}

/** @brief Raises the failure a populated error slot carries; reports whether it raised. */
static bool raised(JNIEnv* environment, usearch_error_t const* error) {
    if (error->status == usearch_success_k) return false;
    raise(environment, error->status, error->message);
    return true;
}

/** @brief Narrows a Java extent into a @c size_t, raising rather than wrapping on a negative. */
static bool narrowed(JNIEnv* environment, jlong extent, char const* what, size_t* into) {
    if (extent < 0) {
        raise(environment, usearch_error_shape_mismatch_k, what);
        return false;
    }
    *into = (size_t)extent;
    return true;
}

/**
 *  @brief Multiplies two extents, raising @c usearch_error_overflow_k rather than wrapping.
 *      The Java side sized its slabs against the same product, so a wrap here would let the
 *      engine write past a buffer that looks long enough.
 */
static bool multiplied(JNIEnv* environment, size_t left, size_t right, char const* what, size_t* into) {
    if (left && right > (size_t)-1 / left) {
        raise(environment, usearch_error_overflow_k, what);
        return false;
    }
    *into = left * right;
    return true;
}

#pragma endregion Failure Reporting

#pragma region Slab Borrowing

/** @brief How a caller's slab is addressed, matching @c Matrix.Storage on the Java side. */
enum storage_t {
    storage_doubles_k = 0,
    storage_floats_k = 1,
    storage_shorts_k = 2,
    storage_bytes_k = 3,
    storage_direct_k = 4,
};

/** @brief Whether a released slab's contents travel back into the JVM's copy of it. */
enum writeback_t {
    /** @brief Discard whatever the call wrote; the slab was a pure input. */
    writeback_discard_k = JNI_ABORT,
    /** @brief Publish whatever the call wrote; the slab was an output. */
    writeback_commit_k = 0,
};

/**
 *  @brief Addresses a caller's slab through the accessor its storage kind names.
 *
 *  A @c NULL answer is the one failure a caller checks, and it covers every way a slab can fail to
 *  become addressable: a missing argument, a heap buffer with no stable address, and a JVM that
 *  could not produce the elements all report the same way.
 */
static void* elements_borrowed(JNIEnv* environment, jobject slab, enum storage_t storage) {
    if (!slab) return NULL;
    switch (storage) {
    case storage_doubles_k: return (*environment)->GetDoubleArrayElements(environment, (jdoubleArray)slab, NULL);
    case storage_floats_k: return (*environment)->GetFloatArrayElements(environment, (jfloatArray)slab, NULL);
    case storage_shorts_k: return (*environment)->GetShortArrayElements(environment, (jshortArray)slab, NULL);
    case storage_bytes_k: return (*environment)->GetByteArrayElements(environment, (jbyteArray)slab, NULL);
    case storage_direct_k: return (*environment)->GetDirectBufferAddress(environment, slab);
    default: return NULL;
    }
}

/**
 *  @brief Returns a slab's elements, publishing or discarding what the call wrote.
 *
 *  A direct buffer was addressed in place and has nothing to return, and a slab that never became
 *  addressable has nothing either, so a failed borrow releases as harmlessly as a finished one.
 */
static void elements_released(JNIEnv* environment, jobject slab, enum storage_t storage, void* base,
                              enum writeback_t writeback) {
    jint const mode = (jint)writeback;
    if (!base) return;
    switch (storage) {
    case storage_doubles_k:
        (*environment)->ReleaseDoubleArrayElements(environment, (jdoubleArray)slab, (jdouble*)base, mode);
        break;
    case storage_floats_k:
        (*environment)->ReleaseFloatArrayElements(environment, (jfloatArray)slab, (jfloat*)base, mode);
        break;
    case storage_shorts_k:
        (*environment)->ReleaseShortArrayElements(environment, (jshortArray)slab, (jshort*)base, mode);
        break;
    case storage_bytes_k:
        (*environment)->ReleaseByteArrayElements(environment, (jbyteArray)slab, (jbyte*)base, mode);
        break;
    default: break;
    }
}

/** @brief Views a path's bytes as the borrowed run the C interface takes, never terminated. */
static usearch_bytes_view_t borrowed_bytes(void const* address, size_t length) {
    usearch_bytes_view_t view = {(uint8_t const*)address, length};
    return view;
}

/** @brief Views a caller's slab as the writable run every buffer-filling verb takes. */
static usearch_bytes_span_t spanned_bytes(void* address, size_t length) {
    usearch_bytes_span_t span = {(uint8_t*)address, length};
    return span;
}

/** @brief Views a caller's rows as the borrowed strided matrix every batch input takes. */
static usearch_matrix_view_t viewed_matrix(void const* address, size_t rows, size_t columns, size_t stride_bytes,
                                           jint scalar_kind) {
    usearch_matrix_view_t view = {address, rows, columns, stride_bytes, (usearch_scalar_t)scalar_kind};
    return view;
}

/** @brief Views a caller's rows as the writable strided matrix every batch output takes. */
static usearch_matrix_span_t spanned_matrix(void* address, size_t capacity, size_t columns, size_t stride_bytes,
                                            jint scalar_kind) {
    usearch_matrix_span_t span = {address, capacity, columns, stride_bytes, (usearch_scalar_t)scalar_kind};
    return span;
}

#pragma endregion Slab Borrowing

#pragma region Options and Telemetry

/** @brief The word offsets a flat options layout uses, matching @c OptionsWords on the Java side. */
enum options_word_t {
    options_metric_k = 0,
    options_quantization_k = 1,
    options_dimensions_k = 2,
    options_connectivity_k = 3,
    options_values_per_key_k = 4,
    options_residency_k = 5,
    options_word_count_k = 6,
};

/** @brief The word offsets a flat metadata layout uses, matching @c DenseMetadata on the Java side. */
enum metadata_word_t {
    metadata_metric_k = 0,
    metadata_quantization_k = 1,
    metadata_dimensions_k = 2,
    metadata_connectivity_k = 3,
    metadata_word_count_k = 4,
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
enum { statistics_count_k = 3 };

/**
 *  @brief Lifts a flat options layout into the C structure, tagging the residency arm it names.
 *
 *  The arm itself stays zeroed: naming memory domains, devices, or device paths is placement, and
 *  a zeroed arm derives it, which is the only shape a flat word layout can carry faithfully.
 */
static bool lower_options(JNIEnv* environment, jlongArray words, usearch_dense_options_t* options) {
    if (!words || (*environment)->GetArrayLength(environment, words) < options_word_count_k) {
        raise(environment, usearch_error_shape_mismatch_k, "the options layout is short");
        return false;
    }
    jlong buffer[options_word_count_k];
    (*environment)->GetLongArrayRegion(environment, words, 0, options_word_count_k, buffer);

    usearch_dense_options_t const zeroed = {0};
    *options = zeroed;
    options->metric = (usearch_dense_metric_t)buffer[options_metric_k];
    options->quantization = (usearch_scalar_t)buffer[options_quantization_k];
    if (!narrowed(environment, buffer[options_dimensions_k], "the dimensionality is negative", &options->dimensions))
        return false;
    if (!narrowed(environment, buffer[options_connectivity_k], "the connectivity is negative", &options->connectivity))
        return false;
    if (!narrowed(environment, buffer[options_values_per_key_k], "the entries per key is negative",
                  &options->values_per_key))
        return false;

    options->residency = (usearch_residency_t)buffer[options_residency_k];
    switch (options->residency) {
    case usearch_residency_ram_k:
    case usearch_residency_vram_k:
    case usearch_residency_nvme_k: return true;
    default: raise(environment, usearch_error_unknown_kind_k, "the residency names neither RAM, VRAM, nor NVMe");
    }
    return false;
}

/** @brief Lowers the C options structure into the flat layout the Java side reads. */
static void lift_options(JNIEnv* environment, usearch_dense_options_t const* options, jlongArray words) {
    if (!words || (*environment)->GetArrayLength(environment, words) < options_word_count_k) {
        raise(environment, usearch_error_shape_mismatch_k, "the options layout is short");
        return;
    }

    jlong buffer[options_word_count_k] = {0};
    buffer[options_metric_k] = (jlong)options->metric;
    buffer[options_quantization_k] = (jlong)options->quantization;
    buffer[options_dimensions_k] = (jlong)options->dimensions;
    buffer[options_connectivity_k] = (jlong)options->connectivity;
    buffer[options_values_per_key_k] = (jlong)options->values_per_key;
    buffer[options_residency_k] = (jlong)options->residency;
    (*environment)->SetLongArrayRegion(environment, words, 0, options_word_count_k, buffer);
}

/**
 *  @brief Lowers a checkpoint's geometry and metric into the flat layout the Java side reads.
 *
 *  Neither the residency nor the entries-per-key bound appears: a checkpoint records where it was
 *  written rather than where a reader must keep it, and that bound governs a live handle's mutable
 *  state rather than the serialized form.
 */
static void lift_metadata(JNIEnv* environment, usearch_dense_metadata_t const* metadata, jlongArray words) {
    if (!words || (*environment)->GetArrayLength(environment, words) < metadata_word_count_k) {
        raise(environment, usearch_error_shape_mismatch_k, "the metadata layout is short");
        return;
    }

    jlong buffer[metadata_word_count_k] = {0};
    buffer[metadata_metric_k] = (jlong)metadata->metric;
    buffer[metadata_quantization_k] = (jlong)metadata->quantization;
    buffer[metadata_dimensions_k] = (jlong)metadata->dimensions;
    buffer[metadata_connectivity_k] = (jlong)metadata->connectivity;
    (*environment)->SetLongArrayRegion(environment, words, 0, metadata_word_count_k, buffer);
}

/** @brief Copies the traversal telemetry out, when the caller asked for any. */
static void report(JNIEnv* environment, jlongArray sink, usearch_dense_traversal_stats_t const* stats) {
    if (!sink || (*environment)->GetArrayLength(environment, sink) < statistics_count_k) return;
    jlong const buffer[statistics_count_k] = {(jlong)stats->visited_members, (jlong)stats->computed_distances,
                                              (jlong)stats->restarts};
    (*environment)->SetLongArrayRegion(environment, sink, 0, statistics_count_k, buffer);
}

/** @brief Reads the sixteen bytes of a key out of the Java slab it arrived in. */
static bool decoded(JNIEnv* environment, jbyteArray source, usearch_key_t* key) {
    if (!source || (*environment)->GetArrayLength(environment, source) != (jsize)sizeof(*key)) {
        raise(environment, usearch_error_malformed_input_k, "the key is not sixteen bytes");
        return false;
    }
    (*environment)->GetByteArrayRegion(environment, source, 0, (jsize)sizeof(*key), (jbyte*)key->bytes);
    return true;
}

#pragma endregion Options and Telemetry

#pragma region Introspection

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Types_nativeVersion(JNIEnv* environment, jclass owner) {
    (void)owner;
    return (*environment)->NewStringUTF(environment, usearch_version());
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Types_nativeCapabilitiesCompiled(JNIEnv* environment, jclass owner) {
    (void)owner;
    return (*environment)->NewStringUTF(environment, usearch_capabilities_compiled());
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Types_nativeCapabilitiesAvailable(JNIEnv* environment, jclass owner) {
    (void)owner;
    return (*environment)->NewStringUTF(environment, usearch_capabilities_available());
}

#pragma endregion Introspection

#pragma region Executor

/**
 *  @brief Spawns host workers for RAM-resident indexes on the memory domains a caller named.
 *
 *  A @c jint is exactly an @c int32_t, so the node ids are borrowed and handed across unconverted;
 *  a null or empty array names nothing and leaves the placement to the engine.
 */
JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Types_nativeExecutorForRam( //
    JNIEnv* environment, jclass owner, jintArray memory_domains, jlong worker_count) {
    (void)owner;
    size_t threads = 0;
    if (!narrowed(environment, worker_count, "the thread count is negative", &threads)) return 0;

    jsize const named = memory_domains ? (*environment)->GetArrayLength(environment, memory_domains) : 0;
    jint* const borrowed = named ? (*environment)->GetIntArrayElements(environment, memory_domains, NULL) : NULL;
    if (named && !borrowed) {
        raise(environment, usearch_error_malformed_input_k, "the memory domains borrow no addressable storage");
        return 0;
    }

    usearch_ram_options_t const ram = {(int32_t const*)borrowed, (size_t)named};
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_t const handle = usearch_executor_for_ram(&ram, threads, &error);
    if (borrowed) (*environment)->ReleaseIntArrayElements(environment, memory_domains, borrowed, JNI_ABORT);

    if (raised(environment, &error)) return 0;
    return (jlong)(size_t)handle;
}

/**
 *  @brief Opens one submission stream on every device a caller named, driven by the calling thread.
 *      It takes no thread count: the width of the pool is the device list.
 */
JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Types_nativeExecutorForVram( //
    JNIEnv* environment, jclass owner, jintArray devices) {
    (void)owner;
    jsize const named = devices ? (*environment)->GetArrayLength(environment, devices) : 0;
    jint* const borrowed = named ? (*environment)->GetIntArrayElements(environment, devices, NULL) : NULL;
    if (named && !borrowed) {
        raise(environment, usearch_error_malformed_input_k, "the devices borrow no addressable storage");
        return 0;
    }

    usearch_vram_options_t const vram = {(int32_t const*)borrowed, (size_t)named};
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_t const handle = usearch_executor_for_vram(&vram, &error);
    if (borrowed) (*environment)->ReleaseIntArrayElements(environment, devices, borrowed, JNI_ABORT);

    if (raised(environment, &error)) return 0;
    return (jlong)(size_t)handle;
}

/**
 *  @brief Spawns host workers for NVMe-resident indexes, one submission ring apiece, on the devices
 *      the paths name.
 *
 *  Every path is copied into one native run and viewed from there, so no slab stays borrowed across
 *  the call and the views the C interface reads sit in one contiguous array.
 */
JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Types_nativeExecutorForNvme( //
    JNIEnv* environment, jclass owner, jobjectArray paths, jlong worker_count) {
    (void)owner;
    size_t threads = 0;
    if (!narrowed(environment, worker_count, "the thread count is negative", &threads)) return 0;

    jsize const named = paths ? (*environment)->GetArrayLength(environment, paths) : 0;
    usearch_bytes_view_t* const views = named ? (usearch_bytes_view_t*)calloc((size_t)named, sizeof(*views)) : NULL;
    uint8_t* staged = NULL;
    usearch_executor_t handle = NULL;
    if (named && !views) {
        raise(environment, usearch_error_out_of_memory_k, "the device paths could not be staged");
        return 0;
    }

    size_t total = 0;
    for (jsize order = 0; order != named; ++order) {
        jbyteArray const path = (jbyteArray)(*environment)->GetObjectArrayElement(environment, paths, order);
        if (!path) {
            raise(environment, usearch_error_malformed_input_k, "a device path is empty");
            goto release;
        }
        views[order].length = (size_t)(*environment)->GetArrayLength(environment, path);
        total += views[order].length;
        (*environment)->DeleteLocalRef(environment, path);
    }

    staged = total ? (uint8_t*)malloc(total) : NULL;
    if (total && !staged) {
        raise(environment, usearch_error_out_of_memory_k, "the device paths could not be staged");
        goto release;
    }
    for (jsize order = 0, filled = 0; order != named; ++order) {
        jbyteArray const path = (jbyteArray)(*environment)->GetObjectArrayElement(environment, paths, order);
        (*environment)->GetByteArrayRegion(environment, path, 0, (jsize)views[order].length, (jbyte*)staged + filled);
        views[order].data = staged + filled;
        filled += (jsize)views[order].length;
        (*environment)->DeleteLocalRef(environment, path);
    }

    usearch_nvme_options_t const nvme = {views, (size_t)named};
    usearch_error_t error = {usearch_success_k, NULL};
    handle = usearch_executor_for_nvme(&nvme, threads, &error);
    if (raised(environment, &error)) handle = NULL;

release:
    free(staged);
    free(views);
    return (jlong)(size_t)handle;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Types_nativeExecutorDrain( //
    JNIEnv* environment, jclass owner, jlong handle) {
    (void)owner;
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_drain((usearch_executor_t)(size_t)handle, &error);
    raised(environment, &error);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Types_nativeExecutorFree( //
    JNIEnv* environment, jclass owner, jlong handle) {
    (void)environment;
    (void)owner;
    usearch_executor_free((usearch_executor_t)(size_t)handle);
}

#pragma endregion Executor

#pragma region Lifetime and Probes

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeInitialize( //
    JNIEnv* environment, jclass owner, jlongArray words) {
    (void)owner;
    usearch_dense_options_t options;
    if (!lower_options(environment, words, &options)) return 0;
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_index_t const index = usearch_dense_init(&options, &error);
    if (raised(environment, &error)) return 0;
    return (jlong)(size_t)index;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeFree( //
    JNIEnv* environment, jclass owner, jlong handle) {
    (void)environment;
    (void)owner;
    usearch_dense_free((usearch_dense_index_t)(size_t)handle);
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeProbe( //
    JNIEnv* environment, jclass owner, jlong handle, jint measure) {
    (void)owner;
    usearch_dense_index_t const index = (usearch_dense_index_t)(size_t)handle;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t reading = 0;
    switch (measure) {
    case measure_size_k: reading = usearch_dense_size(index, &error); break;
    case measure_capacity_k: reading = usearch_dense_capacity(index, &error); break;
    case measure_dimensions_k: reading = usearch_dense_dimensions(index, &error); break;
    case measure_connectivity_k: reading = usearch_dense_connectivity(index, &error); break;
    case measure_memory_usage_k: reading = usearch_dense_memory_usage(index, &error); break;
    case measure_storage_requirements_k: reading = usearch_dense_storage_requirements(index, &error); break;
    default: raise(environment, usearch_error_unknown_kind_k, "the probe names no measure"); return 0;
    }
    if (raised(environment, &error)) return 0;
    return (jlong)reading;
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_DenseIndex_nativeCapabilities( //
    JNIEnv* environment, jclass owner, jlong handle) {
    (void)owner;
    usearch_error_t error = {usearch_success_k, NULL};
    char const* const named = usearch_dense_capabilities((usearch_dense_index_t)(size_t)handle, &error);
    if (raised(environment, &error)) return NULL;
    if (!named) {
        raise(environment, usearch_error_internal_k, "the library reported no capabilities and no error");
        return NULL;
    }
    return (*environment)->NewStringUTF(environment, named);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeReserve( //
    JNIEnv* environment, jclass owner, jlong handle, jlong capacity) {
    (void)owner;
    size_t wanted = 0;
    if (!narrowed(environment, capacity, "the capacity is negative", &wanted)) return;
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_reserve((usearch_dense_index_t)(size_t)handle, wanted, &error);
    raised(environment, &error);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeClear( //
    JNIEnv* environment, jclass owner, jlong handle) {
    (void)owner;
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_clear((usearch_dense_index_t)(size_t)handle, &error);
    raised(environment, &error);
}

#pragma endregion Lifetime and Probes

#pragma region Insertion and Refinement

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeAdd( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jbyteArray keys, jobject vectors, jint storage,
    jlong byte_offset, jlong rows, jlong columns, jlong stride_bytes, jint scalar_kind, jlong expansion_factor,
    jlongArray statistics) {
    (void)owner;

    size_t count = 0, width = 0, row_stride = 0, expansion = 0, key_bytes = 0;
    if (!narrowed(environment, rows, "the row count is negative", &count)) return 0;
    if (!narrowed(environment, columns, "the column count is negative", &width)) return 0;
    if (!narrowed(environment, stride_bytes, "the stride is negative", &row_stride)) return 0;
    if (!narrowed(environment, expansion_factor, "the expansion factor is negative", &expansion)) return 0;
    if (!multiplied(environment, count, sizeof(usearch_key_t), "the key matrix overflows", &key_bytes)) return 0;
    if (!keys || (size_t)(*environment)->GetArrayLength(environment, keys) != key_bytes) {
        raise(environment, usearch_error_shape_mismatch_k, "the key matrix is not one key per row");
        return 0;
    }

    usearch_dense_insertion_options_t const options = {expansion};
    usearch_dense_traversal_stats_t stats = {0, 0, 0};
    usearch_error_t error = {usearch_success_k, NULL};
    size_t added = 0;

    void* const rows_base = elements_borrowed(environment, vectors, (enum storage_t)storage);
    void* const keys_base = elements_borrowed(environment, keys, storage_bytes_k);
    if (!rows_base || !keys_base) {
        raise(environment, usearch_error_malformed_input_k, "the matrix borrows no addressable storage");
        goto release;
    }

    usearch_keys_view_t const identities = {(usearch_key_t const*)keys_base, count};
    usearch_matrix_view_t const matrix = viewed_matrix((char*)rows_base + byte_offset, count, width, row_stride,
                                                       scalar_kind);
    added = usearch_dense_add((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor, identities,
                              &matrix, &options, NULL, NULL, &stats, &error);
    if (raised(environment, &error)) added = 0;
    else report(environment, statistics, &stats);

release:
    elements_released(environment, keys, storage_bytes_k, keys_base, writeback_discard_k);
    elements_released(environment, vectors, (enum storage_t)storage, rows_base, writeback_discard_k);
    return (jlong)added;
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeRefine( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jlong expansion_factor, jlongArray statistics) {
    (void)owner;

    size_t expansion = 0;
    if (!narrowed(environment, expansion_factor, "the expansion factor is negative", &expansion)) return 0;

    usearch_dense_insertion_options_t const options = {expansion};
    usearch_dense_traversal_stats_t stats = {0, 0, 0};
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const changed = usearch_dense_refine((usearch_dense_index_t)(size_t)handle,
                                                (usearch_executor_t)(size_t)executor, &options, NULL, NULL, &stats,
                                                &error);
    if (raised(environment, &error)) return 0;
    report(environment, statistics, &stats);
    return (jlong)changed;
}

#pragma endregion Insertion and Refinement

#pragma region Ranked Walks

/**
 *  @brief The shape @c search and @c maxsim share: borrow the queries, hand the ranked run across,
 *      write the keys, scores and per-query counts back into the JVM's arrays.
 *
 *  A @p rows_per_query above zero selects the late-interaction verb, which is the only difference
 *  between the two - every extent, slab, and output is checked and staged identically.
 */
static void traverse_ranked(JNIEnv* environment, jlong handle, jlong executor, jobject queries, jint storage,
                            jlong byte_offset, jlong rows, jlong columns, jlong stride_bytes, jint scalar_kind,
                            jlong rows_per_query, jbyteArray keys, jdoubleArray scores, jlongArray counts, jlong wanted,
                            jlong expansion_factor, jlongArray statistics) {

    size_t count = 0, width = 0, row_stride = 0, matches = 0, expansion = 0, grouping = 0;
    if (!narrowed(environment, rows, "the query row count is negative", &count)) return;
    if (!narrowed(environment, columns, "the column count is negative", &width)) return;
    if (!narrowed(environment, stride_bytes, "the stride is negative", &row_stride)) return;
    if (!narrowed(environment, wanted, "the wanted count is negative", &matches)) return;
    if (!narrowed(environment, expansion_factor, "the expansion factor is negative", &expansion)) return;
    if (!narrowed(environment, rows_per_query, "the rows per query is negative", &grouping)) return;
    if (!matches) {
        raise(environment, usearch_error_shape_mismatch_k, "the outputs reserve no room for a single match");
        return;
    }

    // The engine resolves one ranked run per query, which a late-interaction batch groups its rows
    // into and a plain search takes one row at a time.
    size_t const resolved = grouping ? count / grouping : count;
    if (grouping && count % grouping) {
        raise(environment, usearch_error_shape_mismatch_k, "the token rows do not divide into whole queries");
        return;
    }

    // The Java side sized its slabs against the same products; re-deriving them here is what keeps
    // an extent that wrapped on the way in from becoming a write past the end of a short slab.
    size_t slots = 0, key_bytes = 0, keys_stride = 0, scores_stride = 0;
    if (!multiplied(environment, resolved, matches, "the match matrix overflows", &slots)) return;
    if (!multiplied(environment, slots, sizeof(usearch_key_t), "the key matrix overflows", &key_bytes)) return;
    if (!multiplied(environment, matches, sizeof(usearch_key_t), "the key row overflows", &keys_stride)) return;
    if (!multiplied(environment, matches, sizeof(usearch_distance_t), "the score row overflows", &scores_stride))
        return;
    if (!keys || !scores || !counts || (size_t)(*environment)->GetArrayLength(environment, keys) < key_bytes ||
        (size_t)(*environment)->GetArrayLength(environment, scores) < slots ||
        (size_t)(*environment)->GetArrayLength(environment, counts) < resolved) {
        raise(environment, usearch_error_shape_mismatch_k, "the outputs are narrower than the batch");
        return;
    }

    // The engine overwrites every slot it reports, so the outputs are staged natively and written
    // back through the region accessors rather than borrowed: borrowing would copy the JVM's
    // untouched rows in for nothing. Counts cross as `size_t` where Java holds them as `long`.
    usearch_key_t* const staged_keys = (usearch_key_t*)calloc(slots, sizeof(usearch_key_t));
    usearch_distance_t* const staged_scores = (usearch_distance_t*)calloc(slots, sizeof(usearch_distance_t));
    size_t* const reported = (size_t*)calloc(resolved, sizeof(size_t));
    jlong* const widened = (jlong*)calloc(resolved, sizeof(jlong));
    void* queries_base = NULL;
    if (!staged_keys || !staged_scores || !reported || !widened) {
        raise(environment, usearch_error_out_of_memory_k, "the ranked outputs could not be staged");
        goto release;
    }

    usearch_dense_search_options_t const options = {expansion};
    usearch_dense_traversal_stats_t stats = {0, 0, 0};
    usearch_error_t error = {usearch_success_k, NULL};

    queries_base = elements_borrowed(environment, queries, (enum storage_t)storage);
    if (!queries_base) {
        raise(environment, usearch_error_malformed_input_k, "the matrix borrows no addressable storage");
        goto release;
    }

    usearch_matrix_view_t const matrix = viewed_matrix((char*)queries_base + byte_offset, count, width, row_stride,
                                                       scalar_kind);
    usearch_matches_span_t const span = {staged_keys, keys_stride, staged_scores, scores_stride, reported, matches};
    if (grouping)
        usearch_dense_maxsim((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor, &matrix,
                             grouping, &span, &options, NULL, NULL, &stats, &error);
    else
        usearch_dense_search((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor, &matrix,
                             &span, &options, NULL, NULL, &stats, &error);
    if (raised(environment, &error)) goto release;

    for (size_t query = 0; query != resolved; ++query) widened[query] = (jlong)reported[query];
    (*environment)->SetByteArrayRegion(environment, keys, 0, (jsize)key_bytes, (jbyte const*)staged_keys);
    (*environment)->SetDoubleArrayRegion(environment, scores, 0, (jsize)slots, (jdouble const*)staged_scores);
    (*environment)->SetLongArrayRegion(environment, counts, 0, (jsize)resolved, widened);
    report(environment, statistics, &stats);

release:
    elements_released(environment, queries, (enum storage_t)storage, queries_base, writeback_discard_k);
    free(staged_keys);
    free(staged_scores);
    free(reported);
    free(widened);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeSearch( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jobject queries, jint storage, jlong byte_offset,
    jlong rows, jlong columns, jlong stride_bytes, jint scalar_kind, jbyteArray keys, jdoubleArray scores,
    jlongArray counts, jlong wanted, jlong expansion_factor, jlongArray statistics) {
    (void)owner;
    traverse_ranked(environment, handle, executor, queries, storage, byte_offset, rows, columns, stride_bytes,
                    scalar_kind, 0, keys, scores, counts, wanted, expansion_factor, statistics);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeMaxsim( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jobject queries, jint storage, jlong byte_offset,
    jlong rows, jlong columns, jlong stride_bytes, jint scalar_kind, jlong rows_per_query, jbyteArray keys,
    jdoubleArray scores, jlongArray counts, jlong wanted, jlong expansion_factor, jlongArray statistics) {
    (void)owner;
    if (rows_per_query <= 0) {
        raise(environment, usearch_error_shape_mismatch_k, "a query cannot carry zero rows");
        return;
    }
    traverse_ranked(environment, handle, executor, queries, storage, byte_offset, rows, columns, stride_bytes,
                    scalar_kind, rows_per_query, keys, scores, counts, wanted, expansion_factor, statistics);
}

#pragma endregion Ranked Walks

#pragma region Retrieval

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeCount( //
    JNIEnv* environment, jclass owner, jlong handle, jbyteArray key) {
    (void)owner;
    usearch_key_t identity = {{0}};
    if (!decoded(environment, key, &identity)) return 0;
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const counted = usearch_dense_count((usearch_dense_index_t)(size_t)handle, identity, &error);
    if (raised(environment, &error)) return 0;
    return (jlong)counted;
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_DenseIndex_nativeGet( //
    JNIEnv* environment, jclass owner, jlong handle, jbyteArray key, jobject vectors, jint storage, jlong byte_offset,
    jlong rows, jlong columns, jlong stride_bytes, jint scalar_kind) {
    (void)owner;

    usearch_key_t identity = {{0}};
    if (!decoded(environment, key, &identity)) return 0;

    size_t capacity = 0, width = 0, row_stride = 0;
    if (!narrowed(environment, rows, "the row capacity is negative", &capacity)) return 0;
    if (!narrowed(environment, columns, "the column count is negative", &width)) return 0;
    if (!narrowed(environment, stride_bytes, "the stride is negative", &row_stride)) return 0;

    usearch_error_t error = {usearch_success_k, NULL};
    void* const rows_base = elements_borrowed(environment, vectors, (enum storage_t)storage);
    if (!rows_base) {
        raise(environment, usearch_error_malformed_input_k, "the matrix borrows no addressable storage");
        return 0;
    }

    usearch_matrix_span_t const span = spanned_matrix((char*)rows_base + byte_offset, capacity, width, row_stride,
                                                      scalar_kind);
    size_t const entries = usearch_dense_get((usearch_dense_index_t)(size_t)handle, identity, &span, &error);
    elements_released(environment, vectors, (enum storage_t)storage, rows_base, writeback_commit_k);

    if (raised(environment, &error)) return 0;
    return (jlong)entries;
}

#pragma endregion Retrieval

#pragma region Serialization

/** @brief The shape @c save and @c load share: borrow a path, hand it across, lift the outcome. */
enum checkpoint_verb_t {
    checkpoint_save_k = 0,
    checkpoint_load_k = 1,
};

static void traverse_checkpoint(JNIEnv* environment, jlong handle, jlong executor, jbyteArray path,
                                enum checkpoint_verb_t verb) {
    if (!path) {
        raise(environment, usearch_error_malformed_input_k, "the path borrows no storage");
        return;
    }
    size_t const length = (size_t)(*environment)->GetArrayLength(environment, path);

    usearch_error_t error = {usearch_success_k, NULL};
    void* const address = elements_borrowed(environment, path, storage_bytes_k);
    if (!address) {
        raise(environment, usearch_error_malformed_input_k, "the path borrows no addressable storage");
        return;
    }

    if (verb == checkpoint_save_k)
        usearch_dense_save((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor,
                           borrowed_bytes(address, length), NULL, NULL, &error);
    else
        usearch_dense_load((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor,
                           borrowed_bytes(address, length), NULL, NULL, &error);
    elements_released(environment, path, storage_bytes_k, address, writeback_discard_k);
    raised(environment, &error);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeSave( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jbyteArray path) {
    (void)owner;
    traverse_checkpoint(environment, handle, executor, path, checkpoint_save_k);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeLoad( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jbyteArray path) {
    (void)owner;
    traverse_checkpoint(environment, handle, executor, path, checkpoint_load_k);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeSaveBuffer( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jobject buffer, jint storage, jlong byte_offset,
    jlong length) {
    (void)owner;
    size_t bytes = 0;
    if (!narrowed(environment, length, "the buffer length is negative", &bytes)) return;

    usearch_error_t error = {usearch_success_k, NULL};
    void* const address = elements_borrowed(environment, buffer, (enum storage_t)storage);
    if (!address) {
        raise(environment, usearch_error_malformed_input_k, "the buffer borrows no addressable storage");
        return;
    }

    usearch_dense_save_buffer((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor,
                              spanned_bytes((char*)address + byte_offset, bytes), NULL, NULL, &error);
    elements_released(environment, buffer, (enum storage_t)storage, address, writeback_commit_k);
    raised(environment, &error);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeLoadBuffer( //
    JNIEnv* environment, jclass owner, jlong handle, jlong executor, jobject buffer, jint storage, jlong byte_offset,
    jlong length) {
    (void)owner;
    size_t bytes = 0;
    if (!narrowed(environment, length, "the buffer length is negative", &bytes)) return;

    usearch_error_t error = {usearch_success_k, NULL};
    void* const address = elements_borrowed(environment, buffer, (enum storage_t)storage);
    if (!address) {
        raise(environment, usearch_error_malformed_input_k, "the buffer borrows no addressable storage");
        return;
    }

    usearch_dense_load_buffer((usearch_dense_index_t)(size_t)handle, (usearch_executor_t)(size_t)executor,
                              borrowed_bytes((char*)address + byte_offset, bytes), NULL, NULL, &error);
    elements_released(environment, buffer, (enum storage_t)storage, address, writeback_discard_k);
    raised(environment, &error);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeMetadata( //
    JNIEnv* environment, jclass owner, jbyteArray path, jlongArray words) {
    (void)owner;
    if (!path) {
        raise(environment, usearch_error_malformed_input_k, "the path borrows no storage");
        return;
    }
    size_t const length = (size_t)(*environment)->GetArrayLength(environment, path);

    usearch_dense_metadata_t metadata = {0};
    usearch_error_t error = {usearch_success_k, NULL};
    void* const address = elements_borrowed(environment, path, storage_bytes_k);
    if (!address) {
        raise(environment, usearch_error_malformed_input_k, "the path borrows no addressable storage");
        return;
    }

    usearch_dense_metadata(borrowed_bytes(address, length), &metadata, &error);
    elements_released(environment, path, storage_bytes_k, address, writeback_discard_k);

    if (raised(environment, &error)) return;
    lift_metadata(environment, &metadata, words);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeMetadataBuffer( //
    JNIEnv* environment, jclass owner, jobject buffer, jint storage, jlong byte_offset, jlong length,
    jlongArray words) {
    (void)owner;
    size_t bytes = 0;
    if (!narrowed(environment, length, "the buffer length is negative", &bytes)) return;

    usearch_dense_metadata_t metadata = {0};
    usearch_error_t error = {usearch_success_k, NULL};
    void* const address = elements_borrowed(environment, buffer, (enum storage_t)storage);
    if (!address) {
        raise(environment, usearch_error_malformed_input_k, "the buffer borrows no addressable storage");
        return;
    }

    usearch_dense_metadata_buffer(borrowed_bytes((char*)address + byte_offset, bytes), &metadata, &error);
    elements_released(environment, buffer, (enum storage_t)storage, address, writeback_discard_k);

    if (raised(environment, &error)) return;
    lift_metadata(environment, &metadata, words);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_DenseIndex_nativeReloadOptions( //
    JNIEnv* environment, jclass owner, jlongArray input, jlongArray output) {
    (void)owner;
    usearch_dense_options_t options;
    if (!lower_options(environment, input, &options)) return;
    lift_options(environment, &options, output);
}

#pragma endregion Serialization
