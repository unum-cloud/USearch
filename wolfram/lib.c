/**
 *  @brief Wolfram LibraryLink bindings for USearch, written against the C interface.
 *  @file wolfram/lib.c
 *  @author Ash Vardanian
 *
 *  LibraryLink hands every verb an @c MArgument array and expects an integer status back, so the
 *  whole binding is C and links the pre-instantiated engine rather than compiling templates of its
 *  own. Wolfram has no object model to hang a handle off, so both the executor and the index cross
 *  the boundary as opaque integers the caller keeps: @c ExecutorCreate and @c DenseIndexCreate mint
 *  them, @c ExecutorDestroy and @c DenseIndexDestroy retire them.
 *
 *  @section wolfram_executors Executors
 *
 *  The workers a fanned-out verb runs on belong to an executor, never to an index, so one pool
 *  serves several indexes and its residency must match theirs. Every verb whose C counterpart takes
 *  one takes it as the leading argument here.
 *
 *  @section wolfram_scalars Tensors and Scalars
 *
 *  Wolfram's real tensors are @c f64, so vectors and queries cross as @c usearch_scalar_f64_k and
 *  the engine quantizes them into whatever the index was built to store. Keys cross as machine
 *  integers, parked big-endian in the low eight bytes of the 128-bit identity.
 */
#include <stdint.h> // `int64_t`, `intptr_t`
#include <stdlib.h> // `malloc`, `free`
#include <string.h> // `memset`, `strlen`

#include <WolframLibrary.h>

#include <usearch/dense_index.h>
#include <usearch/types.h>

#pragma region Vocabulary

/** @brief A borrowed run of bytes over a null-terminated Wolfram string. */
static usearch_bytes_view_t bytes_from_text(char const* text) {
    usearch_bytes_view_t view;
    view.data = (uint8_t const*)text;
    view.length = text ? strlen(text) : 0;
    return view;
}

/**
 *  @brief Reports a failure to the front end and answers the status LibraryLink expects.
 *
 *  The engine's message is static storage, so it is handed over rather than copied. Every verb
 *  ends through here, which is what keeps fifteen distinct statuses from collapsing into one
 *  opaque code the notebook cannot act on.
 */
static int reject(WolframLibraryData library_data, usearch_error_t const* error, char const* fallback) {
    char const* const message = error && error->message ? error->message : fallback;
    if (library_data && message) library_data->Message(message);
    return LIBRARY_FUNCTION_ERROR;
}

/** @brief Answers @c LIBRARY_NO_ERROR when the call succeeded, reporting its message otherwise. */
static int outcome(WolframLibraryData library_data, usearch_error_t const* error) {
    return error->status == usearch_success_k ? LIBRARY_NO_ERROR : reject(library_data, error, NULL);
}

/** @brief Parks a machine integer in the low eight bytes of a 128-bit identity, big-endian. */
static usearch_key_t key_from_integer(mint value) {
    usearch_key_t key;
    memset(key.bytes, 0, sizeof(key.bytes));
    uint64_t const magnitude = (uint64_t)value;
    for (size_t byte_index = 0; byte_index != 8; ++byte_index)
        key.bytes[15 - byte_index] = (uint8_t)(magnitude >> (byte_index * 8));
    return key;
}

/** @brief Reads back what @c key_from_integer parked; identities wider than eight bytes truncate. */
static mint key_to_integer(usearch_key_t key) {
    uint64_t magnitude = 0;
    for (size_t byte_index = 0; byte_index != 8; ++byte_index)
        magnitude = (magnitude << 8) | (uint64_t)key.bytes[8 + byte_index];
    return (mint)magnitude;
}

/** @brief Recovers an opaque handle from the integer a caller kept. */
static void* handle_from_integer(mint value) { return (void*)(intptr_t)value; }

/** @brief Hands an opaque handle back as the integer a caller keeps. */
static mint handle_to_integer(void* handle) { return (mint)(intptr_t)handle; }

#pragma endregion Vocabulary

#pragma region Library Lifecycle

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData library_data) { return LIBRARY_NO_ERROR; }

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData library_data) {}

#pragma endregion Library Lifecycle

#pragma region Executor

/**
 *  @brief Spawns the workers a fanned-out verb runs on: @c residency → @c threads → handle.
 *
 *  The residency picks the factory, and the factory decides what @p threads means - host workers
 *  for RAM, one submission ring apiece for NVMe, and nothing at all for VRAM, where a single host
 *  thread drives one stream per device.
 */
EXTERN_C DLLEXPORT int ExecutorCreate(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    char* residency_text = MArgument_getUTF8String(arguments[0]);
    usearch_residency_t const residency = usearch_residency_from_name(bytes_from_text(residency_text));
    int const named = residency_text && strcmp(residency_text, usearch_residency_name(residency)) == 0;
    library_data->UTF8String_disown(residency_text);
    if (!named) return reject(library_data, NULL, "the residency names neither ram, vram, nor nvme");

    size_t const threads = (size_t)MArgument_getInteger(arguments[1]);

    usearch_ram_options_t ram;
    usearch_vram_options_t vram;
    usearch_nvme_options_t nvme;
    memset(&ram, 0, sizeof(ram));
    memset(&vram, 0, sizeof(vram));
    memset(&nvme, 0, sizeof(nvme));

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_t executor = NULL;
    switch (residency) {
    case usearch_residency_vram_k: executor = usearch_executor_for_vram(&vram, &error); break;
    case usearch_residency_nvme_k: executor = usearch_executor_for_nvme(&nvme, threads, &error); break;
    default: executor = usearch_executor_for_ram(&ram, threads, &error); break;
    }
    if (!executor) return reject(library_data, &error, "the executor could not be spawned");

    MArgument_setInteger(result, handle_to_integer(executor));
    return LIBRARY_NO_ERROR;
}

/** @brief Waits for every deferred call on the executor to finish. */
EXTERN_C DLLEXPORT int ExecutorDrain(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    usearch_executor_t executor = (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0]));
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_executor_drain(executor, &error);
    return outcome(library_data, &error);
}

/** @brief Drains every thread or stream and frees the executor; the handle is invalid afterwards. */
EXTERN_C DLLEXPORT int ExecutorDestroy(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    usearch_executor_free((usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0])));
    return LIBRARY_NO_ERROR;
}

#pragma endregion Executor

#pragma region Dense Index Lifecycle

/**
 *  @brief Builds a dense index and reserves its capacity: @c metric → @c quantization →
 *      @c residency → @c dimensions → @c capacity → @c connectivity → handle.
 *
 *  The candidate-pool breadth rides each call rather than the handle, so it is no argument here.
 *  A zeroed @p connectivity derives from the geometry.
 */
EXTERN_C DLLEXPORT int DenseIndexCreate(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    char* metric_text = MArgument_getUTF8String(arguments[0]);
    char* quantization_text = MArgument_getUTF8String(arguments[1]);
    char* residency_text = MArgument_getUTF8String(arguments[2]);

    // Each lift answers its set's unknown enumerator for a name it does not recognize, which is
    // what distinguishes a misspelling from a deliberate `unknown`.
    usearch_dense_metric_t const metric = usearch_dense_metric_from_name(bytes_from_text(metric_text));
    usearch_scalar_t const quantization = usearch_scalar_from_name(bytes_from_text(quantization_text));
    usearch_residency_t const residency = usearch_residency_from_name(bytes_from_text(residency_text));
    int const named = metric != usearch_dense_metric_unknown_k && quantization != usearch_scalar_unknown_k &&
                      residency_text && strcmp(residency_text, usearch_residency_name(residency)) == 0;

    library_data->UTF8String_disown(metric_text);
    library_data->UTF8String_disown(quantization_text);
    library_data->UTF8String_disown(residency_text);
    if (!named) return reject(library_data, NULL, "the metric, quantization, or residency is not a known name");

    usearch_dense_options_t options;
    memset(&options, 0, sizeof(options));
    options.metric = metric;
    options.quantization = quantization;
    options.residency = residency;
    options.dimensions = (size_t)MArgument_getInteger(arguments[3]);
    options.connectivity = (size_t)MArgument_getInteger(arguments[5]);

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_index_t index = usearch_dense_init(&options, &error);
    if (!index) return reject(library_data, &error, "the index could not be initialized");

    usearch_dense_reserve(index, (size_t)MArgument_getInteger(arguments[4]), &error);
    if (error.status != usearch_success_k) {
        usearch_dense_free(index);
        return reject(library_data, &error, NULL);
    }

    MArgument_setInteger(result, handle_to_integer(index));
    return LIBRARY_NO_ERROR;
}

/** @brief Frees the index; the handle is invalid afterwards. */
EXTERN_C DLLEXPORT int DenseIndexDestroy(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    usearch_dense_free((usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[0])));
    return LIBRARY_NO_ERROR;
}

/**
 *  @brief Writes a checkpoint: @c executor → @c index → @c path.
 *
 *  The path extension picks the layout - a @c .safegraphs path is one monolith, a
 *  @c .safegraphs.json path a manifest beside its shards.
 */
EXTERN_C DLLEXPORT int DenseIndexSave(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    usearch_executor_t executor = (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0]));
    usearch_dense_index_t index = (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[1]));
    char* path_text = MArgument_getUTF8String(arguments[2]);

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_save(index, executor, bytes_from_text(path_text), NULL, NULL, &error);
    library_data->UTF8String_disown(path_text);
    return outcome(library_data, &error);
}

/**
 *  @brief Reads a checkpoint into the handle: @c executor → @c index → @c path.
 *
 *  The handle's residency decides the behavior: materialize into RAM, upload to the devices, or
 *  open in place on disk.
 */
EXTERN_C DLLEXPORT int DenseIndexLoad(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    usearch_executor_t executor = (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0]));
    usearch_dense_index_t index = (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[1]));
    char* path_text = MArgument_getUTF8String(arguments[2]);

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_load(index, executor, bytes_from_text(path_text), NULL, NULL, &error);
    library_data->UTF8String_disown(path_text);
    return outcome(library_data, &error);
}

#pragma endregion Dense Index Lifecycle

#pragma region Dense Index Probes

/** @brief Reports the number of stored entries. */
EXTERN_C DLLEXPORT int DenseIndexSize(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const reported = usearch_dense_size(
        (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[0])), &error);
    if (error.status != usearch_success_k) return reject(library_data, &error, NULL);
    MArgument_setInteger(result, (mint)reported);
    return LIBRARY_NO_ERROR;
}

/** @brief Reports the reserved capacity in entries. */
EXTERN_C DLLEXPORT int DenseIndexCapacity(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const reported = usearch_dense_capacity(
        (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[0])), &error);
    if (error.status != usearch_success_k) return reject(library_data, &error, NULL);
    MArgument_setInteger(result, (mint)reported);
    return LIBRARY_NO_ERROR;
}

/** @brief Reports the frozen dimensionality of the stored vectors. */
EXTERN_C DLLEXPORT int DenseIndexDimensions(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                            MArgument result) {
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const reported = usearch_dense_dimensions(
        (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[0])), &error);
    if (error.status != usearch_success_k) return reject(library_data, &error, NULL);
    MArgument_setInteger(result, (mint)reported);
    return LIBRARY_NO_ERROR;
}

/** @brief Reports the frozen connections-per-node bound of the graph. */
EXTERN_C DLLEXPORT int DenseIndexConnectivity(WolframLibraryData library_data, mint arguments_count,
                                              MArgument* arguments, MArgument result) {
    usearch_error_t error = {usearch_success_k, NULL};
    size_t const reported = usearch_dense_connectivity(
        (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[0])), &error);
    if (error.status != usearch_success_k) return reject(library_data, &error, NULL);
    MArgument_setInteger(result, (mint)reported);
    return LIBRARY_NO_ERROR;
}

#pragma endregion Dense Index Probes

#pragma region Dense Index Verbs

/**
 *  @brief Borrows a real tensor as one input matrix: rank one is a single row, rank two a batch.
 *
 *  A rank-two tensor already @b is what the interface asks for - a base, a row count and a width -
 *  so a whole batch crosses in one call rather than one crossing per vector. The engine checks the
 *  width against the frozen geometry, so nothing here restates that.
 *
 *  @return @c LIBRARY_NO_ERROR when the tensor is real and of rank one or two.
 */
static int matrix_from_tensor(WolframLibraryData library_data, MTensor tensor, usearch_dense_matrix_view_t* matrix) {
    if (library_data->MTensor_getType(tensor) != MType_Real) return LIBRARY_TYPE_ERROR;
    mint const rank = library_data->MTensor_getRank(tensor);
    if (rank != 1 && rank != 2) return LIBRARY_RANK_ERROR;

    mint const* const extents = library_data->MTensor_getDimensions(tensor);
    matrix->data = library_data->MTensor_getRealData(tensor);
    matrix->rows = rank == 1 ? 1 : (size_t)extents[0];
    matrix->columns = rank == 1 ? (size_t)extents[0] : (size_t)extents[1];
    matrix->stride_bytes = 0;
    matrix->scalar_kind = usearch_scalar_f64_k;
    return matrix->data ? LIBRARY_NO_ERROR : LIBRARY_TYPE_ERROR;
}

/**
 *  @brief Inserts a batch: @c executor → @c index → @c keys → @c vectors.
 *
 *  The keys are a rank-one integer tensor and the vectors a rank-two real one, one row apiece;
 *  a single vector is the rank-one case of the same call.
 */
EXTERN_C DLLEXPORT int DenseIndexAdd(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    if (arguments_count != 4) return reject(library_data, NULL, "add takes an executor, an index, keys and vectors");
    usearch_executor_t executor = (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0]));
    usearch_dense_index_t index = (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[1]));

    MTensor const keys_tensor = MArgument_getMTensor(arguments[2]);
    if (library_data->MTensor_getType(keys_tensor) != MType_Integer) return LIBRARY_TYPE_ERROR;
    if (library_data->MTensor_getRank(keys_tensor) != 1) return LIBRARY_RANK_ERROR;
    mint const* const keys_data = library_data->MTensor_getIntegerData(keys_tensor);
    size_t const count = (size_t)library_data->MTensor_getFlattenedLength(keys_tensor);

    usearch_dense_matrix_view_t vectors;
    int const borrowed = matrix_from_tensor(library_data, MArgument_getMTensor(arguments[3]), &vectors);
    if (borrowed != LIBRARY_NO_ERROR) return borrowed;
    if (vectors.rows != count) return reject(library_data, NULL, "the vectors do not split into one row per key");

    usearch_key_t* const keys = (usearch_key_t*)malloc(sizeof(usearch_key_t) * (count ? count : 1));
    if (!keys) return LIBRARY_MEMORY_ERROR;
    for (size_t row = 0; row != count; ++row) keys[row] = key_from_integer(keys_data[row]);

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_add(index, executor, keys, &vectors, NULL, NULL, NULL, NULL, &error);
    free(keys);
    return outcome(library_data, &error);
}

/**
 *  @brief Nearest neighbors of a batch: @c executor → @c index → @c queries → @c wanted → matches.
 *
 *  The result is a rank-three real tensor of `queries × wanted × 2`, each innermost pair a key and
 *  its distance. A pair rather than a bare key because the distance is computed either way, and a
 *  caller that has to recompute it has been handed half an answer; real rather than integer
 *  because one tensor carries both and Wolfram returns one value. Rows past a query's match count
 *  are zeroed, which `DenseIndexSearchCounts` would report - a truncated row is not an error.
 */
EXTERN_C DLLEXPORT int DenseIndexSearch(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    if (arguments_count != 4)
        return reject(library_data, NULL, "search takes an executor, an index, queries and a count");
    usearch_executor_t executor = (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0]));
    usearch_dense_index_t index = (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[1]));

    usearch_dense_matrix_view_t queries;
    int const borrowed = matrix_from_tensor(library_data, MArgument_getMTensor(arguments[2]), &queries);
    if (borrowed != LIBRARY_NO_ERROR) return borrowed;

    mint const requested = MArgument_getInteger(arguments[3]);
    if (requested <= 0) return reject(library_data, NULL, "a search must want at least one match");
    size_t const wanted = (size_t)requested;
    size_t const cells = queries.rows * wanted;

    usearch_key_t* const found = (usearch_key_t*)malloc(sizeof(usearch_key_t) * cells);
    usearch_distance_t* const scores = (usearch_distance_t*)malloc(sizeof(usearch_distance_t) * cells);
    size_t* const counts = (size_t*)malloc(sizeof(size_t) * queries.rows);
    MTensor matches = NULL;
    mint const extents[3] = {(mint)queries.rows, (mint)wanted, 2};
    int status = LIBRARY_MEMORY_ERROR;
    if (!found || !scores || !counts) goto cleanup;
    if (library_data->MTensor_new(MType_Real, 3, extents, &matches) != LIBRARY_NO_ERROR) goto cleanup;

    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_matches_span_t const span = {found, 0, scores, 0, counts, wanted};
    usearch_dense_search(index, executor, &queries, &span, NULL, NULL, NULL, NULL, &error);
    if (error.status != usearch_success_k) {
        library_data->MTensor_free(matches);
        matches = NULL;
        status = reject(library_data, &error, NULL);
        goto cleanup;
    }

    // One bulk fill rather than a call per cell, which is what `MTensor_setReal` would cost.
    double* const cells_data = library_data->MTensor_getRealData(matches);
    for (size_t query = 0; query != queries.rows; ++query) {
        size_t const written = counts[query] < wanted ? counts[query] : wanted;
        for (size_t match = 0; match != wanted; ++match) {
            double* const pair = cells_data + (query * wanted + match) * 2;
            pair[0] = match < written ? (double)key_to_integer(found[query * wanted + match]) : 0.0;
            pair[1] = match < written ? (double)scores[query * wanted + match] : 0.0;
        }
    }
    MArgument_setMTensor(result, matches);
    status = LIBRARY_NO_ERROR;

cleanup:
    free(found);
    free(scores);
    free(counts);
    return status;
}

#pragma endregion Dense Index Verbs
