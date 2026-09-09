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
 *  the engine quantizes them into whatever the index was built to store. An identity is sixteen
 *  bytes and the widest tensor cell is an eight-byte machine integer, so keys cross as integer
 *  tensors of one cell per byte, most significant first - which is what @c IntegerDigits[key,
 *  @c 256, @c 16] renders and @c FromDigits[cells, @c 256] reads back, at full precision either
 *  way. An integer identity is that encoding of a number below @c 2^64, so it parks in the low
 *  eight bytes with the rest zeroed, exactly as the ABI describes.
 *
 *  @section wolfram_outputs Caller-Owned Outputs
 *
 *  A LibraryLink verb answers one value, so a verb with several outputs takes them as @c "Shared"
 *  tensors it writes into rather than folding them into one - which is also what keeps a key's
 *  bytes and a distance's mantissa from having to share a cell type. Shared passing lends the
 *  library the caller's tensor rather than copying it, so each one is disowned before the verb
 *  returns, on the failing paths as much as the succeeding one.
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

/**
 *  @brief Reads one 128-bit identity from sixteen tensor cells, most significant byte first.
 *
 *  @param[in] cells The sixteen cells the identity spans, each holding one byte.
 *  @param[out] key Where the identity lands.
 *  @return @c LIBRARY_NO_ERROR when every cell is a byte.
 *
 *  A cell outside @c 0..255 is refused rather than masked, since a caller who built the identity
 *  from the wrong base wants to hear about it instead of receiving a truncation.
 */
static int key_from_cells(mint const* cells, usearch_key_t* key) {
    for (size_t byte_index = 0; byte_index != sizeof(key->bytes); ++byte_index) {
        if (cells[byte_index] < 0 || cells[byte_index] > 255) return LIBRARY_TYPE_ERROR;
        key->bytes[byte_index] = (uint8_t)cells[byte_index];
    }
    return LIBRARY_NO_ERROR;
}

/**
 *  @brief Writes one 128-bit identity into sixteen tensor cells, most significant byte first.
 *
 *  @param[in] key The identity to render.
 *  @param[out] cells The sixteen cells it spans, each receiving one byte.
 */
static void key_to_cells(usearch_key_t key, mint* cells) {
    for (size_t byte_index = 0; byte_index != sizeof(key.bytes); ++byte_index)
        cells[byte_index] = (mint)key.bytes[byte_index];
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
 *  @brief Checks a tensor's cell type and rank, answering the extents behind them.
 *
 *  @param[in] library_data The callback table the type, rank and extents come from.
 *  @param[in] tensor The tensor to inspect.
 *  @param[in] kind The cell type demanded, either @c MType_Integer or @c MType_Real.
 *  @param[in] rank The rank demanded.
 *  @param[out] extents Where the tensor's extents land, owned by the tensor.
 *  @return @c LIBRARY_NO_ERROR when the tensor matches both.
 */
static int extents_from_tensor(WolframLibraryData library_data, MTensor tensor, mint kind, mint rank,
                               mint const** extents) {
    if (library_data->MTensor_getType(tensor) != kind) return LIBRARY_TYPE_ERROR;
    if (library_data->MTensor_getRank(tensor) != rank) return LIBRARY_RANK_ERROR;
    *extents = library_data->MTensor_getDimensions(tensor);
    return LIBRARY_NO_ERROR;
}

/**
 *  @brief Borrows a real tensor as one input matrix: rank one is a single row, rank two a batch.
 *
 *  A rank-two tensor already @b is what the interface asks for - a base, a row count and a width -
 *  so a whole batch crosses in one call rather than one crossing per vector. The engine checks the
 *  width against the frozen geometry, so nothing here restates that.
 *
 *  @return @c LIBRARY_NO_ERROR when the tensor is real and of rank one or two.
 */
static int matrix_from_tensor(WolframLibraryData library_data, MTensor tensor, usearch_matrix_view_t* matrix) {
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
 *  The keys are a rank-two integer tensor of one identity per row, sixteen cells wide and most
 *  significant byte first, and the vectors a rank-two real one, one row apiece; a single vector is
 *  the rank-one case of the same call.
 */
EXTERN_C DLLEXPORT int DenseIndexAdd(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    if (arguments_count != 4) return reject(library_data, NULL, "add takes an executor, an index, keys and vectors");
    usearch_executor_t executor = (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0]));
    usearch_dense_index_t index = (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[1]));

    MTensor const keys_tensor = MArgument_getMTensor(arguments[2]);
    mint const* keys_extents = NULL;
    int const shaped = extents_from_tensor(library_data, keys_tensor, MType_Integer, 2, &keys_extents);
    if (shaped != LIBRARY_NO_ERROR) return shaped;
    if (keys_extents[1] != (mint)sizeof(usearch_key_t))
        return reject(library_data, NULL, "each key spans sixteen cells, one byte apiece");
    mint const* const keys_cells = library_data->MTensor_getIntegerData(keys_tensor);
    size_t const count = (size_t)keys_extents[0];

    usearch_matrix_view_t vectors;
    int const borrowed = matrix_from_tensor(library_data, MArgument_getMTensor(arguments[3]), &vectors);
    if (borrowed != LIBRARY_NO_ERROR) return borrowed;
    usearch_key_t* const keys = (usearch_key_t*)malloc(sizeof(usearch_key_t) * (count ? count : 1));
    if (!keys) return LIBRARY_MEMORY_ERROR;
    for (size_t row = 0; row != count; ++row) {
        if (key_from_cells(keys_cells + row * sizeof(usearch_key_t), &keys[row]) == LIBRARY_NO_ERROR) continue;
        free(keys);
        return reject(library_data, NULL, "a key cell holds something other than a byte");
    }

    usearch_keys_view_t const identities = {keys, count};
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_add(index, executor, identities, &vectors, NULL, NULL, NULL, NULL, &error);
    free(keys);
    return outcome(library_data, &error);
}

/**
 *  @brief Resolves one batch into three caller-owned tensors, which stay shared on every path.
 *
 *  @param[in] library_data The callback table the tensors are read through.
 *  @param[in] executor The executor the batch fans out over.
 *  @param[in] index The handle to be queried.
 *  @param[in] queries_tensor The real tensor of query rows, one query apiece.
 *  @param[out] keys_tensor The @b [queries,wanted,16] integer tensor the identities land in.
 *  @param[out] distances_tensor The @b [queries,wanted] real tensor the scores land in.
 *  @param[out] counts_tensor The rank-one integer tensor the per-query match counts land in.
 *  @return @c LIBRARY_NO_ERROR when the batch resolved.
 *
 *  Every extent is checked against the queries and against the others, since three tensors sized
 *  apart are three ways to write past a row the caller never allocated.
 */
static int search_into_tensors(WolframLibraryData library_data,                          //
                               usearch_executor_t executor, usearch_dense_index_t index, //
                               MTensor queries_tensor, MTensor keys_tensor,              //
                               MTensor distances_tensor, MTensor counts_tensor) {
    usearch_matrix_view_t queries;
    int const borrowed = matrix_from_tensor(library_data, queries_tensor, &queries);
    if (borrowed != LIBRARY_NO_ERROR) return borrowed;

    mint const* keys_extents = NULL;
    mint const* distances_extents = NULL;
    mint const* counts_extents = NULL;
    int const keys_shaped = extents_from_tensor(library_data, keys_tensor, MType_Integer, 3, &keys_extents);
    if (keys_shaped != LIBRARY_NO_ERROR) return keys_shaped;
    int const distances_shaped = extents_from_tensor(library_data, distances_tensor, MType_Real, 2, &distances_extents);
    if (distances_shaped != LIBRARY_NO_ERROR) return distances_shaped;
    int const counts_shaped = extents_from_tensor(library_data, counts_tensor, MType_Integer, 1, &counts_extents);
    if (counts_shaped != LIBRARY_NO_ERROR) return counts_shaped;

    int const agreed = keys_extents[0] == (mint)queries.rows && distances_extents[0] == keys_extents[0] &&
                       counts_extents[0] == keys_extents[0] && distances_extents[1] == keys_extents[1] &&
                       keys_extents[1] > 0 && keys_extents[2] == (mint)sizeof(usearch_key_t);
    if (!agreed)
        return reject(library_data, NULL,
                      "the outputs must be shaped queries by wanted by sixteen, queries by wanted, and queries");

    size_t const wanted = (size_t)keys_extents[1];
    size_t const cells = queries.rows * wanted;
    usearch_key_t* const found = (usearch_key_t*)malloc(sizeof(usearch_key_t) * (cells ? cells : 1));
    size_t* const counts = (size_t*)malloc(sizeof(size_t) * (queries.rows ? queries.rows : 1));
    if (!found || !counts) {
        free(found);
        free(counts);
        return LIBRARY_MEMORY_ERROR;
    }

    // Scores land in the caller's tensor directly, since a real cell already is what the engine
    // writes; identities stage through `found`, since sixteen bytes do not fit one cell.
    double* const distances_cells = library_data->MTensor_getRealData(distances_tensor);
    usearch_matches_span_t const span = {found, 0, distances_cells, 0, counts, wanted};
    usearch_error_t error = {usearch_success_k, NULL};
    usearch_dense_search(index, executor, &queries, &span, NULL, NULL, NULL, NULL, &error);
    if (error.status != usearch_success_k) {
        free(found);
        free(counts);
        return reject(library_data, &error, NULL);
    }

    usearch_key_t blank;
    memset(blank.bytes, 0, sizeof(blank.bytes));
    mint* const keys_cells = library_data->MTensor_getIntegerData(keys_tensor);
    mint* const counts_cells = library_data->MTensor_getIntegerData(counts_tensor);
    for (size_t query = 0; query != queries.rows; ++query) {
        size_t const written = counts[query] < wanted ? counts[query] : wanted;
        counts_cells[query] = (mint)written;
        for (size_t match = written; match != wanted; ++match) distances_cells[query * wanted + match] = 0.0;
        for (size_t match = 0; match != wanted; ++match) {
            size_t const cell = query * wanted + match;
            key_to_cells(match < written ? found[cell] : blank, keys_cells + cell * sizeof(usearch_key_t));
        }
    }

    free(found);
    free(counts);
    return LIBRARY_NO_ERROR;
}

/**
 *  @brief Nearest neighbors of a batch: @c executor → @c index → @c queries → @c keys →
 *      @c distances → @c counts.
 *
 *  The three outputs are shared tensors the caller allocated: keys of @b [queries,wanted,16]
 *  integer cells, distances of @b [queries,wanted] reals, and one count per query. Three tensors
 *  rather than one because sixteen identity bytes and a distance's mantissa cannot share a cell
 *  type without one of them being rounded, and shared rather than returned because a verb answers
 *  one value. Their extents carry the breadth of the search, so nothing here restates it, and rows
 *  past a query's count are zeroed - a truncated row is not an error.
 */
EXTERN_C DLLEXPORT int DenseIndexSearch(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    if (arguments_count != 6)
        return reject(library_data, NULL, "search takes an executor, an index, queries, keys, distances and counts");
    MTensor const keys_tensor = MArgument_getMTensor(arguments[3]);
    MTensor const distances_tensor = MArgument_getMTensor(arguments[4]);
    MTensor const counts_tensor = MArgument_getMTensor(arguments[5]);

    int const status = search_into_tensors(                                             //
        library_data,                                                                   //
        (usearch_executor_t)handle_from_integer(MArgument_getInteger(arguments[0])),    //
        (usearch_dense_index_t)handle_from_integer(MArgument_getInteger(arguments[1])), //
        MArgument_getMTensor(arguments[2]), keys_tensor, distances_tensor, counts_tensor);

    // Shared passing lends the tensor rather than copying it, so a share the library keeps is one
    // the front end can never reclaim - which is why this runs whether the batch resolved or not.
    library_data->MTensor_disown(keys_tensor);
    library_data->MTensor_disown(distances_tensor);
    library_data->MTensor_disown(counts_tensor);
    return status;
}

#pragma endregion Dense Index Verbs
