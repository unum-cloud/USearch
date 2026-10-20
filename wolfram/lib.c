/**
 *  @file wolfram/lib.c
 *  @author Ash Vardanian
 *  @date September 7, 2026
 *  @brief Wolfram LibraryLink bindings for USearch, written against the C interface.
 *
 *  LibraryLink hands every verb an @c MArgument array and expects an integer status back, so the
 *  whole binding is C and links the pre-instantiated engine rather than compiling templates of its
 *  own. Executors, arenas and indexes cross the boundary as managed library expressions -
 *  @c Executor, @c Arena and @c DenseIndex - which @c USearch.wl creates and the kernel releases
 *  once the last reference to one dies. The verbs here take their IDs and look the handles up in
 *  one table per kind, so an integer a caller typed names no pointer, and a released handle is
 *  refused as @c uninitialized rather than freed twice.
 *
 *  @section wolfram_executors Executors
 *
 *  The workers a fanned-out verb runs on belong to an executor, never to an index, so one pool
 *  serves several indexes and its residency must match theirs. Every verb whose C counterpart takes
 *  one takes its ID, and a zero ID runs the call on the kernel's own thread. Either way the call
 *  blocks, so its outcome is final and its outputs written once the verb returns. The lists an
 *  executor constructor takes - memory domains, devices, streams, paths - cross as @c DataStore
 *  arguments, which unlike a tensor may be empty, and an empty one lets the engine derive.
 *
 *  @section wolfram_arenas Arenas
 *
 *  Every verb whose outputs scale with the call - @c DenseSearch, @c DenseMaxsim, @c DenseGet,
 *  @c DenseMultiply, @c DenseAttend and @c DenseSaveBuffer - takes the ID of an @c Arena, made by
 *  @c ArenaInit where its executor computes, and refuses a released one. The call rewinds and
 *  refills it, and its outputs are copied into Wolfram tensors before the verb returns, since
 *  LibraryLink cannot alias memory it did not allocate, so no result ever reads an arena a later
 *  call rewound. An arena made by a VRAM executor holds device memory the kernel cannot read, so
 *  the verbs refuse it as @c device.
 *
 *  @section wolfram_failures Failures
 *
 *  Every verb answers a @c DataStore: its outputs, named after the C fields they came from, on
 *  success, and otherwise the status name, spelled as @c usearch_status_name spells it, beside the
 *  engine's detail - so a notebook branches on the first and reads the second.
 *
 *  @section wolfram_scalars Tensors and Scalars
 *
 *  Packed real tensors are @c f64; NumericArray rows retain @c f32 or @c f64 storage, and the
 *  engine quantizes them into whatever the index was built to store. An identity is sixteen
 *  bytes, so keys cross as @c UnsignedInteger8 numeric arrays of sixteen cells per key, most
 *  significant first - which is what @c IntegerDigits[key, @c 256, @c 16] renders and
 *  @c FromDigits[cells, @c 256] reads back, at full precision either way. An integer identity is
 *  that encoding of a number below @c 2^64, so it parks in the low eight bytes with the rest
 *  zeroed, exactly as the ABI describes.
 */
#include <stdint.h> // `int32_t`, `intptr_t`, `uint8_t`
#include <stdlib.h> // `malloc`, `realloc`, `free`
#include <string.h> // `memcpy`, `memset`, `strlen`, `strcmp`

#include <WolframIOLibraryFunctions.h>
#include <WolframLibrary.h>
#include <WolframNumericArrayLibrary.h>

#include <numkong/cast.h>

#include "usearch/dense_index.h"
#include "usearch/types.h"

_Static_assert(sizeof(size_t) == sizeof(mint), "counts land in integer tensors cell for cell");

#pragma region Vocabulary

/** A borrowed run of bytes over a null-terminated Wolfram string. */
static usearch_bytes_view_t bytes_from_text(char const* text) {
    usearch_bytes_view_t view;
    view.data = (uint8_t const*)text;
    view.length = text ? strlen(text) : 0;
    return view;
}

/** Answers a failure as a @c DataStore of its status name and detail. */
static int answer_failure(WolframLibraryData library_data, usearch_status_t status, char const* message,
                          MArgument result) {
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    DataStore const store = io->createDataStore();
    if (!store) return LIBRARY_MEMORY_ERROR;
    io->DataStore_addNamedString(store, "Status", (char*)usearch_status_name(status));
    io->DataStore_addNamedString(store, "Message", (char*)(message ? message : ""));
    MArgument_setDataStore(result, store);
    return LIBRARY_NO_ERROR;
}

/** Answers an empty @c DataStore - success with nothing to report. */
static int answer_nothing(WolframLibraryData library_data, MArgument result) {
    DataStore const store = library_data->ioLibraryFunctions->createDataStore();
    if (!store) return LIBRARY_MEMORY_ERROR;
    MArgument_setDataStore(result, store);
    return LIBRARY_NO_ERROR;
}

/** Answers a finished constructor or lookup: nothing on success, its status and detail
 *  otherwise. */
static int answer_outcome(WolframLibraryData library_data, usearch_outcome_t const* outcome, MArgument result) {
    return outcome->status == usearch_status_success_k
               ? answer_nothing(library_data, result)
               : answer_failure(library_data, outcome->status, outcome->message, result);
}

/** Adds @p data under @p name as a byte array of the given extents - keys and monoliths. */
static int store_bytes(WolframLibraryData library_data, DataStore store, char const* name, void const* data, mint rank,
                       mint const* extents) {
    if (!extents[0]) {
        MTensor empty;
        if (library_data->MTensor_new(MType_Integer, rank, extents, &empty) != LIBRARY_NO_ERROR)
            return LIBRARY_MEMORY_ERROR;
        library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, (char*)name, empty);
        return LIBRARY_NO_ERROR;
    }
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    MNumericArray array;
    if (arrays->MNumericArray_new(MNumericArray_Type_UBit8, rank, extents, &array) != LIBRARY_NO_ERROR)
        return LIBRARY_MEMORY_ERROR;
    size_t const length = (size_t)arrays->MNumericArray_getFlattenedLength(array);
    if (length) memcpy(arrays->MNumericArray_getData(array), data, length);
    library_data->ioLibraryFunctions->DataStore_addNamedMNumericArray(store, (char*)name, array);
    return LIBRARY_NO_ERROR;
}

/** Adds @p data under @p name as a real tensor of the given extents - the products of
 *  @c multiply. */
static int store_reals(WolframLibraryData library_data, DataStore store, char const* name, double const* data,
                       mint rank, mint const* extents) {
    MTensor tensor;
    if (library_data->MTensor_new(MType_Real, rank, extents, &tensor) != LIBRARY_NO_ERROR) return LIBRARY_MEMORY_ERROR;
    size_t const length = (size_t)library_data->MTensor_getFlattenedLength(tensor);
    if (length) memcpy(library_data->MTensor_getRealData(tensor), data, length * sizeof(double));
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, (char*)name, tensor);
    return LIBRARY_NO_ERROR;
}

/** Adds one integer under @p name. */
static void store_size(WolframLibraryData library_data, DataStore store, char const* name, size_t value) {
    library_data->ioLibraryFunctions->DataStore_addNamedInteger(store, (char*)name, (mint)value);
}

/** Adds one static string under @p name, or an empty one for @c NULL. */
static void store_text(WolframLibraryData library_data, DataStore store, char const* name, char const* value) {
    library_data->ioLibraryFunctions->DataStore_addNamedString(store, (char*)name, (char*)(value ? value : ""));
}

/** Adds a checkpoint's frozen metric, quantization, dimensions and connectivity. */
static void store_metadata(WolframLibraryData library_data, DataStore store, usearch_dense_metadata_t const* metadata) {
    store_text(library_data, store, "Metric", usearch_dense_metric_name(metadata->metric));
    store_text(library_data, store, "Quantization", usearch_scalar_name(metadata->quantization));
    store_size(library_data, store, "Dimensions", metadata->dimensions);
    store_size(library_data, store, "Connectivity", metadata->connectivity);
}

/**
 *  @brief Reads a @c DataStore of integers as 32-bit ordinals - memory domains or device IDs - into
 *      a fresh array, which the caller frees.
 *  @return @c usearch_status_invalid_argument_k for an entry of another type.
 */
static usearch_status_t ordinals_from_store(WolframLibraryData library_data, MArgument argument, int32_t** ordinals,
                                            size_t* count) {
    DataStore const store = MArgument_getDataStore(argument);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    *count = (size_t)io->DataStore_getLength(store);
    *ordinals = (int32_t*)malloc(sizeof(int32_t) * (*count ? *count : 1));
    if (!*ordinals) return usearch_status_out_of_memory_k;
    size_t position = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); node; node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        if (io->DataStoreNode_getDataType(node) != MType_Integer || io->DataStoreNode_getData(node, &cell) != 0) {
            free(*ordinals);
            *ordinals = NULL;
            return usearch_status_invalid_argument_k;
        }
        (*ordinals)[position++] = (int32_t)MArgument_getInteger(cell);
    }
    return usearch_status_success_k;
}

/**
 *  @brief Reads a @c DataStore of strings as borrowed paths into a fresh array, which the caller
 *      frees.
 *  @return @c usearch_status_invalid_argument_k for an entry of another type.
 */
static usearch_status_t paths_from_store(WolframLibraryData library_data, MArgument argument,
                                         usearch_bytes_view_t** paths, size_t* count) {
    DataStore const store = MArgument_getDataStore(argument);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    *count = (size_t)io->DataStore_getLength(store);
    *paths = (usearch_bytes_view_t*)malloc(sizeof(usearch_bytes_view_t) * (*count ? *count : 1));
    if (!*paths) return usearch_status_out_of_memory_k;
    size_t position = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); node; node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        if (io->DataStoreNode_getDataType(node) != MType_UTF8String || io->DataStoreNode_getData(node, &cell) != 0) {
            free(*paths);
            *paths = NULL;
            return usearch_status_invalid_argument_k;
        }
        (*paths)[position++] = bytes_from_text(MArgument_getUTF8String(cell));
    }
    return usearch_status_success_k;
}

/** Reads a byte array as a borrowed buffer - the monolith the buffer verbs take. */
static usearch_status_t buffer_from_array(WolframLibraryData library_data, MNumericArray array,
                                          usearch_bytes_view_t* buffer) {
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    if (arrays->MNumericArray_getType(array) != MNumericArray_Type_UBit8 || arrays->MNumericArray_getRank(array) != 1)
        return usearch_status_invalid_argument_k;
    buffer->data = (uint8_t const*)arrays->MNumericArray_getData(array);
    buffer->length = (size_t)arrays->MNumericArray_getFlattenedLength(array);
    return usearch_status_success_k;
}

/**
 *  @brief Reads a byte array of sixteen cells per key: rank one is a single key, rank two a batch,
 *      and a rank-one array without cells the empty batch an empty list becomes.
 *  @param[out] keys The keys, borrowing the array's cells.
 *  @param[out] count The keys the array holds.
 */
static usearch_status_t keys_from_array(WolframLibraryData library_data, MNumericArray array,
                                        usearch_key_t const** keys, size_t* count) {
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    mint const rank = arrays->MNumericArray_getRank(array);
    if (arrays->MNumericArray_getType(array) != MNumericArray_Type_UBit8 || (rank != 1 && rank != 2))
        return usearch_status_malformed_k;
    mint const* const extents = arrays->MNumericArray_getDimensions(array);
    *count = rank == 2 ? (size_t)extents[0] : extents[0] != 0;
    if (*count && extents[rank - 1] != (mint)sizeof(usearch_key_t)) return usearch_status_malformed_k;
    *keys = (usearch_key_t const*)arrays->MNumericArray_getData(array);
    return usearch_status_success_k;
}

/**
 *  @brief Borrows a real tensor as a run of rows: rank one is a single row, rank two a batch, and a
 *      rank-one tensor without cells the empty batch an empty list becomes.
 *
 *  The tensor's own width crosses as the view's @c columns, and the engine checks it against the
 *  geometry the index froze.
 */
static usearch_status_t rows_from_tensor(WolframLibraryData library_data, MTensor tensor, usearch_rows_view_t* rows,
                                         size_t* count) {
    mint const rank = library_data->MTensor_getRank(tensor);
    if (library_data->MTensor_getType(tensor) != MType_Real || (rank != 1 && rank != 2))
        return usearch_status_shape_mismatch_k;
    mint const* const extents = library_data->MTensor_getDimensions(tensor);
    rows->data = library_data->MTensor_getRealData(tensor);
    rows->columns = (size_t)extents[rank - 1];
    rows->stride = rows->columns * sizeof(double);
    rows->scalar = usearch_scalar_f64_k;
    *count = rank == 2 ? (size_t)extents[0] : extents[0] != 0;
    return usearch_status_success_k;
}

static usearch_status_t rows_from_array(WolframLibraryData library_data, MNumericArray array, usearch_rows_view_t* rows,
                                        size_t* count) {
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    mint const rank = arrays->MNumericArray_getRank(array);
    int const type = arrays->MNumericArray_getType(array);
    if ((rank != 1 && rank != 2) || (type != MNumericArray_Type_Real32 && type != MNumericArray_Type_Real64))
        return usearch_status_shape_mismatch_k;
    mint const* const extents = arrays->MNumericArray_getDimensions(array);
    rows->data = arrays->MNumericArray_getData(array);
    rows->columns = (size_t)extents[rank - 1];
    rows->scalar = type == MNumericArray_Type_Real32 ? usearch_scalar_f32_k : usearch_scalar_f64_k;
    rows->stride = rows->columns * (type == MNumericArray_Type_Real32 ? sizeof(float) : sizeof(double));
    *count = rank == 2 ? (size_t)extents[0] : extents[0] != 0;
    return usearch_status_success_k;
}

/** Reads a metric and a quantization by their canonical names. */
static int geometry_from_arguments(WolframLibraryData library_data, MArgument const* arguments,
                                   usearch_dense_metric_t* metric, usearch_scalar_t* quantization) {
    char* metric_text = MArgument_getUTF8String(arguments[0]);
    char* quantization_text = MArgument_getUTF8String(arguments[1]);
    *metric = usearch_dense_metric_from_name(bytes_from_text(metric_text));
    *quantization = usearch_scalar_from_name(bytes_from_text(quantization_text));
    library_data->UTF8String_disown(metric_text);
    library_data->UTF8String_disown(quantization_text);
    return *metric != usearch_dense_metric_unknown_k && *quantization != usearch_scalar_unknown_k;
}

#pragma endregion Vocabulary

#pragma region Managed Expressions

/** Live handles of one managed expression kind, indexed by the ID the kernel assigns each
 *  expression. */
typedef struct handles_t {

    /** One slot per ID: @c NULL for an ID no live expression holds, @c empty_slot before its
     *  constructor succeeds. */
    void** slots;

    /** Slots allocated; IDs past them name nothing. */
    mint capacity;
} handles_t;

/** What a slot holds between its expression's creation and its constructor's success. */
static char empty_slot_marker;
static void* const empty_slot = &empty_slot_marker;

static handles_t indexes, executors, arenas;

typedef struct index_t {
    usearch_dense_index_t handle;
    usearch_residency_t residency;
} index_t;

/** What an @c Executor slot holds: the handle and the residency its constructor picked. */
typedef struct executor_t {
    usearch_executor_t handle;
    usearch_residency_t residency;
} executor_t;

/** What an @c Arena slot holds: the handle and the residency of the executor that placed it. */
typedef struct arena_t {
    usearch_arena_t handle;
    usearch_residency_t residency;
} arena_t;

/**
 *  @brief Gives a fresh expression its slot, holding @p content; an ID left without one names
 *      nothing.
 *  @return Whether the slot holds @p content, so the caller frees what it could not admit.
 */
static int handles_admit(handles_t* handles, mint id, void* content) {
    if (id <= 0) return 0;
    if (id >= handles->capacity) {
        mint const capacity = id + 1 > handles->capacity * 2 ? id + 1 : handles->capacity * 2;
        void** const slots = (void**)realloc(handles->slots, sizeof(void*) * (size_t)capacity);
        if (!slots) return 0;
        memset(slots + handles->capacity, 0, sizeof(void*) * (size_t)(capacity - handles->capacity));
        handles->slots = slots;
        handles->capacity = capacity;
    }
    handles->slots[id] = content;
    return 1;
}

/** The slot of an ID a live expression holds, or @c NULL. */
static void** handles_slot(handles_t const* handles, mint id) {
    return id > 0 && id < handles->capacity && handles->slots[id] ? &handles->slots[id] : NULL;
}

/** The handle an ID names, or @c NULL for zero, a released ID, or one whose constructor never
 *  succeeded. */
static void* handles_find(handles_t const* handles, mint id) {
    void** const slot = handles_slot(handles, id);
    return slot && *slot != empty_slot ? *slot : NULL;
}

/** The slot of an ID whose constructor has not run yet, or @c NULL. */
static void** handles_fresh(handles_t const* handles, mint id) {
    void** const slot = handles_slot(handles, id);
    return slot && *slot == empty_slot ? slot : NULL;
}

/** Empties the slot of a released expression, answering the handle it held for the caller to
 *  free. */
static void* handles_release(handles_t* handles, mint id) {
    void** const slot = handles_slot(handles, id);
    if (!slot) return NULL;
    void* const content = *slot;
    *slot = NULL;
    return content == empty_slot ? NULL : content;
}

/** Frees the slot array, once the caller has freed the handles it held. */
static void handles_free(handles_t* handles) {
    free(handles->slots);
    handles->slots = NULL;
    handles->capacity = 0;
}

/** The executor an ID names, or @c NULL for zero, a released ID, or one whose constructor never
 *  succeeded. */
static usearch_executor_t executor_find(mint id) {
    executor_t const* const executor = (executor_t const*)handles_find(&executors, id);
    return executor ? executor->handle : NULL;
}

#pragma endregion Managed Expressions

#pragma region Calls

typedef struct call_t call_t;

/** Lands a final call's outputs in @p store, named after the C fields they came from. */
typedef int (*answer_t)(WolframLibraryData library_data, call_t* call, DataStore store);

/** One blocking index verb on the stack: its receiver, its outcome, and the outputs it lands. */
struct call_t {

    /** The index the verb addresses. */
    usearch_dense_index_t index;
    usearch_residency_t residency;
    usearch_dense_softmax_state_t* carry;

    /** The executor the verb fans out over; @c NULL runs it on the kernel's thread. */
    usearch_executor_t executor;

    /** The outcome the library publishes, final once the verb returns. */
    usearch_outcome_t outcome;

    /** Lands the outputs on success; @c NULL for verbs that answer nothing. */
    answer_t answer;

    /** Queries of @c search and @c maxsim, or left rows of @c multiply: the rows their outputs
     *  hold. */
    size_t count;

    /** Rows of @c get, or pairs of @c multiply and keys of @c maxsim, the call landed. */
    size_t written;

    /** Pairs @c multiply scored before @c wanted cut any, or the keys @c maxsim landed. */
    size_t required;

    /** Entries under the key of @c count, or nodes @c refine changed. */
    size_t tally;

    /** Matches of @c search, inside the arena. */
    usearch_matches_view_t matches;

    /** Rows of @c get, inside the arena. */
    usearch_rows_view_t vectors;

    /** Pairs of @c multiply or keys of @c maxsim, inside the arena. */
    usearch_matches_tape_view_t products;

    /** The @c save_buffer monolith, inside the arena. */
    usearch_bytes_view_t monolith;

    /** The report of @c describe. */
    usearch_dense_index_description_t description;

    /** The audit of @c health. */
    usearch_reachability_t reachability;
};

/** Records a failure the binding caught before the call crossed, as the engine records its own. */
static usearch_status_t call_refuse(call_t* call, usearch_status_t status, char const* message) {
    call->outcome.status = status;
    call->outcome.message = message;
    return status;
}

/** Borrows packed or typed real rows for the blocking call. */
static usearch_status_t call_rows(WolframLibraryData library_data, call_t* call, MArgument argument,
                                  MArgument array_argument, usearch_rows_view_t* rows, size_t* count) {
    usearch_status_t const status = MArgument_getBoolean(array_argument)
                                        ? rows_from_array(library_data, MArgument_getMNumericArray(argument), rows,
                                                          count)
                                        : rows_from_tensor(library_data, MArgument_getMTensor(argument), rows, count);
    if (status != usearch_status_success_k)
        return call_refuse(call, status,
                           "rows are a real vector or matrix, with Real32 or Real64 NumericArray storage");
    return usearch_status_success_k;
}

/** Borrows the keys of a byte array for the call, refusing any other shape. */
static usearch_status_t call_keys(WolframLibraryData library_data, call_t* call, MNumericArray array,
                                  usearch_key_t const** keys, size_t* count) {
    if (keys_from_array(library_data, array, keys, count) != usearch_status_success_k)
        return call_refuse(call, usearch_status_malformed_k, "keys are UnsignedInteger8 arrays, sixteen cells apiece");
    return usearch_status_success_k;
}

/** Reads the one key of a byte array, refusing a batch. */
static usearch_status_t call_key(WolframLibraryData library_data, call_t* call, MNumericArray array,
                                 usearch_key_t* key) {
    usearch_key_t const* keys = NULL;
    size_t count = 0;
    if (keys_from_array(library_data, array, &keys, &count) != usearch_status_success_k || count != 1)
        return call_refuse(call, usearch_status_malformed_k, "a key is an UnsignedInteger8 array of sixteen cells");
    *key = *keys;
    return usearch_status_success_k;
}

/** The arena an ID names for the call, refusing a released one and one holding memory the kernel
 *  cannot read. */
static usearch_arena_t call_arena(call_t* call, mint id) {
    arena_t const* const arena = (arena_t const*)handles_find(&arenas, id);
    if (!arena) return call_refuse(call, usearch_status_uninitialized_k, "the Arena was released"), NULL;
    if (arena->residency == usearch_residency_vram_k)
        return call_refuse(call, usearch_status_device_k, "the Arena holds VRAM, which the kernel cannot read"), NULL;
    return arena->handle;
}

/** Reads the matches a call keeps per row, refusing a negative number rather than wrapping it. */
static usearch_status_t call_size(call_t* call, MArgument argument, size_t* wanted, char const* message) {
    mint const requested = MArgument_getInteger(argument);
    if (requested < 0) return call_refuse(call, usearch_status_invalid_argument_k, message);
    *wanted = (size_t)requested;
    return usearch_status_success_k;
}

/** Reads a traversal by name; the lift answers @c walk for a name it does not recognize, so only a
 *  round trip tells a misspelling from a deliberate @c walk. */
static usearch_status_t call_traversal(WolframLibraryData library_data, call_t* call, MArgument argument,
                                       usearch_traversal_t* traversal) {
    char* traversal_text = MArgument_getUTF8String(argument);
    *traversal = usearch_traversal_from_name(bytes_from_text(traversal_text));
    int const named = traversal_text && strcmp(traversal_text, usearch_traversal_name(*traversal)) == 0;
    library_data->UTF8String_disown(traversal_text);
    if (!named) return call_refuse(call, usearch_status_unknown_kind_k, "the traversal is neither walk nor scan");
    return usearch_status_success_k;
}

/**
 *  @brief Opens a call on the receiver its two leading IDs name: the index and the executor, a zero
 *      executor meaning the kernel's thread.
 *  @return @c usearch_status_success_k when both name live handles, refusing the call otherwise.
 */
static usearch_status_t call_open(MArgument const* arguments, call_t* call) {
    memset(call, 0, sizeof(call_t));
    index_t const* const index = (index_t const*)handles_find(&indexes, MArgument_getInteger(arguments[0]));
    call->index = index ? index->handle : NULL;
    call->residency = index ? index->residency : usearch_residency_ram_k;
    mint const executor_id = MArgument_getInteger(arguments[1]);
    call->executor = executor_find(executor_id);
    if (!call->index)
        return call_refuse(call, usearch_status_uninitialized_k, "the DenseIndex was released or never constructed");
    if (executor_id && !call->executor)
        return call_refuse(call, usearch_status_uninitialized_k, "the Executor was released or never constructed");
    return usearch_status_success_k;
}

/** Answers a returned call: its outputs, copied out of any arena, on success, its status and detail
 *  otherwise. */
static int call_answer(WolframLibraryData library_data, call_t* call, MArgument result) {
    if (call->outcome.status != usearch_status_success_k)
        return answer_failure(library_data, call->outcome.status, call->outcome.message, result);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    DataStore const store = io->createDataStore();
    if (!store) return LIBRARY_MEMORY_ERROR;
    int const answered = call->answer ? call->answer(library_data, call, store) : LIBRARY_NO_ERROR;
    if (answered != LIBRARY_NO_ERROR) {
        io->deleteDataStore(store);
        return answered;
    }
    MArgument_setDataStore(result, store);
    return LIBRARY_NO_ERROR;
}

#pragma endregion Calls

#pragma region Library Lifecycle

/** Frees a released index. */
static void manage_index(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) handles_admit(&indexes, id, empty_slot);
    else {
        index_t* const index = (index_t*)handles_release(&indexes, id);
        if (!index) return;
        usearch_dense_free(index->handle);
        free(index);
    }
}

/** Frees a released executor. */
static void manage_executor(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) {
        handles_admit(&executors, id, empty_slot);
        return;
    }
    executor_t* const executor = (executor_t*)handles_release(&executors, id);
    if (!executor) return;
    usearch_executor_free(executor->handle);
    free(executor);
}

/** Frees a released arena and every output in it. */
static void manage_arena(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) {
        handles_admit(&arenas, id, empty_slot);
        return;
    }
    arena_t* const arena = (arena_t*)handles_release(&arenas, id);
    if (!arena) return;
    usearch_arena_free(arena->handle);
    free(arena);
}

EXTERN_C DLLEXPORT mint WolframLibrary_getVersion(void) { return WolframLibraryVersion; }

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData library_data) {
    int const failed = library_data->registerLibraryExpressionManager("DenseIndex", manage_index) ||
                       library_data->registerLibraryExpressionManager("Executor", manage_executor) ||
                       library_data->registerLibraryExpressionManager("Arena", manage_arena);
    return failed ? LIBRARY_FUNCTION_ERROR : LIBRARY_NO_ERROR;
}

/** Releases whatever the kernel left alive, the executors last since the indexes and arenas may
 *  name them. */
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData library_data) {
    library_data->unregisterLibraryExpressionManager("DenseIndex");
    library_data->unregisterLibraryExpressionManager("Executor");
    library_data->unregisterLibraryExpressionManager("Arena");
    for (mint id = 0; id < indexes.capacity; ++id) manage_index(library_data, 1, id);
    for (mint id = 0; id < arenas.capacity; ++id) manage_arena(library_data, 1, id);
    for (mint id = 0; id < executors.capacity; ++id) manage_executor(library_data, 1, id);
    handles_free(&indexes);
    handles_free(&executors);
    handles_free(&arenas);
}

/** Calls @c usearch_version_name. */
EXTERN_C DLLEXPORT int VersionName(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    MArgument_setUTF8String(result, (char*)usearch_version_name());
    return LIBRARY_NO_ERROR;
}

/** Calls @c usearch_capabilities_compiled. */
EXTERN_C DLLEXPORT int CapabilitiesCompiled(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                            MArgument result) {
    MArgument_setUTF8String(result, (char*)usearch_capabilities_compiled());
    return LIBRARY_NO_ERROR;
}

/** Calls @c usearch_capabilities_available. */
EXTERN_C DLLEXPORT int CapabilitiesAvailable(WolframLibraryData library_data, mint arguments_count,
                                             MArgument* arguments, MArgument result) {
    MArgument_setUTF8String(result, (char*)usearch_capabilities_available());
    return LIBRARY_NO_ERROR;
}

#pragma endregion Library Lifecycle

#pragma region Executor

/** Fills a fresh executor's slot with the handle and its residency, or reports why no executor was
 *  made. */
static int executor_adopt(WolframLibraryData library_data, void** slot, usearch_executor_t handle,
                          usearch_residency_t residency, usearch_outcome_t const* outcome, MArgument result) {
    executor_t* const executor = outcome->status == usearch_status_success_k ? (executor_t*)malloc(sizeof(executor_t))
                                                                             : NULL;
    if (!executor) {
        usearch_executor_free(handle);
        if (outcome->status == usearch_status_success_k)
            return answer_failure(library_data, usearch_status_out_of_memory_k, "the Executor could not be kept",
                                  result);
        return answer_outcome(library_data, outcome, result);
    }
    executor->handle = handle;
    executor->residency = residency;
    *slot = executor;
    return answer_outcome(library_data, outcome, result);
}

/** Calls @c usearch_executor_for_ram: @c executor → @c memoryDomains → @c threads. */
EXTERN_C DLLEXPORT int ExecutorForRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    void** const slot = handles_fresh(&executors, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the Executor is not fresh", result);
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[1], &domains, &domains_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains are a list of integers", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    usearch_executor_for_ram(&(usearch_executor_for_ram_t) {
        .executor = &executor,
        .outcome = &outcome,
        .ram = {domains_count ? domains : NULL, domains_count},
        .threads = (size_t)MArgument_getInteger(arguments[2]),
    });
    free(domains);
    return executor_adopt(library_data, slot, executor, usearch_residency_ram_k, &outcome, result);
}

/** Calls @c usearch_executor_on_pool: @c executor → @c memoryDomains → @c pool. */
EXTERN_C DLLEXPORT int ExecutorOnPool(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    void** const slot = handles_fresh(&executors, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the Executor is not fresh", result);
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[1], &domains, &domains_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains are a list of integers", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    usearch_executor_on_pool(&(usearch_executor_on_pool_t) {
        .executor = &executor,
        .outcome = &outcome,
        .ram = {domains_count ? domains : NULL, domains_count},
        .pool = (void*)(intptr_t)MArgument_getInteger(arguments[2]),
    });
    free(domains);
    return executor_adopt(library_data, slot, executor, usearch_residency_ram_k, &outcome, result);
}

/**
 *  @brief Calls @c usearch_executor_for_vram: @c executor → @c devices → @c streams →
 *      @c scratchLimit.
 *
 *  The streams are borrowed CUDA stream addresses, one per device; an empty store creates its own.
 */
EXTERN_C DLLEXPORT int ExecutorForVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    void** const slot = handles_fresh(&executors, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the Executor is not fresh", result);
    int32_t* devices = NULL;
    size_t devices_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[1], &devices, &devices_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "devices are a list of integers", result);

    DataStore const store = MArgument_getDataStore(arguments[2]);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    size_t const streams_count = (size_t)io->DataStore_getLength(store);
    void** const streams = (void**)malloc(sizeof(void*) * (streams_count ? streams_count : 1));
    usearch_status_t streamed = streams ? usearch_status_success_k : usearch_status_out_of_memory_k;
    size_t position = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); streams && node;
         node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        if (io->DataStoreNode_getDataType(node) != MType_Integer || io->DataStoreNode_getData(node, &cell) != 0) {
            streamed = usearch_status_invalid_argument_k;
            break;
        }
        streams[position++] = (void*)(intptr_t)MArgument_getInteger(cell);
    }
    if (streamed != usearch_status_success_k) {
        free(devices);
        free(streams);
        return answer_failure(library_data, streamed, "streams are a list of integer addresses", result);
    }

    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    usearch_executor_for_vram(&(usearch_executor_for_vram_t) {
        .executor = &executor,
        .outcome = &outcome,
        .vram = {devices_count ? devices : NULL, devices_count},
        .streams = streams_count ? (void* const*)streams : NULL,
        .scratch_limit = (size_t)MArgument_getInteger(arguments[3]),
    });
    free(devices);
    free(streams);
    return executor_adopt(library_data, slot, executor, usearch_residency_vram_k, &outcome, result);
}

/** Calls @c usearch_executor_for_nvme: @c executor → @c paths → @c threads. */
EXTERN_C DLLEXPORT int ExecutorForNvme(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    void** const slot = handles_fresh(&executors, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the Executor is not fresh", result);
    usearch_bytes_view_t* paths = NULL;
    size_t paths_count = 0;
    usearch_status_t const read = paths_from_store(library_data, arguments[1], &paths, &paths_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "paths are a list of strings", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_t executor = NULL;
    usearch_executor_for_nvme(&(usearch_executor_for_nvme_t) {
        .executor = &executor,
        .outcome = &outcome,
        .nvme = {paths_count ? paths : NULL, paths_count},
        .threads = (size_t)MArgument_getInteger(arguments[2]),
    });
    free(paths);
    return executor_adopt(library_data, slot, executor, usearch_residency_nvme_k, &outcome, result);
}

/** Calls @c usearch_executor_drain: @c executor → @c mood, @c parked or @c spinning. */
EXTERN_C DLLEXPORT int ExecutorDrain(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    usearch_executor_t const executor = executor_find(MArgument_getInteger(arguments[0]));
    if (!executor)
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    // The lift answers `parked` for a name it does not recognize, so only a round trip tells a
    // misspelling from a deliberate `parked`.
    char* mood_text = MArgument_getUTF8String(arguments[1]);
    usearch_mood_t const mood = usearch_mood_from_name(bytes_from_text(mood_text));
    int const named = mood_text && strcmp(mood_text, usearch_mood_name(mood)) == 0;
    library_data->UTF8String_disown(mood_text);
    if (!named)
        return answer_failure(library_data, usearch_status_invalid_argument_k,
                              "the mood is neither parked nor spinning", result);

    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_executor_drain(&(usearch_executor_drain_t) {.executor = executor, .outcome = &outcome, .mood = mood});
    return answer_outcome(library_data, &outcome, result);
}

/** Calls @c usearch_arena_init: @c arena → @c executor, a zero ID making host pages for the
 *  kernel's own thread. */
EXTERN_C DLLEXPORT int ArenaInit(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                 MArgument result) {
    void** const slot = handles_fresh(&arenas, MArgument_getInteger(arguments[0]));
    if (!slot) return answer_failure(library_data, usearch_status_invalid_argument_k, "the Arena is not fresh", result);
    mint const executor_id = MArgument_getInteger(arguments[1]);
    executor_t const* const executor = (executor_t const*)handles_find(&executors, executor_id);
    if (executor_id && !executor)
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    arena_t* const arena = (arena_t*)malloc(sizeof(arena_t));
    if (!arena)
        return answer_failure(library_data, usearch_status_out_of_memory_k, "the Arena could not be kept", result);
    arena->handle = NULL;
    arena->residency = executor ? executor->residency : usearch_residency_ram_k;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_arena_init(&(usearch_arena_init_t) {
        .arena = &arena->handle,
        .executor = executor ? executor->handle : NULL,
        .outcome = &outcome,
    });
    if (outcome.status == usearch_status_success_k) *slot = arena;
    else usearch_arena_free(arena->handle), free(arena);
    return answer_outcome(library_data, &outcome, result);
}

#pragma endregion Executor

#pragma region Dense Index Lifecycle

/** Fills a fresh index's slot, or reports why no index was made. */
static int index_adopt(WolframLibraryData library_data, void** slot, usearch_dense_index_t handle,
                       usearch_residency_t residency, usearch_outcome_t const* outcome, MArgument result) {
    if (outcome->status != usearch_status_success_k) {
        usearch_dense_free(handle);
        return answer_outcome(library_data, outcome, result);
    }
    index_t* const index = (index_t*)malloc(sizeof(index_t));
    if (!index) {
        usearch_dense_free(handle);
        return answer_failure(library_data, usearch_status_out_of_memory_k, "the DenseIndex could not be kept", result);
    }
    *index = (index_t) {handle, residency};
    *slot = index;
    return answer_nothing(library_data, result);
}

/** Resolves the optional executor a constructor fans out over; zero is the kernel's thread. */
static int executor_from_argument(MArgument argument, usearch_executor_t* executor) {
    mint const executor_id = MArgument_getInteger(argument);
    *executor = executor_find(executor_id);
    return !executor_id || *executor;
}

/**
 *  @brief Calls @c usearch_dense_init_ram: @c index → @c metric → @c quantization →
 *      @c dimensions → @c connectivity → @c valuesPerKey → @c memoryDomains.
 *
 *  Zero @p connectivity and @p valuesPerKey derive, and an empty list of memory domains places a
 *  shard on every node.
 */
EXTERN_C DLLEXPORT int DenseInitRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_dense_metric_t metric;
    usearch_scalar_t quantization;
    if (!geometry_from_arguments(library_data, arguments + 1, &metric, &quantization))
        return answer_failure(library_data, usearch_status_unknown_kind_k,
                              "the metric or quantization is not a known name", result);
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[6], &domains, &domains_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains are a list of integers", result);

    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_ram(&(usearch_dense_init_ram_t) {
        .index = &index,
        .outcome = &outcome,
        .metric = metric,
        .quantization = quantization,
        .dimensions = (size_t)MArgument_getInteger(arguments[3]),
        .connectivity = (size_t)MArgument_getInteger(arguments[4]),
        .values_per_key = (size_t)MArgument_getInteger(arguments[5]),
        .ram = {domains_count ? domains : NULL, domains_count},
    });
    free(domains);
    return index_adopt(library_data, slot, index, usearch_residency_ram_k, &outcome, result);
}

/** Calls @c usearch_dense_init_vram: @c index → @c metric → @c quantization → @c dimensions
 *  → @c connectivity → @c valuesPerKey → @c devices. */
EXTERN_C DLLEXPORT int DenseInitVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_dense_metric_t metric;
    usearch_scalar_t quantization;
    if (!geometry_from_arguments(library_data, arguments + 1, &metric, &quantization))
        return answer_failure(library_data, usearch_status_unknown_kind_k,
                              "the metric or quantization is not a known name", result);
    int32_t* devices = NULL;
    size_t devices_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[6], &devices, &devices_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "devices are a list of integers", result);

    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_init_vram(&(usearch_dense_init_vram_t) {
        .index = &index,
        .outcome = &outcome,
        .metric = metric,
        .quantization = quantization,
        .dimensions = (size_t)MArgument_getInteger(arguments[3]),
        .connectivity = (size_t)MArgument_getInteger(arguments[4]),
        .values_per_key = (size_t)MArgument_getInteger(arguments[5]),
        .vram = {devices_count ? devices : NULL, devices_count},
    });
    free(devices);
    return index_adopt(library_data, slot, index, usearch_residency_vram_k, &outcome, result);
}

/**
 *  @brief Calls @c usearch_dense_load_ram: @c index → @c executor → @c path →
 *      @c memoryDomains → @c valuesPerKey.
 *
 *  The path extension picks the layout - a @c .safegraphs path is one monolith, a
 *  @c .safegraphs.json path a manifest beside its shards.
 */
EXTERN_C DLLEXPORT int DenseLoadRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &domains, &domains_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains are a list of integers", result);

    char* path_text = MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_load_ram(&(usearch_dense_load_ram_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .ram = {domains_count ? domains : NULL, domains_count},
        .values_per_key = (size_t)MArgument_getInteger(arguments[4]),
    });
    library_data->UTF8String_disown(path_text);
    free(domains);
    return index_adopt(library_data, slot, index, usearch_residency_ram_k, &outcome, result);
}

/** Calls @c usearch_dense_load_vram: @c index → @c executor → @c path → @c devices →
 *  @c valuesPerKey. */
EXTERN_C DLLEXPORT int DenseLoadVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    int32_t* devices = NULL;
    size_t devices_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &devices, &devices_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "devices are a list of integers", result);

    char* path_text = MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_load_vram(&(usearch_dense_load_vram_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .vram = {devices_count ? devices : NULL, devices_count},
        .values_per_key = (size_t)MArgument_getInteger(arguments[4]),
    });
    library_data->UTF8String_disown(path_text);
    free(devices);
    return index_adopt(library_data, slot, index, usearch_residency_vram_k, &outcome, result);
}

/** Calls @c usearch_dense_load_ram_buffer: @c index → @c executor → @c monolith →
 *  @c memoryDomains → @c valuesPerKey. */
EXTERN_C DLLEXPORT int DenseLoadRamBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer;
    if (buffer_from_array(library_data, MArgument_getMNumericArray(arguments[2]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &domains, &domains_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains are a list of integers", result);

    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_load_ram_buffer(&(usearch_dense_load_ram_buffer_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .buffer = buffer,
        .ram = {domains_count ? domains : NULL, domains_count},
        .values_per_key = (size_t)MArgument_getInteger(arguments[4]),
    });
    free(domains);
    return index_adopt(library_data, slot, index, usearch_residency_ram_k, &outcome, result);
}

/** Calls @c usearch_dense_load_vram_buffer: @c index → @c executor → @c monolith → @c devices
 *  → @c valuesPerKey. */
EXTERN_C DLLEXPORT int DenseLoadVramBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                           MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer;
    if (buffer_from_array(library_data, MArgument_getMNumericArray(arguments[2]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    int32_t* devices = NULL;
    size_t devices_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &devices, &devices_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "devices are a list of integers", result);

    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_load_vram_buffer(&(usearch_dense_load_vram_buffer_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .buffer = buffer,
        .vram = {devices_count ? devices : NULL, devices_count},
        .values_per_key = (size_t)MArgument_getInteger(arguments[4]),
    });
    free(devices);
    return index_adopt(library_data, slot, index, usearch_residency_vram_k, &outcome, result);
}

/** Calls @c usearch_dense_view_ram: @c index → @c path, served in place and read-only. */
EXTERN_C DLLEXPORT int DenseViewRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_view_ram(
        &(usearch_dense_view_ram_t) {.index = &index, .outcome = &outcome, .path = bytes_from_text(path_text)});
    library_data->UTF8String_disown(path_text);
    return index_adopt(library_data, slot, index, usearch_residency_ram_k, &outcome, result);
}

/** Calls @c usearch_dense_view_nvme: @c index → @c executor → @c path → @c paths, served
 *  read-only. */
EXTERN_C DLLEXPORT int DenseViewNvme(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t* paths = NULL;
    size_t paths_count = 0;
    usearch_status_t const read = paths_from_store(library_data, arguments[3], &paths, &paths_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "paths are a list of strings", result);

    char* path_text = MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .nvme = {paths_count ? paths : NULL, paths_count},
    });
    library_data->UTF8String_disown(path_text);
    free(paths);
    return index_adopt(library_data, slot, index, usearch_residency_nvme_k, &outcome, result);
}

/** Calls @c usearch_dense_metadata_read: @c executor → @c path → the checkpoint's frozen
 *  geometry. */
EXTERN_C DLLEXPORT int DenseMetadataRead(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_dense_metadata_t metadata;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_metadata_read(&(usearch_dense_metadata_read_t) {
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .metadata = &metadata,
    });
    library_data->UTF8String_disown(path_text);
    if (outcome.status != usearch_status_success_k) return answer_outcome(library_data, &outcome, result);
    int const answered = answer_nothing(library_data, result);
    if (answered == LIBRARY_NO_ERROR) store_metadata(library_data, MArgument_getDataStore(result), &metadata);
    return answered;
}

/** Calls @c usearch_dense_metadata_read_buffer: @c executor → @c monolith → the monolith's
 *  frozen geometry. */
EXTERN_C DLLEXPORT int DenseMetadataReadBuffer(WolframLibraryData library_data, mint arguments_count,
                                               MArgument* arguments, MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer;
    if (buffer_from_array(library_data, MArgument_getMNumericArray(arguments[1]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    usearch_dense_metadata_t metadata;
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_metadata_read_buffer(&(usearch_dense_metadata_read_buffer_t) {
        .executor = executor,
        .outcome = &outcome,
        .buffer = buffer,
        .metadata = &metadata,
    });
    if (outcome.status != usearch_status_success_k) return answer_outcome(library_data, &outcome, result);
    int const answered = answer_nothing(library_data, result);
    if (answered == LIBRARY_NO_ERROR) store_metadata(library_data, MArgument_getDataStore(result), &metadata);
    return answered;
}

#pragma endregion Dense Index Lifecycle

#pragma region Dense Index Verbs

/** Copies the matches of a search out of the arena: every query's found keys and scores end to
 *  end, as @b [found,16] bytes and @b [found] reals, and a count per query. */
static int answer_matches(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_matches_view_t const matches = call->matches;
    size_t found = 0;
    for (size_t row = 0; row != call->count; ++row) found += matches.counts[row];
    mint const rows = (mint)call->count, extents[2] = {(mint)found, (mint)sizeof(usearch_key_t)};
    MTensor counts = NULL, scores = NULL;
    MNumericArray keys = NULL;
    if (library_data->MTensor_new(MType_Integer, 1, &rows, &counts) != LIBRARY_NO_ERROR ||
        library_data->MTensor_new(MType_Real, 1, extents, &scores) != LIBRARY_NO_ERROR ||
        (found && library_data->numericarrayLibraryFunctions->MNumericArray_new(MNumericArray_Type_UBit8, 2, extents,
                                                                                &keys) != LIBRARY_NO_ERROR)) {
        if (counts) library_data->MTensor_free(counts);
        if (scores) library_data->MTensor_free(scores);
        return LIBRARY_MEMORY_ERROR;
    }
    usearch_key_t* const keys_cells =
        keys ? (usearch_key_t*)library_data->numericarrayLibraryFunctions->MNumericArray_getData(keys) : NULL;
    double* const scores_cells = library_data->MTensor_getRealData(scores);
    mint* const counts_cells = library_data->MTensor_getIntegerData(counts);
    // Columns past a row's count are unwritten, so only the matches it found are copied.
    for (size_t row = 0, landed = 0; row != call->count; landed += matches.counts[row++]) {
        counts_cells[row] = (mint)matches.counts[row];
        if (!matches.counts[row]) continue;
        memcpy(keys_cells + landed, matches.keys + row * matches.columns, matches.counts[row] * sizeof(usearch_key_t));
        memcpy(scores_cells + landed, matches.scores + row * matches.columns, matches.counts[row] * sizeof(double));
    }
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    io->DataStore_addNamedMTensor(store, "Scores", scores);
    io->DataStore_addNamedMTensor(store, "Counts", counts);
    if (keys) io->DataStore_addNamedMNumericArray(store, "Keys", keys);
    return keys ? LIBRARY_NO_ERROR : store_bytes(library_data, store, "Keys", NULL, 2, extents);
}

/** Copies the rows of @c get out of the arena as a @b [written,columns] real matrix, whatever their
 *  stride. */
static int answer_get(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_rows_view_t const vectors = call->vectors;
    mint const extents[2] = {(mint)call->written, (mint)vectors.columns};
    MTensor tensor;
    if (library_data->MTensor_new(MType_Real, 2, extents, &tensor) != LIBRARY_NO_ERROR) return LIBRARY_MEMORY_ERROR;
    double* const cells = library_data->MTensor_getRealData(tensor);
    for (size_t row = 0; row != call->written; ++row)
        memcpy(cells + row * vectors.columns, (uint8_t const*)vectors.data + row * vectors.stride,
               vectors.columns * sizeof(double));
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, call->carry ? "Outputs" : "Vectors", tensor);
    return LIBRARY_NO_ERROR;
}

/** Copies the pairs of @c multiply out of the arena: offsets, keys of @b [written,16] bytes,
 *  products, and the pairs scored. */
static int answer_multiply(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_matches_tape_view_t const products = call->products;
    mint const offsets_extents[1] = {(mint)call->count + 1};
    mint const keys_extents[2] = {(mint)call->written, (mint)sizeof(usearch_key_t)};
    MTensor offsets;
    if (library_data->MTensor_new(MType_Integer, 1, offsets_extents, &offsets) != LIBRARY_NO_ERROR)
        return LIBRARY_MEMORY_ERROR;
    mint* const offsets_cells = library_data->MTensor_getIntegerData(offsets);
    // A call with nothing to report leaves its offsets NULL, which reads as every row empty.
    if (products.offsets) memcpy(offsets_cells, products.offsets, (call->count + 1) * sizeof(size_t));
    else memset(offsets_cells, 0, (call->count + 1) * sizeof(mint));
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, "Offsets", offsets);
    int status = store_bytes(library_data, store, "Keys", products.keys, 2, keys_extents);
    if (status == LIBRARY_NO_ERROR)
        status = store_reals(library_data, store, "Scores", products.scores, 1, keys_extents);
    if (status == LIBRARY_NO_ERROR) store_size(library_data, store, "Required", call->required);
    return status;
}

/** Copies the keys of @c maxsim out of the arena as @c multiply lands its pairs; MaxSim reports no
 *  totals, so the last offset counts both. */
static int answer_maxsim(WolframLibraryData library_data, call_t* call, DataStore store) {
    call->written = call->required = call->products.offsets ? call->products.offsets[call->count] : 0;
    return answer_multiply(library_data, call, store);
}

/** Copies the @c save_buffer monolith out of the arena as an @c UnsignedInteger8 vector. */
static int answer_save_buffer(WolframLibraryData library_data, call_t* call, DataStore store) {
    mint const extents[1] = {(mint)call->monolith.length};
    return store_bytes(library_data, store, "Bytes", call->monolith.data, 1, extents);
}

/** Lands the entries under a key. */
static int answer_count(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Entries", call->tally);
    return LIBRARY_NO_ERROR;
}

/** Lands the nodes a refinement changed. */
static int answer_refine(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Changed", call->tally);
    return LIBRARY_NO_ERROR;
}

/** Lands a reachability audit. */
static int answer_health(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Audited", call->reachability.audited);
    store_size(library_data, store, "Reachable", call->reachability.reachable);
    store_size(library_data, store, "Sweeps", call->reachability.sweeps);
    store_size(library_data, store, "BackSweeps", call->reachability.back_sweeps);
    return LIBRARY_NO_ERROR;
}

/** Lands a live index's frozen metadata and current counters. */
static int answer_describe(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_metadata(library_data, store, &call->description.metadata);
    store_size(library_data, store, "Size", call->description.size);
    store_size(library_data, store, "Capacity", call->description.capacity);
    store_size(library_data, store, "MemoryUsage", call->description.memory_usage);
    store_text(library_data, store, "Capabilities", call->description.capabilities);
    return LIBRARY_NO_ERROR;
}

/** The two leading IDs every verb below takes: @c index → @c executor. */
static size_t const receiver_arguments = 2;

/** Calls @c usearch_dense_reserve: @c capacity, the desired total including the current size. */
EXTERN_C DLLEXPORT int DenseReserve(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t call;
    if (call_open(arguments, &call) == usearch_status_success_k)
        usearch_dense_reserve(&(usearch_dense_reserve_t) {
            .index = call.index,
            .executor = call.executor,
            .outcome = &call.outcome,
            .capacity = (size_t)MArgument_getInteger(arguments[receiver_arguments]),
        });
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_clear, keeping the reserved capacity and frozen geometry. */
EXTERN_C DLLEXPORT int DenseClear(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                  MArgument result) {
    call_t call;
    if (call_open(arguments, &call) == usearch_status_success_k)
        usearch_dense_clear(&(usearch_dense_clear_t) {
            .index = call.index,
            .executor = call.executor,
            .outcome = &call.outcome,
        });
    return call_answer(library_data, &call, result);
}

/**
 *  @brief Calls @c usearch_dense_insert: @c keys → @c vectors → @c expansionFactor.
 *
 *  The keys are an @c UnsignedInteger8 array of one sixteen-byte identity per row, and the vectors
 *  a real matrix, one row apiece; a single vector is the rank-one case of both.
 */
EXTERN_C DLLEXPORT int DenseInsert(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    size_t expansion = 0;
    usearch_key_t const* keys = NULL;
    usearch_rows_view_t vectors;
    size_t keys_count = 0, vectors_count = 0;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_size(&call, arguments[receiver_arguments + 3], &expansion, "expansion factor must not be negative") !=
            usearch_status_success_k ||
        call_keys(library_data, &call, MArgument_getMNumericArray(arguments[receiver_arguments]), &keys, &keys_count) !=
            usearch_status_success_k ||
        call_rows(library_data, &call, arguments[receiver_arguments + 1], arguments[receiver_arguments + 2], &vectors,
                  &vectors_count) != usearch_status_success_k)
        return call_answer(library_data, &call, result);
    // The call has one batch axis, so only this side sees two arrays that could disagree on it.
    if (keys_count != vectors_count) {
        call_refuse(&call, usearch_status_shape_mismatch_k, "there must be one key per vector");
        return call_answer(library_data, &call, result);
    }
    usearch_dense_insert(&(usearch_dense_insert_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .count = keys_count,
        .keys = {keys, sizeof(usearch_key_t)},
        .vectors = vectors,
        .expansion_factor = expansion,
    });
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_refine: @c expansionFactor → the nodes whose neighbors changed. */
EXTERN_C DLLEXPORT int DenseRefine(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    if (call_open(arguments, &call) == usearch_status_success_k) {
        call.answer = answer_refine;
        usearch_dense_refine(&(usearch_dense_refine_t) {
            .index = call.index,
            .executor = call.executor,
            .outcome = &call.outcome,
            .changed = &call.tally,
            .expansion_factor = (size_t)MArgument_getInteger(arguments[receiver_arguments]),
        });
    }
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_health: the population audited, the entries a search reaches, and the
 *  sweeps. */
EXTERN_C DLLEXPORT int DenseHealth(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    if (call_open(arguments, &call) == usearch_status_success_k) {
        call.answer = answer_health;
        usearch_dense_health(&(usearch_dense_health_t) {
            .index = call.index,
            .executor = call.executor,
            .outcome = &call.outcome,
            .reachability = &call.reachability,
        });
    }
    return call_answer(library_data, &call, result);
}

/**
 *  @brief Calls @c usearch_dense_search: @c queries → @c wanted → @c arena →
 *      @c expansionFactor → @c traversal, @c walk or @c scan → the found keys of @b [found,16]
 *      bytes, their scores, and one count per query.
 *
 *  A query that found fewer than @p wanted matches is not an error, and its count says so.
 */
EXTERN_C DLLEXPORT int DenseSearch(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    size_t expansion = 0;
    usearch_rows_view_t queries;
    size_t wanted = 0;
    usearch_arena_t arena = NULL;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_size(&call, arguments[receiver_arguments + 4], &expansion, "expansion factor must not be negative") !=
            usearch_status_success_k ||
        call_rows(library_data, &call, arguments[receiver_arguments], arguments[receiver_arguments + 1], &queries,
                  &call.count) != usearch_status_success_k ||
        call_size(&call, arguments[receiver_arguments + 2], &wanted, "wanted must not be negative") !=
            usearch_status_success_k ||
        call_traversal(library_data, &call, arguments[receiver_arguments + 5], &traversal) !=
            usearch_status_success_k ||
        !(arena = call_arena(&call, MArgument_getInteger(arguments[receiver_arguments + 3]))))
        return call_answer(library_data, &call, result);
    call.answer = answer_matches;
    usearch_dense_search(&(usearch_dense_search_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .count = call.count,
        .queries = queries,
        .matches = &call.matches,
        .arena = arena,
        .wanted = wanted,
        .expansion_factor = expansion,
        .traversal = traversal,
    });
    return call_answer(library_data, &call, result);
}

/**
 *  @brief Calls @c usearch_dense_maxsim: @c queries → @c rowsPerQuery → @c wanted → @c arena →
 *      @c expansionFactor → offsets, the scored keys, their summed maxima, and the keys landed.
 *
 *  The token rows of every query are laid end to end, so their number divides by @p rowsPerQuery.
 *  Query @c i spans keys @c offsets[[i+1]]+1 through @c offsets[[i+2]], largest first; keys are
 *  @b [landed,16] byte cells. A zero @p wanted keeps every key a query scored.
 */
EXTERN_C DLLEXPORT int DenseMaxsim(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    size_t expansion = 0;
    usearch_rows_view_t queries;
    size_t rows = 0, wanted = 0;
    usearch_arena_t arena = NULL;
    size_t const rows_per_query = (size_t)MArgument_getInteger(arguments[receiver_arguments + 2]);
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_size(&call, arguments[receiver_arguments + 5], &expansion, "expansion factor must not be negative") !=
            usearch_status_success_k ||
        call_rows(library_data, &call, arguments[receiver_arguments], arguments[receiver_arguments + 1], &queries,
                  &rows) != usearch_status_success_k)
        return call_answer(library_data, &call, result);
    if (!rows_per_query || rows % rows_per_query) {
        call_refuse(&call, usearch_status_shape_mismatch_k, "the token rows must split into whole queries");
        return call_answer(library_data, &call, result);
    }
    if (call_size(&call, arguments[receiver_arguments + 3], &wanted, "wanted must not be negative") !=
            usearch_status_success_k ||
        !(arena = call_arena(&call, MArgument_getInteger(arguments[receiver_arguments + 4]))))
        return call_answer(library_data, &call, result);
    call.count = rows / rows_per_query;
    call.answer = answer_maxsim;
    usearch_dense_maxsim(&(usearch_dense_maxsim_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .count = call.count,
        .queries = queries,
        .rows_per_query = rows_per_query,
        .matches = &call.products,
        .arena = arena,
        .wanted = wanted,
        .expansion_factor = expansion,
    });
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_count: @c key → the entries stored under it. */
EXTERN_C DLLEXPORT int DenseCount(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                  MArgument result) {
    call_t call;
    usearch_key_t key;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_key(library_data, &call, MArgument_getMNumericArray(arguments[receiver_arguments]), &key) !=
            usearch_status_success_k)
        return call_answer(library_data, &call, result);
    call.answer = answer_count;
    usearch_dense_count(&(usearch_dense_count_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .key = key,
        .entries = &call.tally,
    });
    return call_answer(library_data, &call, result);
}

/**
 *  @brief Calls @c usearch_dense_get: @c key → @c arena → every vector stored under the key, as
 *      a @b [rows,columns] real matrix.
 *
 *  The rows land in the arena as @c f64 and are copied out, so an absent key answers an empty
 *  matrix.
 */
EXTERN_C DLLEXPORT int DenseGet(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                MArgument result) {
    call_t call;
    usearch_key_t key;
    usearch_arena_t arena = NULL;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_key(library_data, &call, MArgument_getMNumericArray(arguments[receiver_arguments]), &key) !=
            usearch_status_success_k ||
        !(arena = call_arena(&call, MArgument_getInteger(arguments[receiver_arguments + 1]))))
        return call_answer(library_data, &call, result);
    call.answer = answer_get;
    usearch_dense_get(&(usearch_dense_get_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .key = key,
        .vectors = &call.vectors,
        .written = &call.written,
        .arena = arena,
        .outputs_scalar = usearch_scalar_f64_k,
    });
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_save: @c path, whose extension picks the layout - a @c .safegraphs path
 *  is one monolith, a @c .safegraphs.json path a manifest beside its shards. */
EXTERN_C DLLEXPORT int DenseSave(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                 MArgument result) {
    call_t call;
    if (call_open(arguments, &call) == usearch_status_success_k) {
        char* path_text = MArgument_getUTF8String(arguments[receiver_arguments]);
        usearch_dense_save(&(usearch_dense_save_t) {
            .index = call.index,
            .executor = call.executor,
            .outcome = &call.outcome,
            .path = bytes_from_text(path_text),
        });
        library_data->UTF8String_disown(path_text);
    }
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_save_buffer: @c arena → the monolith, as an @c UnsignedInteger8
 *  vector. */
EXTERN_C DLLEXPORT int DenseSaveBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    call_t call;
    usearch_arena_t arena = NULL;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        !(arena = call_arena(&call, MArgument_getInteger(arguments[receiver_arguments]))))
        return call_answer(library_data, &call, result);
    call.answer = answer_save_buffer;
    usearch_dense_save_buffer(&(usearch_dense_save_buffer_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .monolith = &call.monolith,
        .arena = arena,
    });
    return call_answer(library_data, &call, result);
}

/** Calls @c usearch_dense_describe: the frozen metadata, counters and dispatched capabilities. */
EXTERN_C DLLEXPORT int DenseDescribe(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t call;
    if (call_open(arguments, &call) == usearch_status_success_k) {
        call.answer = answer_describe;
        usearch_dense_describe(&(usearch_dense_describe_t) {
            .index = call.index,
            .executor = call.executor,
            .outcome = &call.outcome,
            .description = &call.description,
        });
    }
    return call_answer(library_data, &call, result);
}

/**
 *  @brief Calls @c usearch_dense_multiply: @c left → @c arena → @c wanted →
 *      @c expansionFactor → offsets, keys, products, and the pairs scored.
 *
 *  Row @c i of @p left spans pairs @c offsets[[i+1]]+1 through @c offsets[[i+2]], in traversal
 *  order; keys are @b [pairs,16] byte cells and products one real per pair. A zero @p wanted keeps
 *  every pair the walk scored.
 */
EXTERN_C DLLEXPORT int DenseMultiply(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t call;
    size_t expansion = 0;
    usearch_rows_view_t left;
    size_t wanted = 0;
    usearch_arena_t arena = NULL;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_size(&call, arguments[receiver_arguments + 4], &expansion, "expansion factor must not be negative") !=
            usearch_status_success_k ||
        call_rows(library_data, &call, arguments[receiver_arguments], arguments[receiver_arguments + 1], &left,
                  &call.count) != usearch_status_success_k ||
        !(arena = call_arena(&call, MArgument_getInteger(arguments[receiver_arguments + 2]))) ||
        call_size(&call, arguments[receiver_arguments + 3], &wanted, "wanted must not be negative") !=
            usearch_status_success_k)
        return call_answer(library_data, &call, result);
    call.answer = answer_multiply;
    usearch_dense_multiply(&(usearch_dense_multiply_t) {
        .index = call.index,
        .executor = call.executor,
        .outcome = &call.outcome,
        .count = call.count,
        .left = left,
        .products = &call.products,
        .written = &call.written,
        .required = &call.required,
        .arena = arena,
        .wanted = wanted,
        .expansion_factor = expansion,
    });
    return call_answer(library_data, &call, result);
}

typedef struct value_row_t {
    usearch_key_t key;
    size_t row;
} value_row_t;

static int value_row_compare(void const* left, void const* right) {
    return memcmp(&((value_row_t const*)left)->key, &((value_row_t const*)right)->key, sizeof(usearch_key_t));
}

static nk_dtype_t value_dtype(usearch_scalar_t scalar) {
    switch (scalar) {
    case usearch_scalar_i4x2_k: return nk_i4_k;
    case usearch_scalar_u4x2_k: return nk_u4_k;
    case usearch_scalar_u1x8_k: return nk_u1_k;
    default: {
        char const* const name = usearch_scalar_name(scalar);
        return nk_dtype_named(name, strlen(name));
    }
    }
}

/** Prepares sorted, quantized value rows once; subsequent attention calls borrow them. */
EXTERN_C DLLEXPORT int DenseValues(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    usearch_key_t const* keys = NULL;
    usearch_rows_view_t values;
    size_t keys_count = 0, count = 0;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_keys(library_data, &call, MArgument_getMNumericArray(arguments[2]), &keys, &keys_count) !=
            usearch_status_success_k ||
        call_rows(library_data, &call, arguments[3], arguments[4], &values, &count) != usearch_status_success_k)
        return call_answer(library_data, &call, result);
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = call.index,
                                                        .executor = call.executor,
                                                        .outcome = &call.outcome,
                                                        .description = &call.description});
    if (call.outcome.status != usearch_status_success_k) return call_answer(library_data, &call, result);
    if (!count || count != keys_count || values.columns != call.description.metadata.dimensions)
        return answer_failure(library_data, usearch_status_shape_mismatch_k, "values need one full-width row per key",
                              result);
    nk_dtype_t const dtype = value_dtype(call.description.metadata.quantization);
    size_t const row_bytes = (values.columns * nk_dtype_bits(dtype) + 7) / 8;
    if (!row_bytes || count > SIZE_MAX / sizeof(value_row_t) || count > (size_t)PTRDIFF_MAX / row_bytes)
        return answer_failure(library_data, usearch_status_shape_mismatch_k,
                              "value storage is too large or unsupported", result);
    value_row_t* const ordered = (value_row_t*)malloc(count * sizeof(value_row_t));
    if (!ordered) return LIBRARY_MEMORY_ERROR;
    for (size_t row = 0; row != count; ++row) ordered[row] = (value_row_t) {keys[row], row};
    qsort(ordered, count, sizeof(value_row_t), value_row_compare);
    for (size_t row = 1; row != count; ++row)
        if (!memcmp(&ordered[row - 1].key, &ordered[row].key, sizeof(usearch_key_t))) {
            free(ordered);
            return answer_failure(library_data, usearch_status_invalid_argument_k, "value keys must be unique", result);
        }
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    MNumericArray stored_keys = NULL, stored_rows = NULL;
    mint const key_shape[2] = {(mint)count, 16}, value_shape[2] = {(mint)count, (mint)row_bytes};
    if (arrays->MNumericArray_new(MNumericArray_Type_UBit8, 2, key_shape, &stored_keys) != LIBRARY_NO_ERROR ||
        arrays->MNumericArray_new(MNumericArray_Type_UBit8, 2, value_shape, &stored_rows) != LIBRARY_NO_ERROR) {
        if (stored_keys) arrays->MNumericArray_free(stored_keys);
        if (stored_rows) arrays->MNumericArray_free(stored_rows);
        free(ordered);
        return LIBRARY_MEMORY_ERROR;
    }
    usearch_key_t* const key_cells = (usearch_key_t*)arrays->MNumericArray_getData(stored_keys);
    uint8_t* const value_cells = (uint8_t*)arrays->MNumericArray_getData(stored_rows);
    nk_status_t status = nk_success_k;
    for (size_t row = 0; row != count && status == nk_success_k; ++row) {
        key_cells[row] = ordered[row].key;
        status = nk_cast_serial((uint8_t const*)values.data + ordered[row].row * values.stride,
                                values.scalar == usearch_scalar_f32_k ? nk_f32_k : nk_f64_k,
                                value_cells + row * row_bytes, dtype, values.columns, NULL);
    }
    free(ordered);
    if (status != nk_success_k) {
        arrays->MNumericArray_free(stored_keys);
        arrays->MNumericArray_free(stored_rows);
        return answer_failure(library_data, usearch_status_unsupported_k, "value quantization is unsupported", result);
    }
    int const answered = answer_nothing(library_data, result);
    if (answered != LIBRARY_NO_ERROR) {
        arrays->MNumericArray_free(stored_keys);
        arrays->MNumericArray_free(stored_rows);
        return answered;
    }
    DataStore const store = MArgument_getDataStore(result);
    library_data->ioLibraryFunctions->DataStore_addNamedMNumericArray(store, "ValueKeys", stored_keys);
    library_data->ioLibraryFunctions->DataStore_addNamedMNumericArray(store, "ValueRows", stored_rows);
    store_size(library_data, store, "Dimensions", values.columns);
    library_data->ioLibraryFunctions->DataStore_addNamedString(
        store, "Quantization", (char*)usearch_scalar_name(call.description.metadata.quantization));
    return LIBRARY_NO_ERROR;
}

typedef struct value_lookup_t {
    usearch_key_t const* keys;
    uint8_t const* values;
    size_t count, stride;
} value_lookup_t;

static void value_lookup(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar, size_t columns,
                         void const** rows) {
    value_lookup_t const* const lookup = (value_lookup_t const*)state;
    for (size_t row = 0; row != count; ++row) {
        size_t first = 0, last = lookup->count;
        while (first != last) {
            size_t const middle = first + (last - first) / 2;
            if (memcmp(&lookup->keys[middle], &keys[row], sizeof(usearch_key_t)) < 0) first = middle + 1;
            else last = middle;
        }
        rows[row] = first != lookup->count && !memcmp(&lookup->keys[first], &keys[row], sizeof(usearch_key_t))
                        ? lookup->values + first * lookup->stride
                        : NULL;
    }
}

static int answer_attend(WolframLibraryData library_data, call_t* call, DataStore store) {
    int const answered = answer_get(library_data, call, store);
    if (answered != LIBRARY_NO_ERROR) return answered;
    mint const shape[2] = {(mint)call->count, 2};
    MTensor tensor;
    if (library_data->MTensor_new(MType_Real, 2, shape, &tensor) != LIBRARY_NO_ERROR) return LIBRARY_MEMORY_ERROR;
    double* const cells = library_data->MTensor_getRealData(tensor);
    for (size_t row = 0; row != call->count; ++row) {
        cells[row * 2] = call->carry[row].maximum_score;
        cells[row * 2 + 1] = call->carry[row].partition;
    }
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, "Carry", tensor);
    return LIBRARY_NO_ERROR;
}

/** Calls the folded attention verb with an immutable, host-readable value cache. */
EXTERN_C DLLEXPORT int DenseAttend(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t call;
    usearch_rows_view_t queries;
    value_lookup_t lookup = {0};
    usearch_arena_t arena = NULL;
    size_t heads = 0, expansion = 0;
    if (call_open(arguments, &call) != usearch_status_success_k ||
        call_rows(library_data, &call, arguments[2], arguments[3], &queries, &call.count) != usearch_status_success_k ||
        call_keys(library_data, &call, MArgument_getMNumericArray(arguments[4]), &lookup.keys, &lookup.count) !=
            usearch_status_success_k ||
        call_size(&call, arguments[9], &heads, "head count must not be negative") != usearch_status_success_k ||
        call_size(&call, arguments[10], &expansion, "expansion factor must not be negative") !=
            usearch_status_success_k ||
        !(arena = call_arena(&call, MArgument_getInteger(arguments[8]))))
        return call_answer(library_data, &call, result);
    if (call.residency == usearch_residency_vram_k)
        return answer_failure(library_data, usearch_status_device_k,
                              "Wolfram value rows require a host attention lookup", result);
    char* scalar_name = MArgument_getUTF8String(arguments[6]);
    usearch_scalar_t const scalar = usearch_scalar_from_name(bytes_from_text(scalar_name));
    library_data->UTF8String_disown(scalar_name);
    usearch_dense_describe(&(usearch_dense_describe_t) {.index = call.index,
                                                        .executor = call.executor,
                                                        .outcome = &call.outcome,
                                                        .description = &call.description});
    if (call.outcome.status != usearch_status_success_k) return call_answer(library_data, &call, result);
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    MNumericArray const values = MArgument_getMNumericArray(arguments[5]);
    mint const* const shape = arrays->MNumericArray_getDimensions(values);
    lookup.stride = (queries.columns * nk_dtype_bits(value_dtype(scalar)) + 7) / 8;
    if (scalar != call.description.metadata.quantization ||
        MArgument_getInteger(arguments[7]) != (mint)queries.columns ||
        arrays->MNumericArray_getType(values) != MNumericArray_Type_UBit8 ||
        arrays->MNumericArray_getRank(values) != 2 || (size_t)shape[0] != lookup.count ||
        (size_t)shape[1] != lookup.stride)
        return answer_failure(library_data, usearch_status_shape_mismatch_k,
                              "value cache must match the index's quantization and query width", result);
    lookup.values = (uint8_t const*)arrays->MNumericArray_getData(values);
    if (call.count > SIZE_MAX / sizeof(usearch_dense_softmax_state_t)) return LIBRARY_MEMORY_ERROR;
    call.carry = (usearch_dense_softmax_state_t*)calloc(call.count, sizeof(usearch_dense_softmax_state_t));
    if (!call.carry) return LIBRARY_MEMORY_ERROR;
    call.written = call.count;
    call.answer = answer_attend;
    usearch_dense_attend(&(usearch_dense_attend_t) {.index = call.index,
                                                    .executor = call.executor,
                                                    .outcome = &call.outcome,
                                                    .count = call.count,
                                                    .queries = queries,
                                                    .head_count = heads,
                                                    .value_lookup = value_lookup,
                                                    .value_lookup_state = &lookup,
                                                    .carry = call.carry,
                                                    .outputs = &call.vectors,
                                                    .arena = arena,
                                                    .outputs_scalar = usearch_scalar_f64_k,
                                                    .expansion_factor = expansion,
                                                    .score_scale = (float)MArgument_getReal(arguments[11])});
    int const answered = call_answer(library_data, &call, result);
    free(call.carry);
    return answered;
}

#pragma endregion Dense Index Verbs
