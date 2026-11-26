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
 *  one takes its ID, and a zero ID runs the call on the kernel's own thread. The lists an executor
 *  constructor takes - memory domains, devices, streams, paths - cross as @c DataStore arguments,
 *  which unlike a tensor may be empty, and an empty one lets the engine derive.
 *
 *  @section wolfram_deferred Receivers & Deferred Calls
 *
 *  Every index verb leads with three IDs: the index, the executor, and a @c DeferredCall. A zero
 *  @c DeferredCall blocks, so the outcome is final and the outputs written once the verb returns.
 *  A fresh one submits the call with @c usearch_completion_deferred_k and parks it in that
 *  expression, which alone polls it through @c DeferredStatus and lands its outputs through
 *  @c DeferredWait; releasing it unwaited waits for the outcome first. A parked call owns what it
 *  passes the library: its tensors and arrays cross as @c Shared and are disowned once the call is
 *  final, and its strings and lists are its own copies.
 *
 *  @section wolfram_plans Plans
 *
 *  A refine or reshard plan is a @c Plan expression boxing the engine's handle and the phase the
 *  last call naming it stopped at. A call copies the box in and writes it back once final, so a
 *  parked call never points into a box the kernel may release, and a finish empties the box before
 *  the call crosses, as the library frees the plan whatever it answers. A box released while its
 *  plan lives leaks the plan, as an unfinished plan does in C.
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
#include <math.h>   // `NAN`
#include <stdlib.h> // `malloc`, `realloc`, `free`
#include <string.h> // `memcpy`, `memset`, `strlen`, `strcmp`

#include <stdint.h> // `int32_t`, `intptr_t`, `uint8_t`

#include <WolframIOLibraryFunctions.h>
#include <WolframLibrary.h>
#include <WolframNumericArrayLibrary.h>

#include <numkong/cast.h>

#include "usearch/dense_index.h"
#include "usearch/geospatial_index.h"
#include "usearch/sequence_index.h"
#include "usearch/shapes_index.h"
#include "usearch/types.h"

_Static_assert(sizeof(size_t) == sizeof(mint), "counts land in integer tensors cell for cell");

#pragma region Vocabulary

/** A viewed run of bytes over a null-terminated Wolfram string. */
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
 *  @brief Copies a @c DataStore of strings as paths into one fresh block, the views first and their
 *      bytes after them, which the caller frees at once.
 *
 *  The copy outlives the store, so a deferred call can pass the paths past the verb's return.
 *  @return @c usearch_status_invalid_argument_k for an entry of another type.
 */
static usearch_status_t paths_from_store(WolframLibraryData library_data, MArgument argument,
                                         usearch_bytes_view_t** paths, size_t* count) {
    DataStore const store = MArgument_getDataStore(argument);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    *count = (size_t)io->DataStore_getLength(store);
    size_t text_bytes = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); node; node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        if (io->DataStoreNode_getDataType(node) != MType_UTF8String || io->DataStoreNode_getData(node, &cell) != 0)
            return *paths = NULL, usearch_status_invalid_argument_k;
        text_bytes += strlen(MArgument_getUTF8String(cell));
    }
    size_t const views_bytes = sizeof(usearch_bytes_view_t) * (*count ? *count : 1);
    *paths = (usearch_bytes_view_t*)malloc(views_bytes + text_bytes);
    if (!*paths) return usearch_status_out_of_memory_k;
    uint8_t* text = (uint8_t*)*paths + views_bytes;
    size_t position = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); node; node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        io->DataStoreNode_getData(node, &cell);
        usearch_bytes_view_t const viewed = bytes_from_text(MArgument_getUTF8String(cell));
        if (viewed.length) memcpy(text, viewed.data, viewed.length);
        (*paths)[position].data = text;
        (*paths)[position++].length = viewed.length;
        text += viewed.length;
    }
    return usearch_status_success_k;
}

/**
 *  @brief Reads a @c DataStore of non-negative integers as sizes - window widths - into a fresh
 *      array, which the caller frees.
 *  @return @c usearch_status_invalid_argument_k for an entry of another type or sign.
 */
static usearch_status_t sizes_from_store(WolframLibraryData library_data, MArgument argument, size_t** sizes,
                                         size_t* count) {
    DataStore const store = MArgument_getDataStore(argument);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    *count = (size_t)io->DataStore_getLength(store);
    *sizes = (size_t*)malloc(sizeof(size_t) * (*count ? *count : 1));
    if (!*sizes) return usearch_status_out_of_memory_k;
    size_t position = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); node; node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        if (io->DataStoreNode_getDataType(node) != MType_Integer || io->DataStoreNode_getData(node, &cell) != 0 ||
            MArgument_getInteger(cell) < 0) {
            free(*sizes);
            *sizes = NULL;
            return usearch_status_invalid_argument_k;
        }
        (*sizes)[position++] = (size_t)MArgument_getInteger(cell);
    }
    return usearch_status_success_k;
}

/** Reads a byte array as a viewed buffer - the monolith the buffer verbs take. */
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
 *  @param[out] keys The keys, viewing the array's cells.
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
 *  @brief Views a real tensor as a run of rows: rank one is a single row, rank two a batch, and a
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

/** Reads a metric and a quantization by their canonical names, answering the status of the first
 *  name the library refuses. */
static usearch_status_t geometry_from_arguments(WolframLibraryData library_data, MArgument const* arguments,
                                                usearch_dense_metric_t* metric, usearch_scalar_t* quantization) {
    char* metric_text = MArgument_getUTF8String(arguments[0]);
    char* quantization_text = MArgument_getUTF8String(arguments[1]);
    usearch_status_t named = usearch_dense_metric_from_name(bytes_from_text(metric_text), metric);
    if (named == usearch_status_success_k)
        named = usearch_scalar_from_name(bytes_from_text(quantization_text), quantization);
    library_data->UTF8String_disown(metric_text);
    library_data->UTF8String_disown(quantization_text);
    return named;
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

static handles_t indexes, sequences, shapes, geospatials, executors, arenas, plans, deferreds;

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
static usearch_executor_t executor_find(mint id) { return (usearch_executor_t)handles_find(&executors, id); }

/** Resolves the optional executor a call fans out over; zero is the kernel's thread. */
static int executor_from_argument(MArgument argument, usearch_executor_t* executor) {
    mint const executor_id = MArgument_getInteger(argument);
    *executor = executor_find(executor_id);
    return !executor_id || *executor;
}

/** The box behind a @c Plan expression: one refine or reshard plan of any family. */
typedef struct plan_t {

    /** The engine's plan: @c NULL until a call opens it, and again once a finish consumes it. */
    union {
        usearch_dense_refine_plan_t dense_refine;
        usearch_dense_reshard_plan_t dense_reshard;
        usearch_sequence_refine_plan_t sequence_refine;
        usearch_sequence_reshard_plan_t sequence_reshard;
        usearch_shapes_refine_plan_t shapes_refine;
        usearch_shapes_reshard_plan_t shapes_reshard;
        usearch_geospatial_refine_plan_t geospatial_refine;
        usearch_geospatial_reshard_plan_t geospatial_reshard;
    } handle;

    /** Where the last call naming the plan stopped. */
    usearch_phase_progress_t progress;
} plan_t;

#pragma endregion Managed Expressions

#pragma region Calls

typedef struct call_t call_t;

/** Lands a final call's outputs in @p store, named after the C fields they came from. */
typedef int (*answer_t)(WolframLibraryData library_data, call_t* call, DataStore store);

/** The value rows @c DenseAttend reads, sorted by key, and the shape the call states for them. */
typedef struct value_lookup_t {
    usearch_key_t const* keys;
    uint8_t const* values;
    size_t count, stride, columns;
    usearch_scalar_t scalar;
} value_lookup_t;

/** The most arrays and lists one verb passes the library. */
#define call_holds_k 4

/** One index verb: its receiver, its outcome, what it passes the library and the outputs it lands.
 *  A blocking call lives until the verb returns, and a deferred one until its @c DeferredCall is
 *  waited on or released. */
struct call_t {

    /** The index the verb addresses, of the family its verb names. */
    void* index;

    /** The executor the verb fans out over; @c NULL runs it on the kernel's thread. */
    usearch_executor_t executor;

    /** Blocking, or deferred when the call parks in a @c DeferredCall. */
    usearch_completion_t completion;

    /** The outcome the library publishes. */
    usearch_outcome_t outcome;

    /** Lands the outputs on success; @c NULL for verbs that answer nothing. */
    answer_t answer;

    /** The @c DeferredCall slot a deferred call parks in; @c NULL for a blocking one. */
    void** parked;

    /** The @c Plan the call names, zero for none, and the copy of its box the library writes. */
    mint plan_id;
    plan_t plan;

    /** Tensors and arrays passed @c Shared, disowned once the call is final. */
    MTensor tensors[call_holds_k];
    MNumericArray arrays[call_holds_k];
    size_t tensors_count, arrays_count;

    /** The one string argument the call passes, disowned once the call is final. */
    char* text;

    /** Lists and copies the call passes, freed once the call is final. */
    void* owned[call_holds_k];
    size_t owned_count;

    /** The value rows of @c attend, and its online-softmax states. */
    value_lookup_t lookup;
    usearch_dense_softmax_state_t* carry;

    /** Queries of @c search and @c within, or left rows of @c multiply, one per output row. */
    size_t count;

    /** Rows or entries @c get wrote, or pairs @c multiply, @c maxsim and @c within wrote. */
    size_t written;

    /** Pairs @c multiply or keys @c maxsim scored before @c wanted cut any. */
    size_t required;

    /** Entries under the key of @c count, with the extent of their payloads. */
    union {
        size_t entries;
        usearch_sequence_key_stats_t sequence;
        usearch_shapes_key_stats_t shapes;
        usearch_geospatial_key_stats_t geospatial;
    } tally;

    /** The report of @c refine, @c reshard and their finishes. */
    usearch_refine_report_t report;

    /** Matches of @c search, inside the arena. */
    usearch_matches_view_t matches;

    /** Rows of @c get or outputs of @c attend, inside the arena. */
    usearch_rows_view_t vectors;

    /** Sequences or shapes of @c get, inside the arena. */
    usearch_tape_view_t tape;

    /** Observations of @c get, inside the arena. */
    usearch_geospatial_observation_t const* observations;

    /** Pairs of @c multiply, keys of @c maxsim or matches of @c within, inside the arena. */
    usearch_matches_tape_view_t products;

    /** The @c save_buffer monolith, inside the arena. */
    usearch_bytes_view_t monolith;

    /** The report of @c describe. */
    union {
        usearch_dense_index_description_t dense;
        usearch_sequence_index_description_t sequence;
        usearch_shapes_index_description_t shapes;
        usearch_geospatial_index_description_t geospatial;
    } description;

    /** The audit of @c health. */
    usearch_reachability_t reachability;
};

/** The call a verb gets when the kernel cannot allocate one, refused before anything crosses. */
static call_t unallocated_call = {
    .outcome = {usearch_status_out_of_memory_k, "the kernel could not allocate the call"}};

/** Records a failure the binding caught before the call crossed, as the engine records its own. */
static usearch_status_t call_refuse(call_t* call, usearch_status_t status, char const* message) {
    call->outcome.status = status;
    call->outcome.message = message;
    return status;
}

/** Whether nothing refused the call yet, so it may read further arguments or cross. */
static int call_ready(call_t const* call) { return call->outcome.status == usearch_status_success_k; }

/** Keeps a @c Shared tensor until the call is final. */
static void call_hold_tensor(WolframLibraryData library_data, call_t* call, MTensor tensor) {
    if (call == &unallocated_call || call->tensors_count == call_holds_k) library_data->MTensor_disown(tensor);
    else call->tensors[call->tensors_count++] = tensor;
}

/** Keeps a @c Shared numeric array until the call is final. */
static void call_hold_array(WolframLibraryData library_data, call_t* call, MNumericArray array) {
    if (call == &unallocated_call || call->arrays_count == call_holds_k)
        library_data->numericarrayLibraryFunctions->MNumericArray_disown(array);
    else call->arrays[call->arrays_count++] = array;
}

/** Keeps a list the call passes until the call is final, refusing the call when it holds no
 *  more. */
static void* call_own(call_t* call, void* memory) {
    if (call == &unallocated_call || call->owned_count == call_holds_k) return free(memory), NULL;
    return call->owned[call->owned_count++] = memory;
}

/** Releases what a final call passed and the call itself. */
static void call_free(WolframLibraryData library_data, call_t* call) {
    if (call == &unallocated_call) return;
    for (size_t position = 0; position != call->tensors_count; ++position)
        library_data->MTensor_disown(call->tensors[position]);
    for (size_t position = 0; position != call->arrays_count; ++position)
        library_data->numericarrayLibraryFunctions->MNumericArray_disown(call->arrays[position]);
    for (size_t position = 0; position != call->owned_count; ++position) free(call->owned[position]);
    if (call->text) library_data->UTF8String_disown(call->text);
    free(call->carry);
    free(call);
}

/** Views a string argument for the call, as a byte view that lives as long as the call. */
static usearch_bytes_view_t call_text(WolframLibraryData library_data, call_t* call, MArgument argument) {
    char* const text = MArgument_getUTF8String(argument);
    if (call == &unallocated_call || call->text) {
        library_data->UTF8String_disown(text);
        return bytes_from_text(NULL);
    }
    call->text = text;
    return bytes_from_text(text);
}

/** Views packed or typed real rows, passed @c Shared, for the call. */
static usearch_status_t call_rows(WolframLibraryData library_data, call_t* call, MArgument argument,
                                  MArgument array_argument, usearch_rows_view_t* rows, size_t* count) {
    usearch_status_t status;
    if (MArgument_getBoolean(array_argument)) {
        MNumericArray const array = MArgument_getMNumericArray(argument);
        call_hold_array(library_data, call, array);
        if (!call_ready(call)) return call->outcome.status;
        status = rows_from_array(library_data, array, rows, count);
    }
    else {
        MTensor const tensor = MArgument_getMTensor(argument);
        call_hold_tensor(library_data, call, tensor);
        if (!call_ready(call)) return call->outcome.status;
        status = rows_from_tensor(library_data, tensor, rows, count);
    }
    if (status != usearch_status_success_k)
        return call_refuse(call, status,
                           "rows are a real vector or matrix, with Real32 or Real64 NumericArray storage");
    return usearch_status_success_k;
}

/** Views the keys of a byte array, passed @c Shared, for the call, refusing any other shape. */
static usearch_status_t call_keys(WolframLibraryData library_data, call_t* call, MArgument argument,
                                  usearch_key_t const** keys, size_t* count) {
    MNumericArray const array = MArgument_getMNumericArray(argument);
    call_hold_array(library_data, call, array);
    if (!call_ready(call)) return call->outcome.status;
    if (keys_from_array(library_data, array, keys, count) != usearch_status_success_k)
        return call_refuse(call, usearch_status_malformed_k, "keys are UnsignedInteger8 arrays, sixteen cells apiece");
    return usearch_status_success_k;
}

/** Reads the one key of a byte array, refusing a batch. */
static usearch_status_t call_key(WolframLibraryData library_data, call_t* call, MArgument argument,
                                 usearch_key_t* key) {
    if (!call_ready(call)) return call->outcome.status;
    usearch_key_t const* keys = NULL;
    size_t count = 0;
    if (keys_from_array(library_data, MArgument_getMNumericArray(argument), &keys, &count) !=
            usearch_status_success_k ||
        count != 1)
        return call_refuse(call, usearch_status_malformed_k, "a key is an UnsignedInteger8 array of sixteen cells");
    *key = *keys;
    return usearch_status_success_k;
}

/** Views the @c Shared offsets of a tape for the call: one more non-negative integer than
 *  entries, which @p count receives. */
static usearch_status_t call_offsets(WolframLibraryData library_data, call_t* call, MArgument argument,
                                     usearch_tape_view_t* tape, size_t* count) {
    MTensor const offsets = MArgument_getMTensor(argument);
    call_hold_tensor(library_data, call, offsets);
    if (!call_ready(call)) return call->outcome.status;
    if (library_data->MTensor_getType(offsets) != MType_Integer || library_data->MTensor_getRank(offsets) != 1 ||
        library_data->MTensor_getFlattenedLength(offsets) < 1)
        return call_refuse(call, usearch_status_malformed_k, "tape offsets are one more integer than entries");
    _Static_assert(sizeof(mint) == sizeof(uint64_t), "integer tensors read as 64-bit offsets");
    tape->layout = usearch_tape_u64_k;
    tape->offsets.u64 = (uint64_t const*)library_data->MTensor_getIntegerData(offsets);
    *count = (size_t)library_data->MTensor_getFlattenedLength(offsets) - 1;
    return usearch_status_success_k;
}

/** Views the @c Shared bytes of a sequence tape, every sequence end to end. */
static usearch_status_t call_bytes_tape(WolframLibraryData library_data, call_t* call, MArgument argument,
                                        usearch_tape_view_t* tape) {
    MNumericArray const bytes = MArgument_getMNumericArray(argument);
    call_hold_array(library_data, call, bytes);
    if (!call_ready(call)) return call->outcome.status;
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    if (arrays->MNumericArray_getType(bytes) != MNumericArray_Type_UBit8 || arrays->MNumericArray_getRank(bytes) != 1)
        return call_refuse(call, usearch_status_malformed_k, "sequence bytes are one UnsignedInteger8 vector");
    tape->data = arrays->MNumericArray_getData(bytes);
    tape->scalar = usearch_scalar_u8_k;
    return usearch_status_success_k;
}

/** Views the @c Shared points of a shapes tape: a real matrix of three columns, @c x, @c y and
 *  @c z, every shape's points end to end. */
static usearch_status_t call_points_tape(WolframLibraryData library_data, call_t* call, MArgument argument,
                                         usearch_tape_view_t* tape) {
    MTensor const points = MArgument_getMTensor(argument);
    call_hold_tensor(library_data, call, points);
    if (!call_ready(call)) return call->outcome.status;
    if (library_data->MTensor_getType(points) != MType_Real || library_data->MTensor_getRank(points) != 2 ||
        library_data->MTensor_getDimensions(points)[1] != 3)
        return call_refuse(call, usearch_status_shape_mismatch_k, "points are a real matrix of three columns");
    tape->data = library_data->MTensor_getRealData(points);
    tape->scalar = usearch_scalar_f64_k;
    return usearch_status_success_k;
}

/**
 *  @brief Copies observations into the call: a real matrix of eleven columns beside an integer one
 *      marking each field recorded, so each unrecorded field reads as NaN.
 *
 *  A Wolfram tensor holds no NaN, so the mask holds what a device left out, and the copy is the
 *  call's own until it is final.
 */
static usearch_status_t call_observations(WolframLibraryData library_data, call_t* call, MArgument values_argument,
                                          MArgument recorded_argument,
                                          usearch_geospatial_observation_t const** observations, size_t* count) {
    if (!call_ready(call)) return call->outcome.status;
    MTensor const values = MArgument_getMTensor(values_argument), recorded = MArgument_getMTensor(recorded_argument);
    size_t const fields = sizeof(usearch_geospatial_observation_t) / sizeof(double);
    if (library_data->MTensor_getType(values) != MType_Real || library_data->MTensor_getRank(values) != 2 ||
        (size_t)library_data->MTensor_getDimensions(values)[1] != fields ||
        library_data->MTensor_getType(recorded) != MType_Integer ||
        library_data->MTensor_getFlattenedLength(recorded) != library_data->MTensor_getFlattenedLength(values))
        return call_refuse(call, usearch_status_shape_mismatch_k,
                           "observations are a real matrix of eleven columns beside its recorded mask");
    *count = (size_t)library_data->MTensor_getDimensions(values)[0];
    double* const copy = (double*)call_own(call, malloc(sizeof(usearch_geospatial_observation_t) * (*count + 1)));
    if (!copy) return call_refuse(call, usearch_status_out_of_memory_k, "the kernel could not copy the observations");
    double const* const cells = library_data->MTensor_getRealData(values);
    mint const* const marks = library_data->MTensor_getIntegerData(recorded);
    for (size_t cell = 0; cell != *count * fields; ++cell) copy[cell] = marks[cell] ? cells[cell] : NAN;
    *observations = (usearch_geospatial_observation_t const*)copy;
    return usearch_status_success_k;
}

/** The arena an ID names for the call, refusing a released one and one holding memory the kernel
 *  cannot copy out of. */
static usearch_arena_t call_arena(call_t* call, MArgument argument) {
    if (!call_ready(call)) return NULL;
    usearch_arena_t const arena = (usearch_arena_t)handles_find(&arenas, MArgument_getInteger(argument));
    if (!arena) return call_refuse(call, usearch_status_uninitialized_k, "the Arena was released"), NULL;
    usearch_arena_description_t description = {0};
    usearch_outcome_t described = {usearch_status_success_k, NULL};
    usearch_arena_describe(
        &(usearch_arena_describe_t) {.arena = arena, .outcome = &described, .description = &description});
    if (described.status != usearch_status_success_k)
        return call_refuse(call, described.status, described.message), NULL;
    if (description.host_access != usearch_host_access_readable_k)
        return call_refuse(call, usearch_status_device_k, "the outputs are memory the host may not read"), NULL;
    return arena;
}

/** Reads a count the call takes, refusing a negative number rather than wrapping it. */
static usearch_status_t call_size(call_t* call, MArgument argument, size_t* size, char const* message) {
    if (!call_ready(call)) return call->outcome.status;
    mint const requested = MArgument_getInteger(argument);
    if (requested < 0) return call_refuse(call, usearch_status_invalid_argument_k, message);
    *size = (size_t)requested;
    return usearch_status_success_k;
}

/** Reads a traversal by name, refusing one the library does not spell. */
static usearch_status_t call_traversal(WolframLibraryData library_data, call_t* call, MArgument argument,
                                       usearch_traversal_t* traversal) {
    if (!call_ready(call)) return call->outcome.status;
    char* traversal_text = MArgument_getUTF8String(argument);
    usearch_status_t const parsed = usearch_traversal_from_name(bytes_from_text(traversal_text), traversal);
    library_data->UTF8String_disown(traversal_text);
    return parsed == usearch_status_success_k ? parsed
                                              : call_refuse(call, parsed, "the traversal is neither walk nor scan");
}

/** Copies the box of the @c Plan an ID names into the call, zero naming none; the call writes the
 *  box back once final. */
static usearch_status_t call_plan(call_t* call, MArgument argument) {
    if (!call_ready(call)) return call->outcome.status;
    mint const plan_id = MArgument_getInteger(argument);
    if (!plan_id) return usearch_status_success_k;
    plan_t const* const box = (plan_t const*)handles_find(&plans, plan_id);
    if (!box) return call_refuse(call, usearch_status_uninitialized_k, "the Plan was released");
    call->plan_id = plan_id;
    call->plan = *box;
    return usearch_status_success_k;
}

/** Moves the plan of a @c Plan out of its box into a finish, which frees it whatever it answers. */
static usearch_status_t call_take_plan(call_t* call, MArgument argument) {
    if (!call_ready(call)) return call->outcome.status;
    plan_t* const box = (plan_t*)handles_find(&plans, MArgument_getInteger(argument));
    if (!box) return call_refuse(call, usearch_status_uninitialized_k, "the Plan was released");
    call->plan = *box;
    memset(&box->handle, 0, sizeof(box->handle));
    return usearch_status_success_k;
}

/** Collects every index ID of a @c DataStore into a list of @p family handles the call passes. */
static usearch_status_t call_handles(WolframLibraryData library_data, call_t* call, MArgument argument,
                                     handles_t const* family, void const** handles, size_t* count) {
    if (!call_ready(call)) return call->outcome.status;
    DataStore const store = MArgument_getDataStore(argument);
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    *count = (size_t)io->DataStore_getLength(store);
    void** const listed = (void**)call_own(call, malloc(sizeof(void*) * (*count ? *count : 1)));
    if (!listed) return call_refuse(call, usearch_status_out_of_memory_k, "the kernel could not copy the list");
    size_t position = 0;
    for (DataStoreNode node = io->DataStore_getFirstNode(store); node; node = io->DataStoreNode_getNextNode(node)) {
        MArgument cell;
        void* const handle = io->DataStoreNode_getDataType(node) == MType_Integer &&
                                     io->DataStoreNode_getData(node, &cell) == 0
                                 ? handles_find(family, MArgument_getInteger(cell))
                                 : NULL;
        if (!handle) return call_refuse(call, usearch_status_uninitialized_k, "a listed index was released");
        listed[position++] = handle;
    }
    *handles = (void const*)listed;
    return usearch_status_success_k;
}

/** The three leading IDs every index verb takes: @c index → @c executor → @c deferred. */
static size_t const receiver_arguments = 3;

/**
 *  @brief Opens a call on the receiver its three leading IDs name: the index of @p family,
 *      the executor, a zero meaning the kernel's thread, and a fresh @c DeferredCall to park
 *      in, a zero blocking.
 *
 *  Never answers @c NULL: a refused call still collects what later arguments pass, so the verb
 *  releases them in one place.
 */
static call_t* call_open(MArgument const* arguments, handles_t const* family, char const* released) {
    call_t* const call = (call_t*)calloc(1, sizeof(call_t));
    if (!call) return &unallocated_call;
    call->index = handles_find(family, MArgument_getInteger(arguments[0]));
    mint const executor_id = MArgument_getInteger(arguments[1]);
    call->executor = executor_find(executor_id);
    mint const deferred_id = MArgument_getInteger(arguments[2]);
    call->parked = deferred_id ? handles_fresh(&deferreds, deferred_id) : NULL;
    call->completion = deferred_id ? usearch_completion_deferred_k : usearch_completion_blocking_k;
    if (deferred_id && !call->parked)
        call_refuse(call, usearch_status_invalid_argument_k, "the DeferredCall is not fresh");
    else if (!call->index) call_refuse(call, usearch_status_uninitialized_k, released);
    else if (executor_id && !call->executor)
        call_refuse(call, usearch_status_uninitialized_k, "the Executor was released or never constructed");
    return call;
}

/** Answers a final call: its outputs, copied out of any arena, on success, its status and detail
 *  otherwise; then writes its plan back and releases it. */
static int call_finish(WolframLibraryData library_data, call_t* call, MArgument result) {
    plan_t* const box = call->plan_id ? (plan_t*)handles_find(&plans, call->plan_id) : NULL;
    if (box) *box = call->plan;
    int answered;
    if (call->outcome.status != usearch_status_success_k)
        answered = answer_failure(library_data, call->outcome.status, call->outcome.message, result);
    else {
        WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
        DataStore const store = io->createDataStore();
        answered = store ? LIBRARY_NO_ERROR : LIBRARY_MEMORY_ERROR;
        if (store && call->answer) answered = call->answer(library_data, call, store);
        if (answered == LIBRARY_NO_ERROR) MArgument_setDataStore(result, store);
        else if (store) io->deleteDataStore(store);
    }
    call_free(library_data, call);
    return answered;
}

/** Answers a returned verb: a blocking call as @c call_finish does, and a deferred one by parking
 *  it in its @c DeferredCall, answering nothing until it is waited on. */
static int call_answer(WolframLibraryData library_data, call_t* call, MArgument result) {
    if (!call->parked) return call_finish(library_data, call, result);
    *call->parked = call;
    return answer_nothing(library_data, result);
}

#pragma endregion Calls

#pragma region Library Lifecycle

/** Frees a released dense index. */
static void manage_index(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) handles_admit(&indexes, id, empty_slot);
    else usearch_dense_free((usearch_dense_index_t)handles_release(&indexes, id));
}

/** Frees a released sequence index. */
static void manage_sequence(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) handles_admit(&sequences, id, empty_slot);
    else usearch_sequence_free((usearch_sequence_index_t)handles_release(&sequences, id));
}

/** Frees a released shapes index. */
static void manage_shapes(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) handles_admit(&shapes, id, empty_slot);
    else usearch_shapes_free((usearch_shapes_index_t)handles_release(&shapes, id));
}

/** Frees a released geospatial index. */
static void manage_geospatial(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) handles_admit(&geospatials, id, empty_slot);
    else usearch_geospatial_free((usearch_geospatial_index_t)handles_release(&geospatials, id));
}

/** Frees a released executor. The engine refuses while an arena made with it lives, which leaks the
 *  executor until the process ends; the kernel may release it before its arenas. */
static void manage_executor(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) {
        handles_admit(&executors, id, empty_slot);
        return;
    }
    [[maybe_unused]] usearch_status_t const refused = usearch_executor_free(
        (usearch_executor_t)handles_release(&executors, id));
}

/** Frees a released arena and every output in it. The free refuses only while an Arrow export of
 *  the outputs is open, and the kernel copies every output instead, so its status is dropped. */
static void manage_arena(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) handles_admit(&arenas, id, empty_slot);
    else usearch_arena_free((usearch_arena_t)handles_release(&arenas, id));
}

/** Boxes a new plan, empty until a call opens it; a released box leaks a plan still live, as an
 *  unfinished plan does in C. */
static void manage_plan(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) {
        plan_t* const box = (plan_t*)calloc(1, sizeof(plan_t));
        if (box && !handles_admit(&plans, id, box)) free(box);
    }
    else free(handles_release(&plans, id));
}

/** Waits for a released deferred call that was never waited on, then releases what it passed. */
static void manage_deferred(WolframLibraryData library_data, mbool mode, mint id) {
    if (!mode) {
        handles_admit(&deferreds, id, empty_slot);
        return;
    }
    call_t* const call = (call_t*)handles_release(&deferreds, id);
    if (!call) return;
    usearch_outcome_wait(&call->outcome);
    call_free(library_data, call);
}

/** The managed expression kinds, in the order the kernel registers them. */
static struct {
    char const* name;
    void (*manage)(WolframLibraryData, mbool, mint);
} const managers[] = {
    {"DenseIndex", manage_index},   {"SequenceIndex", manage_sequence},
    {"ShapesIndex", manage_shapes}, {"GeospatialIndex", manage_geospatial},
    {"Executor", manage_executor},  {"Arena", manage_arena},
    {"Plan", manage_plan},          {"DeferredCall", manage_deferred},
};

EXTERN_C DLLEXPORT mint WolframLibrary_getVersion(void) { return WolframLibraryVersion; }

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData library_data) {
    for (size_t kind = 0; kind != sizeof(managers) / sizeof(managers[0]); ++kind)
        if (library_data->registerLibraryExpressionManager((char*)managers[kind].name, managers[kind].manage))
            return LIBRARY_FUNCTION_ERROR;
    return LIBRARY_NO_ERROR;
}

/** Releases whatever the kernel left alive: the deferred calls first, as they pass the rest, and
 *  the executors last, as the indexes and arenas may name them. */
EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData library_data) {
    for (size_t kind = 0; kind != sizeof(managers) / sizeof(managers[0]); ++kind)
        library_data->unregisterLibraryExpressionManager((char*)managers[kind].name);
    handles_t* const released[] = {&deferreds, &plans, &indexes, &sequences, &shapes, &geospatials, &arenas};
    void (*const manage[])(WolframLibraryData, mbool, mint) = {
        manage_deferred, manage_plan, manage_index, manage_sequence, manage_shapes, manage_geospatial, manage_arena};
    for (size_t kind = 0; kind != sizeof(released) / sizeof(released[0]); ++kind) {
        for (mint id = 0; id < released[kind]->capacity; ++id) manage[kind](library_data, 1, id);
        handles_free(released[kind]);
    }
    for (mint id = 0; id < executors.capacity; ++id) manage_executor(library_data, 1, id);
    handles_free(&executors);
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

/** Calls @c usearch_outcome_status on a parked call: its status name, @c deferred until it is
 *  final, and @c uninitialized once waited on. */
EXTERN_C DLLEXPORT int DeferredStatus(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = (call_t*)handles_find(&deferreds, MArgument_getInteger(arguments[0]));
    usearch_status_t const status = call ? usearch_outcome_status(&call->outcome) : usearch_status_uninitialized_k;
    MArgument_setUTF8String(result, (char*)usearch_status_name(status));
    return LIBRARY_NO_ERROR;
}

/** Calls @c usearch_outcome_wait on a parked call, then answers what the blocking verb would have,
 *  once: the call is released with its answer. */
EXTERN_C DLLEXPORT int DeferredWait(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    void** const slot = handles_slot(&deferreds, MArgument_getInteger(arguments[0]));
    call_t* const call = slot && *slot != empty_slot ? (call_t*)*slot : NULL;
    if (!call)
        return answer_failure(library_data, usearch_status_uninitialized_k,
                              "the DeferredCall was waited on or never submitted", result);
    *slot = empty_slot;
    usearch_outcome_wait(&call->outcome);
    return call_finish(library_data, call, result);
}

/** Reads a plan's box: the @c Phase the last call naming it stopped at, as @c usearch_phase_name
 *  spells it, that phase's @c Ordinal, and its @c Completed and @c Total units. */
EXTERN_C DLLEXPORT int PlanProgress(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    plan_t const* const box = (plan_t const*)handles_find(&plans, MArgument_getInteger(arguments[0]));
    if (!box) return answer_failure(library_data, usearch_status_uninitialized_k, "the Plan was released", result);
    int const answered = answer_nothing(library_data, result);
    if (answered != LIBRARY_NO_ERROR) return answered;
    DataStore const store = MArgument_getDataStore(result);
    store_text(library_data, store, "Phase", usearch_phase_name(box->progress.phase));
    store_size(library_data, store, "Ordinal", box->progress.ordinal);
    store_size(library_data, store, "Completed", box->progress.completed);
    store_size(library_data, store, "Total", box->progress.total);
    return LIBRARY_NO_ERROR;
}

#pragma endregion Library Lifecycle

#pragma region Executor

/** Fills a fresh executor's slot with the handle, or reports why no executor was made. */
static int executor_adopt(WolframLibraryData library_data, void** slot, usearch_executor_t handle,
                          usearch_outcome_t const* outcome, MArgument result) {
    if (outcome->status == usearch_status_success_k) *slot = handle;
    else { [[maybe_unused]] usearch_status_t const refused = usearch_executor_free(handle); }
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
    return executor_adopt(library_data, slot, executor, &outcome, result);
}

/**
 *  @brief Calls @c usearch_executor_for_vram: @c executor → @c devices → @c streams →
 *      @c scratchLimit.
 *
 *  The streams are the caller's stream addresses, one per device: on CUDA and HIP the executor
 *  takes each and destroys it when freed, on Metal it retains each queue and releases its own
 *  reference, and a failed creation leaves them all the caller's. A store of another length than
 *  the devices is refused; an empty one uses the executor's own stream per device.
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
    if (streams_count && streams_count != devices_count) {
        free(devices);
        return answer_failure(library_data, usearch_status_invalid_argument_k, "streams pair one by one with devices",
                              result);
    }
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
    return executor_adopt(library_data, slot, executor, &outcome, result);
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
    return executor_adopt(library_data, slot, executor, &outcome, result);
}

/** Calls @c usearch_executor_drain: @c executor → @c mood, @c parked or @c spinning. */
EXTERN_C DLLEXPORT int ExecutorDrain(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    usearch_executor_t const executor = executor_find(MArgument_getInteger(arguments[0]));
    if (!executor)
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    char* mood_text = MArgument_getUTF8String(arguments[1]);
    usearch_mood_t mood = usearch_mood_parked_k;
    usearch_status_t const parsed = usearch_mood_from_name(bytes_from_text(mood_text), &mood);
    library_data->UTF8String_disown(mood_text);
    if (parsed != usearch_status_success_k)
        return answer_failure(library_data, parsed, "the mood is neither parked nor spinning", result);

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
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_arena_t arena = NULL;
    usearch_arena_init(&(usearch_arena_init_t) {.arena = &arena, .executor = executor, .outcome = &outcome});
    if (outcome.status == usearch_status_success_k) *slot = arena;
    else usearch_arena_free(arena);
    return answer_outcome(library_data, &outcome, result);
}

/** Calls @c usearch_arena_describe: @c arena → @c Residency, its canonical name, @c Device and
 *  @c VendorOrdinal, -1 for host pages, @c Vendor and @c HostAccess, "readable" or "refused". */
EXTERN_C DLLEXPORT int ArenaDescribe(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    usearch_arena_t const arena = (usearch_arena_t)handles_find(&arenas, MArgument_getInteger(arguments[0]));
    if (!arena) return answer_failure(library_data, usearch_status_uninitialized_k, "the Arena was released", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_arena_description_t description = {0};
    usearch_arena_describe(
        &(usearch_arena_describe_t) {.arena = arena, .outcome = &outcome, .description = &description});
    if (outcome.status != usearch_status_success_k) return answer_outcome(library_data, &outcome, result);
    int const answered = answer_nothing(library_data, result);
    if (answered != LIBRARY_NO_ERROR) return answered;
    DataStore const store = MArgument_getDataStore(result);
    store_text(library_data, store, "Residency", usearch_residency_name(description.residency));
    library_data->ioLibraryFunctions->DataStore_addNamedInteger(store, "Device", description.device);
    store_text(library_data, store, "Vendor", usearch_gpu_vendor_name(description.vendor));
    library_data->ioLibraryFunctions->DataStore_addNamedInteger(store, "VendorOrdinal", description.vendor_ordinal);
    store_text(library_data, store, "HostAccess", usearch_host_access_name(description.host_access));
    return LIBRARY_NO_ERROR;
}

#pragma endregion Executor

#pragma region Dense Index Lifecycle

/** Fills a fresh index's slot with a handle the library made, or reports why it made none; the
 *  caller has freed a handle of a failed call. */
static int index_adopt(WolframLibraryData library_data, void** slot, void* handle, usearch_outcome_t const* outcome,
                       MArgument result) {
    if (outcome->status != usearch_status_success_k) return answer_outcome(library_data, outcome, result);
    *slot = handle;
    return answer_nothing(library_data, result);
}

/** Fills a fresh dense index's slot, or frees what a failed call left and reports why. */
static int dense_adopt(WolframLibraryData library_data, void** slot, usearch_dense_index_t handle,
                       usearch_outcome_t const* outcome, MArgument result) {
    if (outcome->status != usearch_status_success_k) usearch_dense_free(handle);
    return index_adopt(library_data, slot, handle, outcome, result);
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
    usearch_dense_metric_t metric = usearch_dense_metric_unknown_k;
    usearch_scalar_t quantization = usearch_scalar_unknown_k;
    usearch_status_t const named = geometry_from_arguments(library_data, arguments + 1, &metric, &quantization);
    if (named != usearch_status_success_k)
        return answer_failure(library_data, named, "the metric or quantization names no kind", result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_dense_init_vram: @c index → @c metric → @c quantization → @c dimensions
 *  → @c connectivity → @c valuesPerKey → @c devices. */
EXTERN_C DLLEXPORT int DenseInitVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    void** const slot = handles_fresh(&indexes, MArgument_getInteger(arguments[0]));
    if (!slot)
        return answer_failure(library_data, usearch_status_invalid_argument_k, "the DenseIndex is not fresh", result);
    usearch_dense_metric_t metric = usearch_dense_metric_unknown_k;
    usearch_scalar_t quantization = usearch_scalar_unknown_k;
    usearch_status_t const named = geometry_from_arguments(library_data, arguments + 1, &metric, &quantization);
    if (named != usearch_status_success_k)
        return answer_failure(library_data, named, "the metric or quantization names no kind", result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
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
    return dense_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_dense_view_nvme: @c index → @c executor → @c path → @c paths → whether a key
 *  table is held → its memory domains, served read-only. */
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
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const domains_read = ordinals_from_store(library_data, arguments[5], &domains, &domains_count);
    if (domains_read != usearch_status_success_k) {
        free(paths);
        return answer_failure(library_data, domains_read, "key table domains are a list of integers", result);
    }

    char* path_text = MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_ram_options_t const key_table = {domains_count ? domains : NULL, domains_count};
    usearch_dense_view_nvme(&(usearch_dense_view_nvme_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .nvme = {paths_count ? paths : NULL, paths_count},
        .ram = MArgument_getBoolean(arguments[4]) ? &key_table : NULL,
    });
    library_data->UTF8String_disown(path_text);
    free(domains);
    free(paths);
    return dense_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_dense_load_nvme: @c index → @c executor → @c path → @c paths → whether a key
 *  table is held → its memory domains, a manifest opened writable. */
EXTERN_C DLLEXPORT int DenseLoadNvme(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
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
    int32_t* domains = NULL;
    size_t domains_count = 0;
    usearch_status_t const domains_read = ordinals_from_store(library_data, arguments[5], &domains, &domains_count);
    if (domains_read != usearch_status_success_k) {
        free(paths);
        return answer_failure(library_data, domains_read, "key table domains are a list of integers", result);
    }

    char* path_text = MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_dense_index_t index = NULL;
    usearch_ram_options_t const key_table = {domains_count ? domains : NULL, domains_count};
    usearch_dense_load_nvme(&(usearch_dense_load_nvme_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .nvme = {paths_count ? paths : NULL, paths_count},
        .ram = MArgument_getBoolean(arguments[4]) ? &key_table : NULL,
    });
    library_data->UTF8String_disown(path_text);
    free(domains);
    free(paths);
    return dense_adopt(library_data, slot, index, &outcome, result);
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

#pragma region Answers

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
    // Columns past a query's count are unwritten, so only the matches it found are copied.
    for (size_t row = 0, copied = 0; row != call->count; copied += matches.counts[row++]) {
        counts_cells[row] = (mint)matches.counts[row];
        if (!matches.counts[row]) continue;
        memcpy(keys_cells + copied, matches.keys + row * matches.columns, matches.counts[row] * sizeof(usearch_key_t));
        memcpy(scores_cells + copied, matches.scores + row * matches.columns, matches.counts[row] * sizeof(double));
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

/** Copies a ragged run of matches out of the arena: one more offset than rows, keys of
 *  @b [written,16] bytes, and one real per key. */
static int answer_tape(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_matches_tape_view_t const products = call->products;
    mint const offsets_extents[1] = {(mint)call->count + 1};
    mint const keys_extents[2] = {(mint)call->written, (mint)sizeof(usearch_key_t)};
    MTensor offsets;
    if (library_data->MTensor_new(MType_Integer, 1, offsets_extents, &offsets) != LIBRARY_NO_ERROR)
        return LIBRARY_MEMORY_ERROR;
    mint* const offsets_cells = library_data->MTensor_getIntegerData(offsets);
    // A call with nothing to report leaves its offsets NULL, which reads as every query empty.
    if (products.offsets) memcpy(offsets_cells, products.offsets, (call->count + 1) * sizeof(size_t));
    else memset(offsets_cells, 0, (call->count + 1) * sizeof(mint));
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, "Offsets", offsets);
    int const status = store_bytes(library_data, store, "Keys", products.keys, 2, keys_extents);
    return status == LIBRARY_NO_ERROR ? store_reals(library_data, store, "Scores", products.scores, 1, keys_extents)
                                      : status;
}

/** Copies the pairs of @c multiply or the keys of @c maxsim out of the arena, with the pairs
 *  scored before @c wanted cut any. */
static int answer_multiply(WolframLibraryData library_data, call_t* call, DataStore store) {
    int const status = answer_tape(library_data, call, store);
    if (status == LIBRARY_NO_ERROR) store_size(library_data, store, "Required", call->required);
    return status;
}

/** Copies the @c save_buffer monolith out of the arena as an @c UnsignedInteger8 vector. */
static int answer_save_buffer(WolframLibraryData library_data, call_t* call, DataStore store) {
    mint const extents[1] = {(mint)call->monolith.length};
    return store_bytes(library_data, store, "Bytes", call->monolith.data, 1, extents);
}

/** Lands the entries under a key. */
static int answer_count(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Entries", call->tally.entries);
    return LIBRARY_NO_ERROR;
}

/** Adds every field of one reachability audit, each under its own name. */
static void store_reachability(WolframLibraryData library_data, DataStore store,
                               usearch_reachability_t const* reachability) {
    store_size(library_data, store, "Audited", reachability->audited);
    store_size(library_data, store, "Reachable", reachability->reachable);
    store_size(library_data, store, "DepthFromRoot", reachability->depth_from_root);
    store_size(library_data, store, "DepthToRoot", reachability->depth_to_root);
    store_size(library_data, store, "Unlinked", reachability->unlinked);
    store_size(library_data, store, "Underlinked", reachability->underlinked);
    store_size(library_data, store, "BeyondMostPopulatedDepth", reachability->beyond_most_populated_depth);
    store_size(library_data, store, "UnreachedIslands", reachability->unreached_islands);
    store_size(library_data, store, "StrandedIslands", reachability->stranded_islands);
    store_size(library_data, store, "Components", reachability->components);
}

/** Adds one reachability audit under @p name, as a nested store. */
static int store_audit(WolframLibraryData library_data, DataStore store, char const* name,
                       usearch_reachability_t const* reachability) {
    WolframIOLibrary_Functions const io = library_data->ioLibraryFunctions;
    DataStore const audit = io->createDataStore();
    if (!audit) return LIBRARY_MEMORY_ERROR;
    store_reachability(library_data, audit, reachability);
    io->DataStore_addNamedDataStore(store, (char*)name, audit);
    return LIBRARY_NO_ERROR;
}

/** Lands what a refinement did, its two audits nested under @c Opening and @c Reachability. */
static int answer_refine(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_refine_report_t const* const report = &call->report;
    store_size(library_data, store, "Reselected", report->reselected);
    store_size(library_data, store, "Changed", report->changed);
    store_size(library_data, store, "IslandsFound", report->islands_found);
    store_size(library_data, store, "IslandsReconnected", report->islands_reconnected);
    store_size(library_data, store, "Neighborhoods", report->neighborhoods);
    store_size(library_data, store, "WidestNeighborhood", report->widest_neighborhood);
    store_size(library_data, store, "AddedLinks", report->added_links);
    store_size(library_data, store, "DroppedLinks", report->dropped_links);
    store_text(library_data, store, "Verdict", usearch_refine_verdict_name(report->verdict));
    int const opened = store_audit(library_data, store, "Opening", &report->opening);
    return opened != LIBRARY_NO_ERROR ? opened
                                      : store_audit(library_data, store, "Reachability", &report->reachability);
}

/** Lands a reachability audit. */
static int answer_health(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_reachability(library_data, store, &call->reachability);
    return LIBRARY_NO_ERROR;
}

/** Adds the counters every family's description shares, after its frozen metadata. */
static void store_counters(WolframLibraryData library_data, DataStore store, size_t size, size_t capacity,
                           size_t memory_usage, usearch_cpu_class_t cpu_class, usearch_gpu_class_t gpu_class,
                           size_t shards_count, usearch_load_path_t load_path) {
    store_size(library_data, store, "Size", size);
    store_size(library_data, store, "Capacity", capacity);
    store_size(library_data, store, "MemoryUsage", memory_usage);
    store_text(library_data, store, "CPUClass", usearch_cpu_class_name(cpu_class));
    store_text(library_data, store, "GPUClass", usearch_gpu_class_name(gpu_class));
    store_size(library_data, store, "ShardsCount", shards_count);
    store_text(library_data, store, "LoadPath", usearch_load_path_name(load_path));
}

/** Lands a live dense index's frozen metadata and current counters. */
static int answer_dense_describe(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_dense_index_description_t const* const description = &call->description.dense;
    store_metadata(library_data, store, &description->metadata);
    store_counters(library_data, store, description->size, description->capacity, description->memory_usage,
                   description->cpu_class, description->gpu_class, description->shards_count, description->load_path);
    return LIBRARY_NO_ERROR;
}

#pragma endregion Answers

#pragma region Dense Index Verbs

/** Opens a call on a dense index. */
static call_t* dense_open(MArgument const* arguments) {
    return call_open(arguments, &indexes, "the DenseIndex was released or never constructed");
}

/** Calls @c usearch_dense_reserve: @c capacity, the desired total including the current size. */
EXTERN_C DLLEXPORT int DenseReserve(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = dense_open(arguments);
    size_t capacity = 0;
    call_size(call, arguments[receiver_arguments], &capacity, "capacity must not be negative");
    if (call_ready(call))
        usearch_dense_reserve(&(usearch_dense_reserve_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .capacity = capacity,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_clear, keeping the reserved capacity and frozen geometry. */
EXTERN_C DLLEXPORT int DenseClear(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                  MArgument result) {
    call_t* const call = dense_open(arguments);
    if (call_ready(call))
        usearch_dense_clear(&(usearch_dense_clear_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
        });
    return call_answer(library_data, call, result);
}

/**
 *  @brief Calls @c usearch_dense_insert: @c keys → @c vectors → @c expansionFactor.
 *
 *  The keys are an @c UnsignedInteger8 array of one sixteen-byte identity per row, and the vectors
 *  a real matrix, one row apiece; a single vector is the rank-one case of both.
 */
EXTERN_C DLLEXPORT int DenseInsert(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = dense_open(arguments);
    usearch_key_t const* keys = NULL;
    usearch_rows_view_t vectors;
    size_t keys_count = 0, vectors_count = 0, expansion = 0;
    call_keys(library_data, call, arguments[receiver_arguments], &keys, &keys_count);
    call_rows(library_data, call, arguments[receiver_arguments + 1], arguments[receiver_arguments + 2], &vectors,
              &vectors_count);
    call_size(call, arguments[receiver_arguments + 3], &expansion, "expansion factor must not be negative");
    // The call has one batch axis, so only this side sees two arrays that could disagree on it.
    if (call_ready(call) && keys_count != vectors_count)
        call_refuse(call, usearch_status_shape_mismatch_k, "there must be one key per vector");
    if (call_ready(call))
        usearch_dense_insert(&(usearch_dense_insert_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = keys_count,
            .keys = {keys, sizeof(usearch_key_t)},
            .vectors = vectors,
            .expansion_factor = expansion,
        });
    return call_answer(library_data, call, result);
}

/** Reads the walk breadth and NVMe part budget a refine or reshard takes, refusing negatives. */
static usearch_refine_options_t call_refine_options(call_t* call, MArgument const* arguments) {
    usearch_refine_options_t options = {0};
    call_size(call, arguments[0], &options.expansion_factor, "expansion factor must not be negative");
    call_size(call, arguments[1], &options.memory_limit, "memory limit must not be negative");
    return options;
}

/** Calls @c usearch_dense_refine: @c expansionFactor → @c memoryLimit → @c plan, zero running one
 *  round → the refinement's report. */
EXTERN_C DLLEXPORT int DenseRefine(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = dense_open(arguments);
    usearch_refine_options_t const refine = call_refine_options(call, arguments + receiver_arguments);
    call_plan(call, arguments[receiver_arguments + 2]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_dense_refine(&(usearch_dense_refine_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .plan = call->plan_id ? &call->plan.handle.dense_refine : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_refine_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int DenseRefinePlanFinish(WolframLibraryData library_data, mint arguments_count,
                                             MArgument* arguments, MArgument result) {
    call_t* const call = dense_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_dense_refine_plan_finish(&(usearch_dense_refine_plan_finish_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.dense_refine,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_reshard: @c sources → @c targets → whether a RAM placement is named → its
 *  memory domains → whether a VRAM one is → its devices → whether an NVMe one is → its paths →
 *  @c shardCapacity → @c expansionFactor → @c memoryLimit → @c seed → @c plan → the report of the
 *  closing rounds. The receiver's index only names the call's family. */
EXTERN_C DLLEXPORT int DenseReshard(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = dense_open(arguments);
    MArgument const* const listed = arguments + receiver_arguments;
    void const *sources = NULL, *targets = NULL;
    int32_t *domains = NULL, *devices = NULL;
    usearch_bytes_view_t* paths = NULL;
    size_t sources_count = 0, targets_count = 0, domains_count = 0, devices_count = 0, paths_count = 0;
    call_handles(library_data, call, listed[0], &indexes, &sources, &sources_count);
    call_handles(library_data, call, listed[1], &indexes, &targets, &targets_count);
    if (call_ready(call)) {
        usearch_status_t read = ordinals_from_store(library_data, listed[3], &domains, &domains_count);
        if (read == usearch_status_success_k && call_own(call, domains))
            read = ordinals_from_store(library_data, listed[5], &devices, &devices_count);
        if (read == usearch_status_success_k && call_own(call, devices))
            read = paths_from_store(library_data, listed[7], &paths, &paths_count);
        if (read == usearch_status_success_k && !call_own(call, paths)) read = usearch_status_out_of_memory_k;
        if (read != usearch_status_success_k) call_refuse(call, read, "placements list integers or strings");
    }
    size_t shard_capacity = 0;
    call_size(call, listed[8], &shard_capacity, "shard capacity must not be negative");
    usearch_refine_options_t const refine = call_refine_options(call, listed + 9);
    uint64_t const seed = (uint64_t)MArgument_getInteger(listed[11]);
    call_plan(call, listed[12]);
    call->answer = answer_refine;
    // The placements live in the call, so a parked call keeps them alive until it is final.
    usearch_ram_options_t* const ram = (usearch_ram_options_t*)call_own(call, calloc(1, sizeof(usearch_ram_options_t)));
    usearch_vram_options_t* const vram = (usearch_vram_options_t*)call_own(call,
                                                                           calloc(1, sizeof(usearch_vram_options_t)));
    usearch_nvme_options_t* const nvme = (usearch_nvme_options_t*)call_own(call,
                                                                           calloc(1, sizeof(usearch_nvme_options_t)));
    if (call_ready(call) && (!ram || !vram || !nvme))
        call_refuse(call, usearch_status_out_of_memory_k, "the kernel could not copy the placements");
    if (call_ready(call)) {
        *ram = (usearch_ram_options_t) {domains_count ? domains : NULL, domains_count, shard_capacity};
        *vram = (usearch_vram_options_t) {devices_count ? devices : NULL, devices_count, shard_capacity};
        *nvme = (usearch_nvme_options_t) {paths_count ? paths : NULL, paths_count};
        usearch_dense_reshard(&(usearch_dense_reshard_t) {
            .sources_count = sources_count,
            .sources = (usearch_dense_index_t const*)sources,
            .targets_count = targets_count,
            .targets = (usearch_dense_index_t const*)targets,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .ram = MArgument_getBoolean(listed[2]) ? ram : NULL,
            .vram = MArgument_getBoolean(listed[4]) ? vram : NULL,
            .nvme = MArgument_getBoolean(listed[6]) ? nvme : NULL,
            .report = &call->report,
            .refine = refine,
            .seed = seed,
            .plan = call->plan_id ? &call->plan.handle.dense_reshard : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    }
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_reshard_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int DenseReshardPlanFinish(WolframLibraryData library_data, mint arguments_count,
                                              MArgument* arguments, MArgument result) {
    call_t* const call = dense_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_dense_reshard_plan_finish(&(usearch_dense_reshard_plan_finish_t) {
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.dense_reshard,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_health: the population audited, the entries a search reaches, and the
 *  depths. */
EXTERN_C DLLEXPORT int DenseHealth(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = dense_open(arguments);
    call->answer = answer_health;
    if (call_ready(call))
        usearch_dense_health(&(usearch_dense_health_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .reachability = &call->reachability,
        });
    return call_answer(library_data, call, result);
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
    call_t* const call = dense_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_rows_view_t queries;
    size_t wanted = 0, expansion = 0;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    call_rows(library_data, call, verb[0], verb[1], &queries, &call->count);
    call_size(call, verb[2], &wanted, "wanted must not be negative");
    usearch_arena_t const arena = call_arena(call, verb[3]);
    call_size(call, verb[4], &expansion, "expansion factor must not be negative");
    call_traversal(library_data, call, verb[5], &traversal);
    call->answer = answer_matches;
    if (call_ready(call))
        usearch_dense_search(&(usearch_dense_search_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .queries = queries,
            .matches = &call->matches,
            .arena = arena,
            .wanted = wanted,
            .expansion_factor = expansion,
            .traversal = traversal,
        });
    return call_answer(library_data, call, result);
}

/**
 *  @brief Calls @c usearch_dense_maxsim: @c queries → @c rowsPerQuery → @c wanted → @c arena →
 *      @c expansionFactor → offsets, the scored keys, their summed maxima, and the keys written.
 *
 *  The token rows of every query are laid end to end, so their number divides by @p rowsPerQuery.
 *  Query @c i spans keys @c offsets[[i+1]]+1 through @c offsets[[i+2]], largest first; keys are
 *  @b [written,16] byte cells. A zero @p wanted keeps every key a query scored.
 */
EXTERN_C DLLEXPORT int DenseMaxsim(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = dense_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_rows_view_t queries;
    size_t rows = 0, rows_per_query = 0, wanted = 0, expansion = 0;
    call_rows(library_data, call, verb[0], verb[1], &queries, &rows);
    call_size(call, verb[2], &rows_per_query, "rows per query must not be negative");
    if (call_ready(call) && (!rows_per_query || rows % rows_per_query))
        call_refuse(call, usearch_status_shape_mismatch_k, "the token rows must split into whole queries");
    call_size(call, verb[3], &wanted, "wanted must not be negative");
    usearch_arena_t const arena = call_arena(call, verb[4]);
    call_size(call, verb[5], &expansion, "expansion factor must not be negative");
    call->count = rows_per_query ? rows / rows_per_query : 0;
    call->answer = answer_multiply;
    if (call_ready(call))
        usearch_dense_maxsim(&(usearch_dense_maxsim_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .queries = queries,
            .rows_per_query = rows_per_query,
            .matches = &call->products,
            .written = &call->written,
            .required = &call->required,
            .arena = arena,
            .wanted = wanted,
            .expansion_factor = expansion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_count: @c key → the entries stored under it. */
EXTERN_C DLLEXPORT int DenseCount(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                  MArgument result) {
    call_t* const call = dense_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    call->answer = answer_count;
    if (call_ready(call))
        usearch_dense_count(&(usearch_dense_count_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .entries = &call->tally.entries,
        });
    return call_answer(library_data, call, result);
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
    call_t* const call = dense_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments + 1]);
    call->answer = answer_get;
    if (call_ready(call))
        usearch_dense_get(&(usearch_dense_get_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .vectors = &call->vectors,
            .written = &call->written,
            .arena = arena,
            .outputs_scalar = usearch_scalar_f64_k,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_save: @c path, whose extension picks the layout - a @c .safegraphs path
 *  is one monolith, a @c .safegraphs.json path a manifest beside its shards. */
EXTERN_C DLLEXPORT int DenseSave(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                 MArgument result) {
    call_t* const call = dense_open(arguments);
    usearch_bytes_view_t const path = call_text(library_data, call, arguments[receiver_arguments]);
    if (call_ready(call))
        usearch_dense_save(&(usearch_dense_save_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .path = path,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_save_buffer: @c arena → the monolith, as an @c UnsignedInteger8
 *  vector. */
EXTERN_C DLLEXPORT int DenseSaveBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    call_t* const call = dense_open(arguments);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments]);
    call->answer = answer_save_buffer;
    if (call_ready(call))
        usearch_dense_save_buffer(&(usearch_dense_save_buffer_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .monolith = &call->monolith,
            .arena = arena,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_dense_describe: the frozen metadata, counters, classes and shard count. */
EXTERN_C DLLEXPORT int DenseDescribe(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t* const call = dense_open(arguments);
    call->answer = answer_dense_describe;
    if (call_ready(call))
        usearch_dense_describe(&(usearch_dense_describe_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .description = &call->description.dense,
        });
    return call_answer(library_data, call, result);
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
    call_t* const call = dense_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_rows_view_t left;
    size_t wanted = 0, expansion = 0;
    call_rows(library_data, call, verb[0], verb[1], &left, &call->count);
    usearch_arena_t const arena = call_arena(call, verb[2]);
    call_size(call, verb[3], &wanted, "wanted must not be negative");
    call_size(call, verb[4], &expansion, "expansion factor must not be negative");
    call->answer = answer_multiply;
    if (call_ready(call))
        usearch_dense_multiply(&(usearch_dense_multiply_t) {
            .index = (usearch_dense_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .left = left,
            .products = &call->products,
            .written = &call->written,
            .required = &call->required,
            .arena = arena,
            .wanted = wanted,
            .expansion_factor = expansion,
        });
    return call_answer(library_data, call, result);
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

/** Prepares sorted, quantized value rows once on a blocking receiver, for attention to view. */
EXTERN_C DLLEXPORT int DenseValues(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = dense_open(arguments);
    usearch_key_t const* keys = NULL;
    usearch_rows_view_t values;
    size_t keys_count = 0, count = 0;
    call_keys(library_data, call, arguments[receiver_arguments], &keys, &keys_count);
    call_rows(library_data, call, arguments[receiver_arguments + 1], arguments[receiver_arguments + 2], &values,
              &count);
    if (call_ready(call))
        usearch_dense_describe(&(usearch_dense_describe_t) {.index = (usearch_dense_index_t)call->index,
                                                            .executor = call->executor,
                                                            .outcome = &call->outcome,
                                                            .description = &call->description.dense});
    if (call_ready(call) && (!count || count != keys_count))
        call_refuse(call, usearch_status_shape_mismatch_k, "values need one row per key");
    nk_dtype_t const dtype = value_dtype(call->description.dense.metadata.quantization);
    size_t const row_bytes = (values.columns * nk_dtype_bits(dtype) + 7) / 8;
    if (call_ready(call) &&
        (!row_bytes || count > SIZE_MAX / sizeof(value_row_t) || count > (size_t)PTRDIFF_MAX / row_bytes))
        call_refuse(call, usearch_status_shape_mismatch_k, "value storage is too large or unsupported");
    value_row_t* const ordered = call_ready(call) ? (value_row_t*)call_own(call, malloc(count * sizeof(value_row_t)))
                                                  : NULL;
    if (call_ready(call) && !ordered)
        call_refuse(call, usearch_status_out_of_memory_k, "the kernel could not sort the value keys");
    if (!call_ready(call)) return call_finish(library_data, call, result);
    for (size_t row = 0; row != count; ++row) ordered[row] = (value_row_t) {keys[row], row};
    qsort(ordered, count, sizeof(value_row_t), value_row_compare);
    for (size_t row = 1; row != count; ++row)
        if (!memcmp(&ordered[row - 1].key, &ordered[row].key, sizeof(usearch_key_t))) {
            call_refuse(call, usearch_status_invalid_argument_k, "value keys must be unique");
            return call_finish(library_data, call, result);
        }
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    MNumericArray stored_keys = NULL, stored_rows = NULL;
    mint const key_shape[2] = {(mint)count, 16}, value_shape[2] = {(mint)count, (mint)row_bytes};
    if (arrays->MNumericArray_new(MNumericArray_Type_UBit8, 2, key_shape, &stored_keys) != LIBRARY_NO_ERROR ||
        arrays->MNumericArray_new(MNumericArray_Type_UBit8, 2, value_shape, &stored_rows) != LIBRARY_NO_ERROR) {
        if (stored_keys) arrays->MNumericArray_free(stored_keys);
        if (stored_rows) arrays->MNumericArray_free(stored_rows);
        call_free(library_data, call);
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
    usearch_scalar_t const quantization = call->description.dense.metadata.quantization;
    call_free(library_data, call);
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
    store_text(library_data, store, "Quantization", usearch_scalar_name(quantization));
    return LIBRARY_NO_ERROR;
}

/** Finds each key's value row; rows of another kind or width than the call states resolve to none,
 *  which refuses the call. */
static void value_lookup(void* state, usearch_key_t const* keys, size_t count, usearch_scalar_t scalar, size_t columns,
                         void const** rows) {
    value_lookup_t const* const lookup = (value_lookup_t const*)state;
    int const shaped = scalar == lookup->scalar && columns == lookup->columns;
    for (size_t row = 0; row != count; ++row) {
        size_t first = 0, last = lookup->count;
        while (first != last) {
            size_t const middle = first + (last - first) / 2;
            if (memcmp(&lookup->keys[middle], &keys[row], sizeof(usearch_key_t)) < 0) first = middle + 1;
            else last = middle;
        }
        rows[row] = shaped && first != lookup->count && !memcmp(&lookup->keys[first], &keys[row], sizeof(usearch_key_t))
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

/** Calls @c usearch_dense_attend on a RAM index: @c queries → value keys → value rows → their
 *  scalar → their columns → @c arena → @c headCount → @c expansionFactor → @c scoreScale, the value
 *  rows read from a @c DenseValues cache the call passes. */
EXTERN_C DLLEXPORT int DenseAttend(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = dense_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_rows_view_t queries;
    size_t heads = 0, expansion = 0;
    call_rows(library_data, call, verb[0], verb[1], &queries, &call->count);
    call_keys(library_data, call, verb[2], &call->lookup.keys, &call->lookup.count);
    MNumericArray const values = MArgument_getMNumericArray(verb[3]);
    call_hold_array(library_data, call, values);
    usearch_arena_t const arena = call_arena(call, verb[6]);
    call_size(call, verb[7], &heads, "head count must not be negative");
    call_size(call, verb[8], &expansion, "expansion factor must not be negative");
    if (call_ready(call)) {
        char* scalar_name = MArgument_getUTF8String(verb[4]);
        usearch_status_t const named = usearch_scalar_from_name(bytes_from_text(scalar_name), &call->lookup.scalar);
        library_data->UTF8String_disown(scalar_name);
        if (named != usearch_status_success_k) call_refuse(call, named, "the scalar names no kind");
    }
    call->lookup.columns = (size_t)MArgument_getInteger(verb[5]);
    call->lookup.stride = (call->lookup.columns * nk_dtype_bits(value_dtype(call->lookup.scalar)) + 7) / 8;
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    mint const* const shape = arrays->MNumericArray_getDimensions(values);
    // The value rows are this side's own, so only this side can see them disagree with their keys.
    if (call_ready(call) && (arrays->MNumericArray_getType(values) != MNumericArray_Type_UBit8 ||
                             arrays->MNumericArray_getRank(values) != 2 || (size_t)shape[0] != call->lookup.count ||
                             (size_t)shape[1] != call->lookup.stride))
        call_refuse(call, usearch_status_shape_mismatch_k,
                    "value rows must hold one row per key, as wide as their dimensions");
    call->lookup.values = (uint8_t const*)arrays->MNumericArray_getData(values);
    if (call_ready(call) && call->count > SIZE_MAX / sizeof(usearch_dense_softmax_state_t))
        call_refuse(call, usearch_status_out_of_memory_k, "the softmax states overflow");
    if (call_ready(call) && !(call->carry = (usearch_dense_softmax_state_t*)calloc(
                                  call->count ? call->count : 1, sizeof(usearch_dense_softmax_state_t))))
        call_refuse(call, usearch_status_out_of_memory_k, "the kernel could not allocate the softmax states");
    call->written = call->count;
    call->answer = answer_attend;
    if (call_ready(call))
        usearch_dense_attend(&(usearch_dense_attend_t) {.index = (usearch_dense_index_t)call->index,
                                                        .executor = call->executor,
                                                        .outcome = &call->outcome,
                                                        .completion = call->completion,
                                                        .count = call->count,
                                                        .queries = queries,
                                                        .head_count = heads,
                                                        .value_lookup = value_lookup,
                                                        .value_lookup_state = &call->lookup,
                                                        .carry = call->carry,
                                                        .outputs = &call->vectors,
                                                        .arena = arena,
                                                        .outputs_scalar = usearch_scalar_f64_k,
                                                        .expansion_factor = expansion,
                                                        .score_scale = (float)MArgument_getReal(verb[9])});
    return call_answer(library_data, call, result);
}

#pragma endregion Dense Index Verbs

#pragma region Modality Answers

/** The offset at @p position of a tape the library wrote, whichever width its layout names. */
static size_t tape_offset(usearch_tape_view_t const* tape, size_t position) {
    return tape->layout == usearch_tape_u32_k ? (size_t)tape->offsets.u32[position]
                                              : (size_t)tape->offsets.u64[position];
}

/** Adds the length of every entry of a tape the library wrote, under @c Lengths. */
static int store_lengths(WolframLibraryData library_data, DataStore store, usearch_tape_view_t const* tape,
                         size_t written) {
    mint const extents[1] = {(mint)written};
    MTensor lengths;
    if (library_data->MTensor_new(MType_Integer, 1, extents, &lengths) != LIBRARY_NO_ERROR) return LIBRARY_MEMORY_ERROR;
    mint* const cells = library_data->MTensor_getIntegerData(lengths);
    for (size_t entry = 0; entry != written; ++entry)
        cells[entry] = (mint)(tape_offset(tape, entry + 1) - tape_offset(tape, entry));
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, "Lengths", lengths);
    return LIBRARY_NO_ERROR;
}

/** Copies the sequences of @c get out of the arena: their bytes end to end and their lengths. */
static int answer_sequences(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_tape_view_t const* const tape = &call->tape;
    size_t const first = call->written ? tape_offset(tape, 0) : 0;
    size_t const last = call->written ? tape_offset(tape, call->written) : 0;
    mint const extents[1] = {(mint)(last - first)};
    int const status = store_bytes(library_data, store, "Sequences", (uint8_t const*)tape->data + first, 1, extents);
    return status == LIBRARY_NO_ERROR ? store_lengths(library_data, store, tape, call->written) : status;
}

/** Copies the shapes of @c get out of the arena: their @c f64 points end to end as a three-column
 *  real matrix, and each shape's point count. */
static int answer_shapes(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_tape_view_t const* const tape = &call->tape;
    size_t const first = call->written ? tape_offset(tape, 0) : 0;
    size_t const last = call->written ? tape_offset(tape, call->written) : 0;
    mint const extents[2] = {(mint)(last - first), 3};
    int const status = store_reals(library_data, store, "Points", (double const*)tape->data + first * 3, 2, extents);
    return status == LIBRARY_NO_ERROR ? store_lengths(library_data, store, tape, call->written) : status;
}

/** Copies the observations of @c get out of the arena as an eleven-column real matrix beside an
 *  integer mask, a NaN field reading as zero and unrecorded. */
static int answer_observations(WolframLibraryData library_data, call_t* call, DataStore store) {
    size_t const fields = sizeof(usearch_geospatial_observation_t) / sizeof(double);
    mint const extents[2] = {(mint)call->written, (mint)fields};
    MTensor values, recorded;
    if (library_data->MTensor_new(MType_Real, 2, extents, &values) != LIBRARY_NO_ERROR) return LIBRARY_MEMORY_ERROR;
    if (library_data->MTensor_new(MType_Integer, 2, extents, &recorded) != LIBRARY_NO_ERROR) {
        library_data->MTensor_free(values);
        return LIBRARY_MEMORY_ERROR;
    }
    double const* const cells = (double const*)call->observations;
    double* const value_cells = library_data->MTensor_getRealData(values);
    mint* const recorded_cells = library_data->MTensor_getIntegerData(recorded);
    for (size_t cell = 0; cell != call->written * fields; ++cell) {
        recorded_cells[cell] = !isnan(cells[cell]);
        value_cells[cell] = recorded_cells[cell] ? cells[cell] : 0;
    }
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, "Observations", values);
    library_data->ioLibraryFunctions->DataStore_addNamedMTensor(store, "Recorded", recorded);
    return LIBRARY_NO_ERROR;
}

/** Lands the entries under a sequence key with their total and largest byte lengths. */
static int answer_sequence_count(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Entries", call->tally.sequence.entries);
    store_size(library_data, store, "TotalBytes", call->tally.sequence.total_bytes);
    store_size(library_data, store, "LargestBytes", call->tally.sequence.largest_bytes);
    return LIBRARY_NO_ERROR;
}

/** Lands the entries under a shapes key with their total and largest point counts. */
static int answer_shapes_count(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Entries", call->tally.shapes.entries);
    store_size(library_data, store, "TotalPoints", call->tally.shapes.total_points);
    store_size(library_data, store, "LargestPoints", call->tally.shapes.largest_points);
    return LIBRARY_NO_ERROR;
}

/** Lands the entries under a geospatial key. */
static int answer_geospatial_count(WolframLibraryData library_data, call_t* call, DataStore store) {
    store_size(library_data, store, "Entries", call->tally.geospatial.entries);
    return LIBRARY_NO_ERROR;
}

/** Adds a sequence checkpoint's frozen metric and connectivity. */
static void store_sequence_metadata(WolframLibraryData library_data, DataStore store,
                                    usearch_sequence_metadata_t const* metadata) {
    store_text(library_data, store, "Metric", usearch_sequence_metric_name(metadata->metric));
    store_size(library_data, store, "Connectivity", metadata->connectivity);
}

/** Adds a shapes checkpoint's frozen metric, quantization, points and connectivity. */
static void store_shapes_metadata(WolframLibraryData library_data, DataStore store,
                                  usearch_shapes_metadata_t const* metadata) {
    store_text(library_data, store, "Metric", usearch_shapes_metric_name(metadata->metric));
    store_text(library_data, store, "Quantization", usearch_scalar_name(metadata->quantization));
    store_size(library_data, store, "Points", metadata->points);
    store_size(library_data, store, "Connectivity", metadata->connectivity);
}

/** Adds a geospatial checkpoint's frozen connectivity. */
static void store_geospatial_metadata(WolframLibraryData library_data, DataStore store,
                                      usearch_geospatial_metadata_t const* metadata) {
    store_size(library_data, store, "Connectivity", metadata->connectivity);
}

/** Lands a live sequence index's frozen metadata and current counters. */
static int answer_sequence_describe(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_sequence_index_description_t const* const description = &call->description.sequence;
    store_sequence_metadata(library_data, store, &description->metadata);
    store_counters(library_data, store, description->size, description->capacity, description->memory_usage,
                   description->cpu_class, description->gpu_class, description->shards_count, description->load_path);
    return LIBRARY_NO_ERROR;
}

/** Lands a live shapes index's frozen metadata, current counters and points stored. */
static int answer_shapes_describe(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_shapes_index_description_t const* const description = &call->description.shapes;
    store_shapes_metadata(library_data, store, &description->metadata);
    store_counters(library_data, store, description->size, description->capacity, description->memory_usage,
                   description->cpu_class, description->gpu_class, description->shards_count, description->load_path);
    store_size(library_data, store, "TotalPoints", description->total_points);
    return LIBRARY_NO_ERROR;
}

/** Lands a live geospatial index's frozen metadata and current counters. */
static int answer_geospatial_describe(WolframLibraryData library_data, call_t* call, DataStore store) {
    usearch_geospatial_index_description_t const* const description = &call->description.geospatial;
    store_geospatial_metadata(library_data, store, &description->metadata);
    store_counters(library_data, store, description->size, description->capacity, description->memory_usage,
                   description->cpu_class, description->gpu_class, description->shards_count, description->load_path);
    return LIBRARY_NO_ERROR;
}

#pragma endregion Modality Answers

#pragma region Modality Lifecycle

/** Reads a fresh managed expression's slot, refusing an ID a constructor already filled. */
static void** fresh_slot(WolframLibraryData library_data, handles_t const* family, MArgument argument, MArgument result,
                         int* answered) {
    void** const slot = handles_fresh(family, MArgument_getInteger(argument));
    if (!slot)
        *answered = answer_failure(library_data, usearch_status_invalid_argument_k, "the index is not fresh", result);
    return slot;
}

/**
 *  @brief Reads the weights an alignment kind scores with: whether any are named, a 256×256
 *      @c Integer8 substitution matrix, and the gap cost.
 *  @return @c usearch_status_shape_mismatch_k for a matrix of another shape, which the library
 *      cannot see, as it reads a fixed 256×256.
 */
static usearch_status_t weights_from_arguments(WolframLibraryData library_data, MArgument const* arguments,
                                               usearch_sequence_weights_t* weights, int* named) {
    *named = MArgument_getBoolean(arguments[0]);
    if (!*named) return usearch_status_success_k;
    WolframNumericArrayLibrary_Functions const arrays = library_data->numericarrayLibraryFunctions;
    MNumericArray const matrix = MArgument_getMNumericArray(arguments[1]);
    mint const* const extents = arrays->MNumericArray_getDimensions(matrix);
    if (arrays->MNumericArray_getType(matrix) != MNumericArray_Type_Bit8 ||
        arrays->MNumericArray_getRank(matrix) != 2 || extents[0] != 256 || extents[1] != 256)
        return usearch_status_shape_mismatch_k;
    weights->substitution_matrix = (int8_t const*)arrays->MNumericArray_getData(matrix);
    weights->gap = (int8_t)MArgument_getInteger(arguments[2]);
    return usearch_status_success_k;
}

/** Reads a sequence metric by its canonical name. */
static usearch_status_t sequence_metric_from_argument(WolframLibraryData library_data, MArgument argument,
                                                      usearch_sequence_metric_t* metric) {
    char* metric_text = MArgument_getUTF8String(argument);
    usearch_status_t const named = usearch_sequence_metric_from_name(bytes_from_text(metric_text), metric);
    library_data->UTF8String_disown(metric_text);
    return named;
}

/** Fills a fresh sequence index's slot, or frees what a failed call left and reports why. */
static int sequence_adopt(WolframLibraryData library_data, void** slot, usearch_sequence_index_t handle,
                          usearch_outcome_t const* outcome, MArgument result) {
    if (outcome->status != usearch_status_success_k) usearch_sequence_free(handle);
    return index_adopt(library_data, slot, handle, outcome, result);
}

/** Calls @c usearch_sequence_init_ram or @c usearch_sequence_init_vram: @c index → @c metric →
 *  whether weights are named → @c substitutionMatrix → @c gap → @c windowWidths → @c connectivity →
 *  @c valuesPerKey → the memory domains or devices. */
static int sequence_init(WolframLibraryData library_data, MArgument const* arguments, MArgument result, int vram) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &sequences, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_sequence_metric_t metric = usearch_sequence_metric_unknown_k;
    usearch_status_t const named = sequence_metric_from_argument(library_data, arguments[1], &metric);
    if (named != usearch_status_success_k)
        return answer_failure(library_data, named, "the metric names no kind", result);
    usearch_sequence_weights_t weights = {0};
    int weighted = 0;
    if (weights_from_arguments(library_data, arguments + 2, &weights, &weighted) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_shape_mismatch_k,
                              "a substitution matrix is a 256 by 256 Integer8 array", result);
    size_t* widths = NULL;
    size_t widths_count = 0;
    usearch_status_t read = sizes_from_store(library_data, arguments[5], &widths, &widths_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "window widths are a list of non-negative integers", result);
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    read = ordinals_from_store(library_data, arguments[8], &ordinals, &ordinals_count);
    if (read != usearch_status_success_k) {
        free(widths);
        return answer_failure(library_data, read, "memory domains and devices are lists of integers", result);
    }
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_index_t index = NULL;
    if (vram)
        usearch_sequence_init_vram(&(usearch_sequence_init_vram_t) {
            .index = &index,
            .outcome = &outcome,
            .metric = metric,
            .weights = weighted ? &weights : NULL,
            .window_widths = widths_count ? widths : NULL,
            .window_widths_count = widths_count,
            .connectivity = (size_t)MArgument_getInteger(arguments[6]),
            .values_per_key = (size_t)MArgument_getInteger(arguments[7]),
            .vram = {ordinals_count ? ordinals : NULL, ordinals_count},
        });
    else
        usearch_sequence_init_ram(&(usearch_sequence_init_ram_t) {
            .index = &index,
            .outcome = &outcome,
            .metric = metric,
            .weights = weighted ? &weights : NULL,
            .window_widths = widths_count ? widths : NULL,
            .window_widths_count = widths_count,
            .connectivity = (size_t)MArgument_getInteger(arguments[6]),
            .values_per_key = (size_t)MArgument_getInteger(arguments[7]),
            .ram = {ordinals_count ? ordinals : NULL, ordinals_count},
        });
    free(ordinals);
    free(widths);
    return sequence_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_sequence_init_ram, as @c sequence_init lists. */
EXTERN_C DLLEXPORT int SequenceInitRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    return sequence_init(library_data, arguments, result, 0);
}

/** Calls @c usearch_sequence_init_vram, as @c sequence_init lists. */
EXTERN_C DLLEXPORT int SequenceInitVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    return sequence_init(library_data, arguments, result, 1);
}

/**
 *  @brief Calls a sequence load: @c index → @c executor → @c path or @c monolith → the memory
 *      domains or devices → @c valuesPerKey → whether weights are named → @c substitutionMatrix
 *      → @c gap.
 *  @param[in] vram Whether the copy lands in device memory.
 *  @param[in] buffered Whether the checkpoint is a monolith in memory rather than a path.
 */
static int sequence_load(WolframLibraryData library_data, MArgument const* arguments, MArgument result, int vram,
                         int buffered) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &sequences, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer = {0};
    if (buffered &&
        buffer_from_array(library_data, MArgument_getMNumericArray(arguments[2]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    usearch_sequence_weights_t weights = {0};
    int weighted = 0;
    if (weights_from_arguments(library_data, arguments + 5, &weights, &weighted) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_shape_mismatch_k,
                              "a substitution matrix is a 256 by 256 Integer8 array", result);
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &ordinals, &ordinals_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains and devices are lists of integers", result);
    char* path_text = buffered ? NULL : MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_index_t index = NULL;
    usearch_ram_options_t const ram = {ordinals_count ? ordinals : NULL, ordinals_count};
    usearch_vram_options_t const devices = {ordinals_count ? ordinals : NULL, ordinals_count};
    size_t const values_per_key = (size_t)MArgument_getInteger(arguments[4]);
    usearch_sequence_weights_t const* const named = weighted ? &weights : NULL;
    if (vram && buffered)
        usearch_sequence_load_vram_buffer(&(usearch_sequence_load_vram_buffer_t) {.index = &index,
                                                                                  .executor = executor,
                                                                                  .outcome = &outcome,
                                                                                  .buffer = buffer,
                                                                                  .vram = devices,
                                                                                  .values_per_key = values_per_key,
                                                                                  .weights = named});
    else if (vram)
        usearch_sequence_load_vram(&(usearch_sequence_load_vram_t) {.index = &index,
                                                                    .executor = executor,
                                                                    .outcome = &outcome,
                                                                    .path = bytes_from_text(path_text),
                                                                    .vram = devices,
                                                                    .values_per_key = values_per_key,
                                                                    .weights = named});
    else if (buffered)
        usearch_sequence_load_ram_buffer(&(usearch_sequence_load_ram_buffer_t) {.index = &index,
                                                                                .executor = executor,
                                                                                .outcome = &outcome,
                                                                                .buffer = buffer,
                                                                                .ram = ram,
                                                                                .values_per_key = values_per_key,
                                                                                .weights = named});
    else
        usearch_sequence_load_ram(&(usearch_sequence_load_ram_t) {.index = &index,
                                                                  .executor = executor,
                                                                  .outcome = &outcome,
                                                                  .path = bytes_from_text(path_text),
                                                                  .ram = ram,
                                                                  .values_per_key = values_per_key,
                                                                  .weights = named});
    if (path_text) library_data->UTF8String_disown(path_text);
    free(ordinals);
    return sequence_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_sequence_load_ram, as @c sequence_load lists. */
EXTERN_C DLLEXPORT int SequenceLoadRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    return sequence_load(library_data, arguments, result, 0, 0);
}

/** Calls @c usearch_sequence_load_vram, as @c sequence_load lists. */
EXTERN_C DLLEXPORT int SequenceLoadVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    return sequence_load(library_data, arguments, result, 1, 0);
}

/** Calls @c usearch_sequence_load_ram_buffer, as @c sequence_load lists. */
EXTERN_C DLLEXPORT int SequenceLoadRamBuffer(WolframLibraryData library_data, mint arguments_count,
                                             MArgument* arguments, MArgument result) {
    return sequence_load(library_data, arguments, result, 0, 1);
}

/** Calls @c usearch_sequence_load_vram_buffer, as @c sequence_load lists. */
EXTERN_C DLLEXPORT int SequenceLoadVramBuffer(WolframLibraryData library_data, mint arguments_count,
                                              MArgument* arguments, MArgument result) {
    return sequence_load(library_data, arguments, result, 1, 1);
}

/** Calls @c usearch_sequence_view_ram: @c index → @c path → whether weights are named →
 *  @c substitutionMatrix → @c gap, served in place and read-only. */
EXTERN_C DLLEXPORT int SequenceViewRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &sequences, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_sequence_weights_t weights = {0};
    int weighted = 0;
    if (weights_from_arguments(library_data, arguments + 2, &weights, &weighted) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_shape_mismatch_k,
                              "a substitution matrix is a 256 by 256 Integer8 array", result);
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_index_t index = NULL;
    usearch_sequence_view_ram(&(usearch_sequence_view_ram_t) {
        .index = &index,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .weights = weighted ? &weights : NULL,
    });
    library_data->UTF8String_disown(path_text);
    return sequence_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_sequence_view_nvme: @c index → @c executor → @c path → @c paths → whether
 *  weights are named → @c substitutionMatrix → @c gap, served read-only. */
EXTERN_C DLLEXPORT int SequenceViewNvme(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &sequences, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_sequence_weights_t weights = {0};
    int weighted = 0;
    if (weights_from_arguments(library_data, arguments + 4, &weights, &weighted) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_shape_mismatch_k,
                              "a substitution matrix is a 256 by 256 Integer8 array", result);
    usearch_bytes_view_t* paths = NULL;
    size_t paths_count = 0;
    usearch_status_t const read = paths_from_store(library_data, arguments[3], &paths, &paths_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "paths are a list of strings", result);
    char* path_text = MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_index_t index = NULL;
    usearch_sequence_view_nvme(&(usearch_sequence_view_nvme_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .nvme = {paths_count ? paths : NULL, paths_count},
        .weights = weighted ? &weights : NULL,
    });
    library_data->UTF8String_disown(path_text);
    free(paths);
    return sequence_adopt(library_data, slot, index, &outcome, result);
}

/** Answers checkpoint metadata a read filled, or its status. */
static int answer_metadata(WolframLibraryData library_data, usearch_outcome_t const* outcome, MArgument result,
                           void (*store_into)(WolframLibraryData, DataStore, void const*), void const* metadata) {
    if (outcome->status != usearch_status_success_k) return answer_outcome(library_data, outcome, result);
    int const answered = answer_nothing(library_data, result);
    if (answered == LIBRARY_NO_ERROR) store_into(library_data, MArgument_getDataStore(result), metadata);
    return answered;
}

static void store_sequence_metadata_any(WolframLibraryData library_data, DataStore store, void const* metadata) {
    store_sequence_metadata(library_data, store, (usearch_sequence_metadata_t const*)metadata);
}

static void store_shapes_metadata_any(WolframLibraryData library_data, DataStore store, void const* metadata) {
    store_shapes_metadata(library_data, store, (usearch_shapes_metadata_t const*)metadata);
}

static void store_geospatial_metadata_any(WolframLibraryData library_data, DataStore store, void const* metadata) {
    store_geospatial_metadata(library_data, store, (usearch_geospatial_metadata_t const*)metadata);
}

/** Calls @c usearch_sequence_metadata_read: @c executor → @c path → the frozen
 *  metric and connectivity. */
EXTERN_C DLLEXPORT int SequenceMetadataRead(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                            MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_sequence_metadata_t metadata = {0};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_metadata_read(&(usearch_sequence_metadata_read_t) {.executor = executor,
                                                                        .outcome = &outcome,
                                                                        .path = bytes_from_text(path_text),
                                                                        .metadata = &metadata});
    library_data->UTF8String_disown(path_text);
    return answer_metadata(library_data, &outcome, result, store_sequence_metadata_any, &metadata);
}

/** Calls @c usearch_sequence_metadata_read_buffer: @c executor → @c monolith → the frozen
 *  metric and connectivity. */
EXTERN_C DLLEXPORT int SequenceMetadataReadBuffer(WolframLibraryData library_data, mint arguments_count,
                                                  MArgument* arguments, MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer;
    if (buffer_from_array(library_data, MArgument_getMNumericArray(arguments[1]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    usearch_sequence_metadata_t metadata = {0};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_sequence_metadata_read_buffer(&(usearch_sequence_metadata_read_buffer_t) {.executor = executor,
                                                                                      .outcome = &outcome,
                                                                                      .buffer = buffer,
                                                                                      .metadata = &metadata});
    return answer_metadata(library_data, &outcome, result, store_sequence_metadata_any, &metadata);
}

/** Fills a fresh shapes index's slot, or frees what a failed call left and reports why. */
static int shapes_adopt(WolframLibraryData library_data, void** slot, usearch_shapes_index_t handle,
                        usearch_outcome_t const* outcome, MArgument result) {
    if (outcome->status != usearch_status_success_k) usearch_shapes_free(handle);
    return index_adopt(library_data, slot, handle, outcome, result);
}

/** Reads a shapes metric and a quantization by their canonical names, answering the status of the
 *  first name the library refuses. */
static usearch_status_t shapes_geometry_from_arguments(WolframLibraryData library_data, MArgument const* arguments,
                                                       usearch_shapes_metric_t* metric,
                                                       usearch_scalar_t* quantization) {
    char* metric_text = MArgument_getUTF8String(arguments[0]);
    char* quantization_text = MArgument_getUTF8String(arguments[1]);
    usearch_status_t named = usearch_shapes_metric_from_name(bytes_from_text(metric_text), metric);
    if (named == usearch_status_success_k)
        named = usearch_scalar_from_name(bytes_from_text(quantization_text), quantization);
    library_data->UTF8String_disown(metric_text);
    library_data->UTF8String_disown(quantization_text);
    return named;
}

/** Calls @c usearch_shapes_init_ram or @c usearch_shapes_init_vram: @c index → @c metric
 *  → @c quantization → @c points → @c connectivity → @c valuesPerKey → the memory
 *  domains or devices. */
static int shapes_init(WolframLibraryData library_data, MArgument const* arguments, MArgument result, int vram) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &shapes, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_shapes_metric_t metric = usearch_shapes_metric_unknown_k;
    usearch_scalar_t quantization = usearch_scalar_unknown_k;
    usearch_status_t const named = shapes_geometry_from_arguments(library_data, arguments + 1, &metric, &quantization);
    if (named != usearch_status_success_k)
        return answer_failure(library_data, named, "the metric or quantization names no kind", result);
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[6], &ordinals, &ordinals_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains and devices are lists of integers", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_shapes_index_t index = NULL;
    if (vram)
        usearch_shapes_init_vram(&(usearch_shapes_init_vram_t) {
            .index = &index,
            .outcome = &outcome,
            .metric = metric,
            .quantization = quantization,
            .points = (size_t)MArgument_getInteger(arguments[3]),
            .connectivity = (size_t)MArgument_getInteger(arguments[4]),
            .values_per_key = (size_t)MArgument_getInteger(arguments[5]),
            .vram = {ordinals_count ? ordinals : NULL, ordinals_count},
        });
    else
        usearch_shapes_init_ram(&(usearch_shapes_init_ram_t) {
            .index = &index,
            .outcome = &outcome,
            .metric = metric,
            .quantization = quantization,
            .points = (size_t)MArgument_getInteger(arguments[3]),
            .connectivity = (size_t)MArgument_getInteger(arguments[4]),
            .values_per_key = (size_t)MArgument_getInteger(arguments[5]),
            .ram = {ordinals_count ? ordinals : NULL, ordinals_count},
        });
    free(ordinals);
    return shapes_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_shapes_init_ram, as @c shapes_init lists. */
EXTERN_C DLLEXPORT int ShapesInitRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    return shapes_init(library_data, arguments, result, 0);
}

/** Calls @c usearch_shapes_init_vram, as @c shapes_init lists. */
EXTERN_C DLLEXPORT int ShapesInitVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    return shapes_init(library_data, arguments, result, 1);
}

/** Calls a shapes load: @c index → @c executor → @c path or @c monolith → the memory domains or
 *  devices → @c valuesPerKey. */
static int shapes_load(WolframLibraryData library_data, MArgument const* arguments, MArgument result, int vram,
                       int buffered) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &shapes, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer = {0};
    if (buffered &&
        buffer_from_array(library_data, MArgument_getMNumericArray(arguments[2]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &ordinals, &ordinals_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains and devices are lists of integers", result);
    char* path_text = buffered ? NULL : MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_shapes_index_t index = NULL;
    usearch_ram_options_t const ram = {ordinals_count ? ordinals : NULL, ordinals_count};
    usearch_vram_options_t const devices = {ordinals_count ? ordinals : NULL, ordinals_count};
    size_t const values_per_key = (size_t)MArgument_getInteger(arguments[4]);
    if (vram && buffered)
        usearch_shapes_load_vram_buffer(&(usearch_shapes_load_vram_buffer_t) {.index = &index,
                                                                              .executor = executor,
                                                                              .outcome = &outcome,
                                                                              .buffer = buffer,
                                                                              .vram = devices,
                                                                              .values_per_key = values_per_key});
    else if (vram)
        usearch_shapes_load_vram(&(usearch_shapes_load_vram_t) {.index = &index,
                                                                .executor = executor,
                                                                .outcome = &outcome,
                                                                .path = bytes_from_text(path_text),
                                                                .vram = devices,
                                                                .values_per_key = values_per_key});
    else if (buffered)
        usearch_shapes_load_ram_buffer(&(usearch_shapes_load_ram_buffer_t) {.index = &index,
                                                                            .executor = executor,
                                                                            .outcome = &outcome,
                                                                            .buffer = buffer,
                                                                            .ram = ram,
                                                                            .values_per_key = values_per_key});
    else
        usearch_shapes_load_ram(&(usearch_shapes_load_ram_t) {.index = &index,
                                                              .executor = executor,
                                                              .outcome = &outcome,
                                                              .path = bytes_from_text(path_text),
                                                              .ram = ram,
                                                              .values_per_key = values_per_key});
    if (path_text) library_data->UTF8String_disown(path_text);
    free(ordinals);
    return shapes_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_shapes_load_ram, as @c shapes_load lists. */
EXTERN_C DLLEXPORT int ShapesLoadRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    return shapes_load(library_data, arguments, result, 0, 0);
}

/** Calls @c usearch_shapes_load_vram, as @c shapes_load lists. */
EXTERN_C DLLEXPORT int ShapesLoadVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    return shapes_load(library_data, arguments, result, 1, 0);
}

/** Calls @c usearch_shapes_load_ram_buffer, as @c shapes_load lists. */
EXTERN_C DLLEXPORT int ShapesLoadRamBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                           MArgument result) {
    return shapes_load(library_data, arguments, result, 0, 1);
}

/** Calls @c usearch_shapes_load_vram_buffer, as @c shapes_load lists. */
EXTERN_C DLLEXPORT int ShapesLoadVramBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                            MArgument result) {
    return shapes_load(library_data, arguments, result, 1, 1);
}

/** Calls @c usearch_shapes_view_ram: @c index → @c path, served in place and read-only. */
EXTERN_C DLLEXPORT int ShapesViewRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &shapes, arguments[0], result, &answered);
    if (!slot) return answered;
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_shapes_index_t index = NULL;
    usearch_shapes_view_ram(
        &(usearch_shapes_view_ram_t) {.index = &index, .outcome = &outcome, .path = bytes_from_text(path_text)});
    library_data->UTF8String_disown(path_text);
    return shapes_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_shapes_view_nvme: @c index → @c executor → @c path →
 *  @c paths, served read-only. */
EXTERN_C DLLEXPORT int ShapesViewNvme(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &shapes, arguments[0], result, &answered);
    if (!slot) return answered;
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
    usearch_shapes_index_t index = NULL;
    usearch_shapes_view_nvme(&(usearch_shapes_view_nvme_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .nvme = {paths_count ? paths : NULL, paths_count},
    });
    library_data->UTF8String_disown(path_text);
    free(paths);
    return shapes_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_shapes_metadata_read: @c executor → @c path → the frozen geometry. */
EXTERN_C DLLEXPORT int ShapesMetadataRead(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_shapes_metadata_t metadata = {0};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_shapes_metadata_read(&(usearch_shapes_metadata_read_t) {.executor = executor,
                                                                    .outcome = &outcome,
                                                                    .path = bytes_from_text(path_text),
                                                                    .metadata = &metadata});
    library_data->UTF8String_disown(path_text);
    return answer_metadata(library_data, &outcome, result, store_shapes_metadata_any, &metadata);
}

/** Calls @c usearch_shapes_metadata_read_buffer: @c executor → @c monolith →
 *  the frozen geometry. */
EXTERN_C DLLEXPORT int ShapesMetadataReadBuffer(WolframLibraryData library_data, mint arguments_count,
                                                MArgument* arguments, MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer;
    if (buffer_from_array(library_data, MArgument_getMNumericArray(arguments[1]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    usearch_shapes_metadata_t metadata = {0};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_shapes_metadata_read_buffer(&(usearch_shapes_metadata_read_buffer_t) {.executor = executor,
                                                                                  .outcome = &outcome,
                                                                                  .buffer = buffer,
                                                                                  .metadata = &metadata});
    return answer_metadata(library_data, &outcome, result, store_shapes_metadata_any, &metadata);
}

/** Fills a fresh geospatial index's slot, or frees what a failed call left and reports why. */
static int geospatial_adopt(WolframLibraryData library_data, void** slot, usearch_geospatial_index_t handle,
                            usearch_outcome_t const* outcome, MArgument result) {
    if (outcome->status != usearch_status_success_k) usearch_geospatial_free(handle);
    return index_adopt(library_data, slot, handle, outcome, result);
}

/** Calls @c usearch_geospatial_init_ram or @c usearch_geospatial_init_vram: @c index →
 *  @c connectivity → @c valuesPerKey → the memory domains or devices, with no terrain. */
static int geospatial_init(WolframLibraryData library_data, MArgument const* arguments, MArgument result, int vram) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &geospatials, arguments[0], result, &answered);
    if (!slot) return answered;
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &ordinals, &ordinals_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains and devices are lists of integers", result);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_geospatial_index_t index = NULL;
    if (vram)
        usearch_geospatial_init_vram(&(usearch_geospatial_init_vram_t) {
            .index = &index,
            .outcome = &outcome,
            .connectivity = (size_t)MArgument_getInteger(arguments[1]),
            .values_per_key = (size_t)MArgument_getInteger(arguments[2]),
            .vram = {ordinals_count ? ordinals : NULL, ordinals_count},
        });
    else
        usearch_geospatial_init_ram(&(usearch_geospatial_init_ram_t) {
            .index = &index,
            .outcome = &outcome,
            .connectivity = (size_t)MArgument_getInteger(arguments[1]),
            .values_per_key = (size_t)MArgument_getInteger(arguments[2]),
            .ram = {ordinals_count ? ordinals : NULL, ordinals_count},
        });
    free(ordinals);
    return geospatial_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_geospatial_init_ram, as @c geospatial_init lists. */
EXTERN_C DLLEXPORT int GeospatialInitRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    return geospatial_init(library_data, arguments, result, 0);
}

/** Calls @c usearch_geospatial_init_vram, as @c geospatial_init lists. */
EXTERN_C DLLEXPORT int GeospatialInitVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    return geospatial_init(library_data, arguments, result, 1);
}

/** Calls a geospatial load: @c index → @c executor → @c path or @c monolith → the memory domains or
 *  devices → @c valuesPerKey, with no terrain. */
static int geospatial_load(WolframLibraryData library_data, MArgument const* arguments, MArgument result, int vram,
                           int buffered) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &geospatials, arguments[0], result, &answered);
    if (!slot) return answered;
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[1], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer = {0};
    if (buffered &&
        buffer_from_array(library_data, MArgument_getMNumericArray(arguments[2]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    int32_t* ordinals = NULL;
    size_t ordinals_count = 0;
    usearch_status_t const read = ordinals_from_store(library_data, arguments[3], &ordinals, &ordinals_count);
    if (read != usearch_status_success_k)
        return answer_failure(library_data, read, "memory domains and devices are lists of integers", result);
    char* path_text = buffered ? NULL : MArgument_getUTF8String(arguments[2]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_geospatial_index_t index = NULL;
    usearch_ram_options_t const ram = {ordinals_count ? ordinals : NULL, ordinals_count};
    usearch_vram_options_t const devices = {ordinals_count ? ordinals : NULL, ordinals_count};
    size_t const values_per_key = (size_t)MArgument_getInteger(arguments[4]);
    if (vram && buffered)
        usearch_geospatial_load_vram_buffer(
            &(usearch_geospatial_load_vram_buffer_t) {.index = &index,
                                                      .executor = executor,
                                                      .outcome = &outcome,
                                                      .buffer = buffer,
                                                      .vram = devices,
                                                      .values_per_key = values_per_key});
    else if (vram)
        usearch_geospatial_load_vram(&(usearch_geospatial_load_vram_t) {.index = &index,
                                                                        .executor = executor,
                                                                        .outcome = &outcome,
                                                                        .path = bytes_from_text(path_text),
                                                                        .vram = devices,
                                                                        .values_per_key = values_per_key});
    else if (buffered)
        usearch_geospatial_load_ram_buffer(&(usearch_geospatial_load_ram_buffer_t) {.index = &index,
                                                                                    .executor = executor,
                                                                                    .outcome = &outcome,
                                                                                    .buffer = buffer,
                                                                                    .ram = ram,
                                                                                    .values_per_key = values_per_key});
    else
        usearch_geospatial_load_ram(&(usearch_geospatial_load_ram_t) {.index = &index,
                                                                      .executor = executor,
                                                                      .outcome = &outcome,
                                                                      .path = bytes_from_text(path_text),
                                                                      .ram = ram,
                                                                      .values_per_key = values_per_key});
    if (path_text) library_data->UTF8String_disown(path_text);
    free(ordinals);
    return geospatial_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_geospatial_load_ram, as @c geospatial_load lists. */
EXTERN_C DLLEXPORT int GeospatialLoadRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    return geospatial_load(library_data, arguments, result, 0, 0);
}

/** Calls @c usearch_geospatial_load_vram, as @c geospatial_load lists. */
EXTERN_C DLLEXPORT int GeospatialLoadVram(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    return geospatial_load(library_data, arguments, result, 1, 0);
}

/** Calls @c usearch_geospatial_load_ram_buffer, as @c geospatial_load lists. */
EXTERN_C DLLEXPORT int GeospatialLoadRamBuffer(WolframLibraryData library_data, mint arguments_count,
                                               MArgument* arguments, MArgument result) {
    return geospatial_load(library_data, arguments, result, 0, 1);
}

/** Calls @c usearch_geospatial_load_vram_buffer, as @c geospatial_load lists. */
EXTERN_C DLLEXPORT int GeospatialLoadVramBuffer(WolframLibraryData library_data, mint arguments_count,
                                                MArgument* arguments, MArgument result) {
    return geospatial_load(library_data, arguments, result, 1, 1);
}

/** Calls @c usearch_geospatial_view_ram: @c index → @c path, served in place and read-only. */
EXTERN_C DLLEXPORT int GeospatialViewRam(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &geospatials, arguments[0], result, &answered);
    if (!slot) return answered;
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_geospatial_index_t index = NULL;
    usearch_geospatial_view_ram(
        &(usearch_geospatial_view_ram_t) {.index = &index, .outcome = &outcome, .path = bytes_from_text(path_text)});
    library_data->UTF8String_disown(path_text);
    return geospatial_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_geospatial_view_nvme: @c index → @c executor → @c path →
 *  @c paths, served read-only. */
EXTERN_C DLLEXPORT int GeospatialViewNvme(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    int answered = LIBRARY_NO_ERROR;
    void** const slot = fresh_slot(library_data, &geospatials, arguments[0], result, &answered);
    if (!slot) return answered;
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
    usearch_geospatial_index_t index = NULL;
    usearch_geospatial_view_nvme(&(usearch_geospatial_view_nvme_t) {
        .index = &index,
        .executor = executor,
        .outcome = &outcome,
        .path = bytes_from_text(path_text),
        .nvme = {paths_count ? paths : NULL, paths_count},
    });
    library_data->UTF8String_disown(path_text);
    free(paths);
    return geospatial_adopt(library_data, slot, index, &outcome, result);
}

/** Calls @c usearch_geospatial_metadata_read: @c executor → @c path → the frozen connectivity. */
EXTERN_C DLLEXPORT int GeospatialMetadataRead(WolframLibraryData library_data, mint arguments_count,
                                              MArgument* arguments, MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    char* path_text = MArgument_getUTF8String(arguments[1]);
    usearch_geospatial_metadata_t metadata = {0};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_geospatial_metadata_read(&(usearch_geospatial_metadata_read_t) {.executor = executor,
                                                                            .outcome = &outcome,
                                                                            .path = bytes_from_text(path_text),
                                                                            .metadata = &metadata});
    library_data->UTF8String_disown(path_text);
    return answer_metadata(library_data, &outcome, result, store_geospatial_metadata_any, &metadata);
}

/** Calls @c usearch_geospatial_metadata_read_buffer: @c executor → @c monolith → the
 *  frozen connectivity. */
EXTERN_C DLLEXPORT int GeospatialMetadataReadBuffer(WolframLibraryData library_data, mint arguments_count,
                                                    MArgument* arguments, MArgument result) {
    usearch_executor_t executor;
    if (!executor_from_argument(arguments[0], &executor))
        return answer_failure(library_data, usearch_status_uninitialized_k, "the Executor was released", result);
    usearch_bytes_view_t buffer;
    if (buffer_from_array(library_data, MArgument_getMNumericArray(arguments[1]), &buffer) != usearch_status_success_k)
        return answer_failure(library_data, usearch_status_malformed_k, "a monolith is an UnsignedInteger8 vector",
                              result);
    usearch_geospatial_metadata_t metadata = {0};
    usearch_outcome_t outcome = {usearch_status_success_k, NULL};
    usearch_geospatial_metadata_read_buffer(&(usearch_geospatial_metadata_read_buffer_t) {.executor = executor,
                                                                                          .outcome = &outcome,
                                                                                          .buffer = buffer,
                                                                                          .metadata = &metadata});
    return answer_metadata(library_data, &outcome, result, store_geospatial_metadata_any, &metadata);
}

#pragma endregion Modality Lifecycle

#pragma region Sequence Index Verbs

/** Opens a call on a sequence index. */
static call_t* sequence_open(MArgument const* arguments) {
    return call_open(arguments, &sequences, "the SequenceIndex was released or never constructed");
}

/** Calls @c usearch_sequence_reserve: @c entries, the desired total including the current size →
 *  @c payloadBytes, the bytes they hold in all. */
EXTERN_C DLLEXPORT int SequenceReserve(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    call_t* const call = sequence_open(arguments);
    size_t entries = 0, extra = 0;
    call_size(call, arguments[receiver_arguments], &entries, "entries must not be negative");
    call_size(call, arguments[receiver_arguments + 1], &extra, "payload bytes must not be negative");
    if (call_ready(call))
        usearch_sequence_reserve(&(usearch_sequence_reserve_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .entries = entries,
            .payload_bytes = extra,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_clear, keeping the reserved capacity and frozen settings. */
EXTERN_C DLLEXPORT int SequenceClear(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t* const call = sequence_open(arguments);
    if (call_ready(call))
        usearch_sequence_clear(&(usearch_sequence_clear_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_insert: @c keys → the bytes of every sequence end to end → their
 *  @c count + 1 offsets → @c expansionFactor. */
EXTERN_C DLLEXPORT int SequenceInsert(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = sequence_open(arguments);
    usearch_key_t const* keys = NULL;
    usearch_tape_view_t tape = {0};
    size_t keys_count = 0, entries = 0, expansion = 0;
    call_keys(library_data, call, arguments[receiver_arguments], &keys, &keys_count);
    call_bytes_tape(library_data, call, arguments[receiver_arguments + 1], &tape);
    call_offsets(library_data, call, arguments[receiver_arguments + 2], &tape, &entries);
    call_size(call, arguments[receiver_arguments + 3], &expansion, "expansion factor must not be negative");
    // The call has one batch axis, so only this side sees the keys and the tape disagree on it.
    if (call_ready(call) && keys_count != entries)
        call_refuse(call, usearch_status_shape_mismatch_k, "there must be one key per sequence");
    if (call_ready(call))
        usearch_sequence_insert(&(usearch_sequence_insert_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = entries,
            .keys = {keys, sizeof(usearch_key_t)},
            .sequences = tape,
            .expansion_factor = expansion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_refine: @c expansionFactor → @c memoryLimit → @c plan, zero running
 *  one round → the refinement's report. */
EXTERN_C DLLEXPORT int SequenceRefine(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = sequence_open(arguments);
    usearch_refine_options_t const refine = call_refine_options(call, arguments + receiver_arguments);
    call_plan(call, arguments[receiver_arguments + 2]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_sequence_refine(&(usearch_sequence_refine_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .plan = call->plan_id ? &call->plan.handle.sequence_refine : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_refine_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int SequenceRefinePlanFinish(WolframLibraryData library_data, mint arguments_count,
                                                MArgument* arguments, MArgument result) {
    call_t* const call = sequence_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_sequence_refine_plan_finish(&(usearch_sequence_refine_plan_finish_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.sequence_refine,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_health: the population audited, the entries a search reaches,
 *  and the depths. */
EXTERN_C DLLEXPORT int SequenceHealth(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = sequence_open(arguments);
    call->answer = answer_health;
    if (call_ready(call))
        usearch_sequence_health(&(usearch_sequence_health_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .reachability = &call->reachability,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_search: the bytes of every query end to end → their @c count + 1
 *  offsets → @c wanted → @c arena → @c expansionFactor → @c traversal → the found keys, their
 *  scores, and one count per query. */
EXTERN_C DLLEXPORT int SequenceSearch(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = sequence_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_tape_view_t queries = {0};
    size_t wanted = 0, expansion = 0;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    call_bytes_tape(library_data, call, verb[0], &queries);
    call_offsets(library_data, call, verb[1], &queries, &call->count);
    call_size(call, verb[2], &wanted, "wanted must not be negative");
    usearch_arena_t const arena = call_arena(call, verb[3]);
    call_size(call, verb[4], &expansion, "expansion factor must not be negative");
    call_traversal(library_data, call, verb[5], &traversal);
    call->answer = answer_matches;
    if (call_ready(call))
        usearch_sequence_search(&(usearch_sequence_search_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .queries = queries,
            .matches = &call->matches,
            .arena = arena,
            .wanted = wanted,
            .expansion_factor = expansion,
            .traversal = traversal,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_count: @c key → the entries stored under it, with their extent. */
EXTERN_C DLLEXPORT int SequenceCount(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t* const call = sequence_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    call->answer = answer_sequence_count;
    if (call_ready(call))
        usearch_sequence_count(&(usearch_sequence_count_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .key_stats = &call->tally.sequence,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_get: @c key → @c arena → every sequence stored under the key, its
 *  bytes and lengths. */
EXTERN_C DLLEXPORT int SequenceGet(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = sequence_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments + 1]);
    call->answer = answer_sequences;
    if (call_ready(call))
        usearch_sequence_get(&(usearch_sequence_get_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .sequences = &call->tape,
            .written = &call->written,
            .arena = arena,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_reshard: @c sources → @c targets → @c expansionFactor →
 *  @c memoryLimit → @c seed → @c plan → the report of the closing rounds. The receiver's index
 *  only names the call's family. */
EXTERN_C DLLEXPORT int SequenceReshard(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    call_t* const call = sequence_open(arguments);
    MArgument const* const listed = arguments + receiver_arguments;
    void const *sources = NULL, *targets = NULL;
    size_t sources_count = 0, targets_count = 0;
    call_handles(library_data, call, listed[0], &sequences, &sources, &sources_count);
    call_handles(library_data, call, listed[1], &sequences, &targets, &targets_count);
    usearch_refine_options_t const refine = call_refine_options(call, listed + 2);
    uint64_t const seed = (uint64_t)MArgument_getInteger(listed[4]);
    call_plan(call, listed[5]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_sequence_reshard(&(usearch_sequence_reshard_t) {
            .sources_count = sources_count,
            .sources = (usearch_sequence_index_t const*)sources,
            .targets_count = targets_count,
            .targets = (usearch_sequence_index_t const*)targets,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .seed = seed,
            .plan = call->plan_id ? &call->plan.handle.sequence_reshard : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_reshard_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int SequenceReshardPlanFinish(WolframLibraryData library_data, mint arguments_count,
                                                 MArgument* arguments, MArgument result) {
    call_t* const call = sequence_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_sequence_reshard_plan_finish(&(usearch_sequence_reshard_plan_finish_t) {
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.sequence_reshard,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_save: @c path, whose extension picks the layout. */
EXTERN_C DLLEXPORT int SequenceSave(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = sequence_open(arguments);
    usearch_bytes_view_t const path = call_text(library_data, call, arguments[receiver_arguments]);
    if (call_ready(call))
        usearch_sequence_save(&(usearch_sequence_save_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .path = path,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_save_buffer: @c arena → the @c UnsignedInteger8 monolith. */
EXTERN_C DLLEXPORT int SequenceSaveBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    call_t* const call = sequence_open(arguments);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments]);
    call->answer = answer_save_buffer;
    if (call_ready(call))
        usearch_sequence_save_buffer(&(usearch_sequence_save_buffer_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .monolith = &call->monolith,
            .arena = arena,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_sequence_describe: the frozen metadata, counters, classes and shard count. */
EXTERN_C DLLEXPORT int SequenceDescribe(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = sequence_open(arguments);
    call->answer = answer_sequence_describe;
    if (call_ready(call))
        usearch_sequence_describe(&(usearch_sequence_describe_t) {
            .index = (usearch_sequence_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .description = &call->description.sequence,
        });
    return call_answer(library_data, call, result);
}

#pragma endregion Sequence Index Verbs

#pragma region Shapes Index Verbs

/** Opens a call on a shapes index. */
static call_t* shapes_open(MArgument const* arguments) {
    return call_open(arguments, &shapes, "the ShapesIndex was released or never constructed");
}

/** Calls @c usearch_shapes_reserve: @c entries, the desired total including the current size →
 *  @c points, the points they hold in all. */
EXTERN_C DLLEXPORT int ShapesReserve(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t* const call = shapes_open(arguments);
    size_t entries = 0, extra = 0;
    call_size(call, arguments[receiver_arguments], &entries, "entries must not be negative");
    call_size(call, arguments[receiver_arguments + 1], &extra, "points must not be negative");
    if (call_ready(call))
        usearch_shapes_reserve(&(usearch_shapes_reserve_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .entries = entries,
            .points = extra,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_clear, keeping the reserved capacity and frozen settings. */
EXTERN_C DLLEXPORT int ShapesClear(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = shapes_open(arguments);
    if (call_ready(call))
        usearch_shapes_clear(&(usearch_shapes_clear_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_insert: @c keys → the points of every shape end to end, a three-column
 *  real matrix → their @c count + 1 point offsets → @c expansionFactor. */
EXTERN_C DLLEXPORT int ShapesInsert(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = shapes_open(arguments);
    usearch_key_t const* keys = NULL;
    usearch_tape_view_t tape = {0};
    size_t keys_count = 0, entries = 0, expansion = 0;
    call_keys(library_data, call, arguments[receiver_arguments], &keys, &keys_count);
    call_points_tape(library_data, call, arguments[receiver_arguments + 1], &tape);
    call_offsets(library_data, call, arguments[receiver_arguments + 2], &tape, &entries);
    call_size(call, arguments[receiver_arguments + 3], &expansion, "expansion factor must not be negative");
    // The call has one batch axis, so only this side sees the keys and the tape disagree on it.
    if (call_ready(call) && keys_count != entries)
        call_refuse(call, usearch_status_shape_mismatch_k, "there must be one key per shape");
    if (call_ready(call))
        usearch_shapes_insert(&(usearch_shapes_insert_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = entries,
            .keys = {keys, sizeof(usearch_key_t)},
            .shapes = tape,
            .expansion_factor = expansion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_refine: @c expansionFactor → @c memoryLimit → @c plan, zero running one
 *  round → the refinement's report. */
EXTERN_C DLLEXPORT int ShapesRefine(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = shapes_open(arguments);
    usearch_refine_options_t const refine = call_refine_options(call, arguments + receiver_arguments);
    call_plan(call, arguments[receiver_arguments + 2]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_shapes_refine(&(usearch_shapes_refine_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .plan = call->plan_id ? &call->plan.handle.shapes_refine : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_refine_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int ShapesRefinePlanFinish(WolframLibraryData library_data, mint arguments_count,
                                              MArgument* arguments, MArgument result) {
    call_t* const call = shapes_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_shapes_refine_plan_finish(&(usearch_shapes_refine_plan_finish_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.shapes_refine,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_health: the population audited, the entries a search reaches,
 *  and the depths. */
EXTERN_C DLLEXPORT int ShapesHealth(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = shapes_open(arguments);
    call->answer = answer_health;
    if (call_ready(call))
        usearch_shapes_health(&(usearch_shapes_health_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .reachability = &call->reachability,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_search: the points of every query end to end, a three-column real matrix
 *  → their @c count + 1 offsets → @c wanted → @c arena → @c expansionFactor → @c traversal → the
 *  found keys, their scores, and one count per query. */
EXTERN_C DLLEXPORT int ShapesSearch(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                    MArgument result) {
    call_t* const call = shapes_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_tape_view_t queries = {0};
    size_t wanted = 0, expansion = 0;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    call_points_tape(library_data, call, verb[0], &queries);
    call_offsets(library_data, call, verb[1], &queries, &call->count);
    call_size(call, verb[2], &wanted, "wanted must not be negative");
    usearch_arena_t const arena = call_arena(call, verb[3]);
    call_size(call, verb[4], &expansion, "expansion factor must not be negative");
    call_traversal(library_data, call, verb[5], &traversal);
    call->answer = answer_matches;
    if (call_ready(call))
        usearch_shapes_search(&(usearch_shapes_search_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .queries = queries,
            .matches = &call->matches,
            .arena = arena,
            .wanted = wanted,
            .expansion_factor = expansion,
            .traversal = traversal,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_count: @c key → the entries stored under it, with their extent. */
EXTERN_C DLLEXPORT int ShapesCount(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                   MArgument result) {
    call_t* const call = shapes_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    call->answer = answer_shapes_count;
    if (call_ready(call))
        usearch_shapes_count(&(usearch_shapes_count_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .key_stats = &call->tally.shapes,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_get: @c key → @c arena → every shape stored under the key, as @c f64
 *  points and counts. */
EXTERN_C DLLEXPORT int ShapesGet(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                 MArgument result) {
    call_t* const call = shapes_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments + 1]);
    call->answer = answer_shapes;
    if (call_ready(call))
        usearch_shapes_get(&(usearch_shapes_get_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .shapes = &call->tape,
            .written = &call->written,
            .arena = arena,
            .outputs_scalar = usearch_scalar_f64_k,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_reshard: @c sources → @c targets → @c expansionFactor →
 *  @c memoryLimit → @c seed → @c plan → the report of the closing rounds. The receiver's index
 *  only names the call's family. */
EXTERN_C DLLEXPORT int ShapesReshard(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t* const call = shapes_open(arguments);
    MArgument const* const listed = arguments + receiver_arguments;
    void const *sources = NULL, *targets = NULL;
    size_t sources_count = 0, targets_count = 0;
    call_handles(library_data, call, listed[0], &shapes, &sources, &sources_count);
    call_handles(library_data, call, listed[1], &shapes, &targets, &targets_count);
    usearch_refine_options_t const refine = call_refine_options(call, listed + 2);
    uint64_t const seed = (uint64_t)MArgument_getInteger(listed[4]);
    call_plan(call, listed[5]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_shapes_reshard(&(usearch_shapes_reshard_t) {
            .sources_count = sources_count,
            .sources = (usearch_shapes_index_t const*)sources,
            .targets_count = targets_count,
            .targets = (usearch_shapes_index_t const*)targets,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .seed = seed,
            .plan = call->plan_id ? &call->plan.handle.shapes_reshard : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_reshard_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int ShapesReshardPlanFinish(WolframLibraryData library_data, mint arguments_count,
                                               MArgument* arguments, MArgument result) {
    call_t* const call = shapes_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_shapes_reshard_plan_finish(&(usearch_shapes_reshard_plan_finish_t) {
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.shapes_reshard,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_save: @c path, whose extension picks the layout. */
EXTERN_C DLLEXPORT int ShapesSave(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                  MArgument result) {
    call_t* const call = shapes_open(arguments);
    usearch_bytes_view_t const path = call_text(library_data, call, arguments[receiver_arguments]);
    if (call_ready(call))
        usearch_shapes_save(&(usearch_shapes_save_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .path = path,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_save_buffer: @c arena → the @c UnsignedInteger8 monolith. */
EXTERN_C DLLEXPORT int ShapesSaveBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = shapes_open(arguments);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments]);
    call->answer = answer_save_buffer;
    if (call_ready(call))
        usearch_shapes_save_buffer(&(usearch_shapes_save_buffer_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .monolith = &call->monolith,
            .arena = arena,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_shapes_describe: the frozen metadata, counters, classes and shard count. */
EXTERN_C DLLEXPORT int ShapesDescribe(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = shapes_open(arguments);
    call->answer = answer_shapes_describe;
    if (call_ready(call))
        usearch_shapes_describe(&(usearch_shapes_describe_t) {
            .index = (usearch_shapes_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .description = &call->description.shapes,
        });
    return call_answer(library_data, call, result);
}

#pragma endregion Shapes Index Verbs

#pragma region Geospatial Index Verbs

/** Opens a call on a geospatial index. */
static call_t* geospatial_open(MArgument const* arguments) {
    return call_open(arguments, &geospatials, "the GeospatialIndex was released or never constructed");
}

/** Calls @c usearch_geospatial_reserve: @c entries, the desired total with the current size. */
EXTERN_C DLLEXPORT int GeospatialReserve(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    call_t* const call = geospatial_open(arguments);
    size_t entries = 0;
    call_size(call, arguments[receiver_arguments], &entries, "entries must not be negative");
    if (call_ready(call))
        usearch_geospatial_reserve(&(usearch_geospatial_reserve_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .entries = entries,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_clear, keeping the reserved capacity and frozen settings. */
EXTERN_C DLLEXPORT int GeospatialClear(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    call_t* const call = geospatial_open(arguments);
    if (call_ready(call))
        usearch_geospatial_clear(&(usearch_geospatial_clear_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_insert: @c keys → the observations, an eleven-column real matrix →
 *  its recorded mask → @c expansionFactor. */
EXTERN_C DLLEXPORT int GeospatialInsert(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = geospatial_open(arguments);
    usearch_key_t const* keys = NULL;
    usearch_geospatial_observation_t const* observations = NULL;
    size_t keys_count = 0, entries = 0, expansion = 0;
    call_keys(library_data, call, arguments[receiver_arguments], &keys, &keys_count);
    call_observations(library_data, call, arguments[receiver_arguments + 1], arguments[receiver_arguments + 2],
                      &observations, &entries);
    call_size(call, arguments[receiver_arguments + 3], &expansion, "expansion factor must not be negative");
    // The call has one batch axis, so only this side sees keys and observations disagree on it.
    if (call_ready(call) && keys_count != entries)
        call_refuse(call, usearch_status_shape_mismatch_k, "there must be one key per observation");
    if (call_ready(call))
        usearch_geospatial_insert(&(usearch_geospatial_insert_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = entries,
            .keys = {keys, sizeof(usearch_key_t)},
            .observations = observations,
            .expansion_factor = expansion,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_refine: @c expansionFactor → @c memoryLimit → @c plan, zero running
 *  one round → the refinement's report. */
EXTERN_C DLLEXPORT int GeospatialRefine(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = geospatial_open(arguments);
    usearch_refine_options_t const refine = call_refine_options(call, arguments + receiver_arguments);
    call_plan(call, arguments[receiver_arguments + 2]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_geospatial_refine(&(usearch_geospatial_refine_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .plan = call->plan_id ? &call->plan.handle.geospatial_refine : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_refine_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int GeospatialRefinePlanFinish(WolframLibraryData library_data, mint arguments_count,
                                                  MArgument* arguments, MArgument result) {
    call_t* const call = geospatial_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_geospatial_refine_plan_finish(&(usearch_geospatial_refine_plan_finish_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.geospatial_refine,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_health: the population audited, the entries a search reaches,
 *  and the depths. */
EXTERN_C DLLEXPORT int GeospatialHealth(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = geospatial_open(arguments);
    call->answer = answer_health;
    if (call_ready(call))
        usearch_geospatial_health(&(usearch_geospatial_health_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .reachability = &call->reachability,
        });
    return call_answer(library_data, call, result);
}

/** Reads the relation, as its C value, and the tolerance a geospatial query narrows by. */
static usearch_geospatial_tolerance_t call_tolerance(call_t* call, MArgument const* arguments,
                                                     usearch_geospatial_relation_t* relation) {
    *relation = (usearch_geospatial_relation_t)MArgument_getInteger(arguments[0]);
    return (usearch_geospatial_tolerance_t) {MArgument_getReal(arguments[1]), MArgument_getReal(arguments[2])};
}

/** Calls @c usearch_geospatial_search: the query observations → their recorded mask → @c relation →
 *  @c metres → @c seconds → @c wanted → @c arena → @c expansionFactor → @c traversal → the found
 *  keys, their chords in metres, and one count per query. */
EXTERN_C DLLEXPORT int GeospatialSearch(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = geospatial_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_geospatial_observation_t const* queries = NULL;
    usearch_geospatial_relation_t relation;
    size_t wanted = 0, expansion = 0;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    call_observations(library_data, call, verb[0], verb[1], &queries, &call->count);
    usearch_geospatial_tolerance_t const tolerance = call_tolerance(call, verb + 2, &relation);
    call_size(call, verb[5], &wanted, "wanted must not be negative");
    usearch_arena_t const arena = call_arena(call, verb[6]);
    call_size(call, verb[7], &expansion, "expansion factor must not be negative");
    call_traversal(library_data, call, verb[8], &traversal);
    call->answer = answer_matches;
    if (call_ready(call))
        usearch_geospatial_search(&(usearch_geospatial_search_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .queries = queries,
            .relation = relation,
            .tolerance = tolerance,
            .matches = &call->matches,
            .arena = arena,
            .wanted = wanted,
            .expansion_factor = expansion,
            .traversal = traversal,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_within: the query observations → their recorded mask → @c relation →
 *  @c metres → @c seconds → @c arena → @c traversal → offsets, the keys every query admits, and
 *  their chords in metres. */
EXTERN_C DLLEXPORT int GeospatialWithin(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                        MArgument result) {
    call_t* const call = geospatial_open(arguments);
    MArgument const* const verb = arguments + receiver_arguments;
    usearch_geospatial_observation_t const* queries = NULL;
    usearch_geospatial_relation_t relation;
    usearch_traversal_t traversal = usearch_traversal_walk_k;
    call_observations(library_data, call, verb[0], verb[1], &queries, &call->count);
    usearch_geospatial_tolerance_t const tolerance = call_tolerance(call, verb + 2, &relation);
    usearch_arena_t const arena = call_arena(call, verb[5]);
    call_traversal(library_data, call, verb[6], &traversal);
    call->answer = answer_tape;
    if (call_ready(call))
        usearch_geospatial_within(&(usearch_geospatial_within_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .count = call->count,
            .queries = queries,
            .relation = relation,
            .tolerance = tolerance,
            .matches = &call->products,
            .written = &call->written,
            .arena = arena,
            .traversal = traversal,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_count: @c key → the entries stored under it. */
EXTERN_C DLLEXPORT int GeospatialCount(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                       MArgument result) {
    call_t* const call = geospatial_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    call->answer = answer_geospatial_count;
    if (call_ready(call))
        usearch_geospatial_count(&(usearch_geospatial_count_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .key_stats = &call->tally.geospatial,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_get: @c key → @c arena → every observation stored under the key,
 *  beside its recorded mask. */
EXTERN_C DLLEXPORT int GeospatialGet(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                     MArgument result) {
    call_t* const call = geospatial_open(arguments);
    usearch_key_t key;
    call_key(library_data, call, arguments[receiver_arguments], &key);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments + 1]);
    call->answer = answer_observations;
    if (call_ready(call))
        usearch_geospatial_get(&(usearch_geospatial_get_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .key = key,
            .observations = &call->observations,
            .written = &call->written,
            .arena = arena,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_reshard: @c sources → @c targets → @c expansionFactor →
 *  @c memoryLimit → @c seed → @c plan → the report of the closing rounds. The receiver's index only
 *  names the call's family. */
EXTERN_C DLLEXPORT int GeospatialReshard(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                         MArgument result) {
    call_t* const call = geospatial_open(arguments);
    MArgument const* const listed = arguments + receiver_arguments;
    void const *sources = NULL, *targets = NULL;
    size_t sources_count = 0, targets_count = 0;
    call_handles(library_data, call, listed[0], &geospatials, &sources, &sources_count);
    call_handles(library_data, call, listed[1], &geospatials, &targets, &targets_count);
    usearch_refine_options_t const refine = call_refine_options(call, listed + 2);
    uint64_t const seed = (uint64_t)MArgument_getInteger(listed[4]);
    call_plan(call, listed[5]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_geospatial_reshard(&(usearch_geospatial_reshard_t) {
            .sources_count = sources_count,
            .sources = (usearch_geospatial_index_t const*)sources,
            .targets_count = targets_count,
            .targets = (usearch_geospatial_index_t const*)targets,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .report = &call->report,
            .refine = refine,
            .seed = seed,
            .plan = call->plan_id ? &call->plan.handle.geospatial_reshard : NULL,
            .phase = call->plan_id ? &call->plan.progress : NULL,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_reshard_plan_finish: @c plan → the report of every round it ran. */
EXTERN_C DLLEXPORT int GeospatialReshardPlanFinish(WolframLibraryData library_data, mint arguments_count,
                                                   MArgument* arguments, MArgument result) {
    call_t* const call = geospatial_open(arguments);
    call_take_plan(call, arguments[receiver_arguments]);
    call->answer = answer_refine;
    if (call_ready(call))
        usearch_geospatial_reshard_plan_finish(&(usearch_geospatial_reshard_plan_finish_t) {
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .plan = call->plan.handle.geospatial_reshard,
            .report = &call->report,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_save: @c path, whose extension picks the layout. */
EXTERN_C DLLEXPORT int GeospatialSave(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                      MArgument result) {
    call_t* const call = geospatial_open(arguments);
    usearch_bytes_view_t const path = call_text(library_data, call, arguments[receiver_arguments]);
    if (call_ready(call))
        usearch_geospatial_save(&(usearch_geospatial_save_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .path = path,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_save_buffer: @c arena → the @c UnsignedInteger8 monolith. */
EXTERN_C DLLEXPORT int GeospatialSaveBuffer(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                            MArgument result) {
    call_t* const call = geospatial_open(arguments);
    usearch_arena_t const arena = call_arena(call, arguments[receiver_arguments]);
    call->answer = answer_save_buffer;
    if (call_ready(call))
        usearch_geospatial_save_buffer(&(usearch_geospatial_save_buffer_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .monolith = &call->monolith,
            .arena = arena,
        });
    return call_answer(library_data, call, result);
}

/** Calls @c usearch_geospatial_describe: the frozen metadata, counters, classes and shard count. */
EXTERN_C DLLEXPORT int GeospatialDescribe(WolframLibraryData library_data, mint arguments_count, MArgument* arguments,
                                          MArgument result) {
    call_t* const call = geospatial_open(arguments);
    call->answer = answer_geospatial_describe;
    if (call_ready(call))
        usearch_geospatial_describe(&(usearch_geospatial_describe_t) {
            .index = (usearch_geospatial_index_t)call->index,
            .executor = call->executor,
            .outcome = &call->outcome,
            .completion = call->completion,
            .description = &call->description.geospatial,
        });
    return call_answer(library_data, call, result);
}

#pragma endregion Geospatial Index Verbs
