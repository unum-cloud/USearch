/**
 *  @file javascript/lib.h
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief Declarations shared by every USearch Node-API translation unit.
 *
 *  One addon, one file per header: @c types.c owns the vocabulary every modality shares,
 *  each @c <modality>_index.c owns its verbs, and @c lib.c owns the module glue alone.
 *
 *  @section javascript_batching Batching and Threading
 *
 *  Every verb is a batch and the engine fans it out itself, so this layer spawns nothing: it
 *  borrows the caller's @c TypedArray storage, hands the engine one strided matrix, and views
 *  the outputs the engine lands in the caller's @c Arena without copying them. The workers live
 *  in an @c Executor the caller constructs and passes in, which is what lets several indexes
 *  share one pool. A verb handed no executor runs on the calling thread, as the C interface's
 *  @c NULL executor does.
 *
 *  Every verb takes the C @c completion beside its executor. A blocking call answers its result; a
 *  deferred one answers a @c CompiledDeferred holding the outcome and strong references to every
 *  JavaScript value the library borrows until the call is final, and borrows its arena as a
 *  blocking call does. Its @c wait blocks the calling thread in @c usearch_outcome_wait, and its
 *  @c settled hands that wait to the libuv pool through @c napi_async_work, the one thread this
 *  layer ever borrows, and only while a caller awaits.
 *
 *  A JavaScript callback runs only on the thread owning the context, which an executor's threads
 *  never are, and a thread-safe function would queue onto a loop blocked inside the verb. So only
 *  @c attend takes one, its value lookup, and only on the calling thread, where the library calls
 *  it in place. A predicate or a progress handler is not exposed at all; lazy pulling through
 *  @c usearch_dense_search_stream_next stays the filtered path.
 *
 *  @section javascript_keys Identities
 *
 *  An identity is sixteen opaque bytes, most significant first, and crosses as a
 *  @c Uint8Array of that many per key. An integer identity parks in the trailing eight bytes
 *  with the leading eight zeroed, which is what a @c BigUint64Array spells on the way in.
 */
#ifndef UNUM_USEARCH_JAVASCRIPT_LIB_H
#define UNUM_USEARCH_JAVASCRIPT_LIB_H

#include <stdbool.h> // `bool`
#include <stddef.h>  // `size_t`
#include <stdint.h>  // `uint8_t`, `uint64_t`

#include <node_api.h>

#include "usearch/dense_index.h"
#include "usearch/geospatial_index.h"
#include "usearch/sequence_index.h"
#include "usearch/shapes_index.h"
#include "usearch/types.h"

#pragma region Vocabulary

/** Compile-time constants of the binding layer. */
enum {

    /** Bytes an option name may occupy, terminator included. */
    usearch_js_name_capacity_k = 64,

    /** Bytes a formatted diagnostic may occupy, terminator included. */
    usearch_js_message_capacity_k = 256,

    /** Bytes one identity occupies, on the wire and in the engine alike. */
    usearch_js_key_bytes_k = 16,

    /** Heap blocks one call may own, freed together once it is final. */
    usearch_js_owned_k = 4,
};

/** Element count of a statically-sized table, which every naming table asks for. */
#define usearch_js_table_size_(table) (sizeof(table) / sizeof((table)[0]))

/** Everything the addon owns per environment: the constructors its verbs type-check
 *  against, reached through @c napi_get_instance_data rather than a namespace-scope global. */
typedef struct usearch_js_addon_t {

    /** The @c Executor constructor, kept alive for @c napi_instanceof. */
    napi_ref executor_class;

    /** The @c CompiledIndex constructor, which every dense lifecycle verb wraps its handle in. */
    napi_ref index_class;

    /** The @c CompiledSequenceIndex constructor, wrapping each handle a sequence verb opens. */
    napi_ref sequence_index_class;

    /** The @c CompiledShapesIndex constructor, wrapping each handle a shapes verb opens. */
    napi_ref shapes_index_class;

    /** The @c CompiledGeospatialIndex constructor, wrapping each handle a geospatial verb opens. */
    napi_ref geospatial_index_class;

    /** The @c Arena constructor, kept alive for @c napi_instanceof. */
    napi_ref arena_class;

    /** The @c CompiledDeferred constructor, which every deferred submission instantiates. */
    napi_ref deferred_class;

    /** The token only this addon holds, which a direct construction from JavaScript lacks. */
    napi_ref seal;
} usearch_js_addon_t;

struct usearch_js_call_t;

/** The workers a fanned-out verb runs on, owned apart from any index so one pool serves many. */
typedef struct usearch_js_executor_t {

    /** The engine handle, freed by @c free or when the JavaScript wrapper is collected. */
    usearch_executor_t handle;

    /** Live GPU arenas it made, each of which must be freed before it is. */
    size_t arenas;
} usearch_js_executor_t;

/** Memory every output whose length scales with a call lands in, and the views handed over it. */
typedef struct usearch_js_arena_t {

    /** The engine handle, made with the wrapper and @c NULL once freed. */
    usearch_arena_t handle;

    /** The executor a GPU arena grows, fills and frees on, held until the arena is freed; @c NULL
     *  for host pages, which may outlive theirs. */
    usearch_js_executor_t* executor;

    /** The strong reference keeping @c executor's wrapper alive alongside it. */
    napi_ref executor_reference;
} usearch_js_arena_t;

/** An owned index handle, one per instance of a modality's compiled index class. */
typedef struct usearch_js_index_t {

    /** The engine handle of the modality its class wraps, freed by @c free or the collector. */
    void* handle;
} usearch_js_index_t;

/**
 *  @brief The paths a residency arm borrows for the length of one factory call.
 *
 *  The ABI reads the views contiguously, so the bytes they point at cannot ride inside them
 *  and trail them in the same block instead - one allocation, released as one.
 */
typedef struct usearch_js_paths_arena_t {

    /** The run the arm reads, or NULL when the caller named no paths. */
    usearch_bytes_view_t* views;

    /** Paths the arena holds; zero when the caller named none. */
    size_t count;
} usearch_js_paths_arena_t;

/** The JavaScript values a call reads or writes, in the order every verb lays them out. */
typedef enum usearch_js_slot_t {

    /** The @c CompiledIndex the verb runs on. */
    usearch_js_slot_receiver_k = 0,

    /** The @c Executor it runs on, absent for the calling thread. */
    usearch_js_slot_executor_k,

    /** The @c Arena its outputs land in. */
    usearch_js_slot_arena_k,

    /** The rows it reads: vectors, queries, a left-hand matrix, or a reshard's sources. */
    usearch_js_slot_rows_k,

    /** The identities it reads, or a reshard's targets. */
    usearch_js_slot_keys_k,

    /** The @c DenseTraversalStats reading its telemetry lands in. */
    usearch_js_slot_stats_k,

    /** The keys a sample looks around, beside the shown ones it reads through the keys slot. */
    usearch_js_slot_focus_k,

    /** The plan a refine or a reshard continues, which the library writes in place. */
    usearch_js_slot_plan_k,

    /** Slots in every call. */
    usearch_js_slots_k,
} usearch_js_slot_t;

/** Builds a verb's answer from a successful call's outputs; @c NULL with an exception pending. */
typedef napi_value (*usearch_js_finish_t)(napi_env env, struct usearch_js_call_t* call, napi_value const* slots);

/**
 *  @brief One call submitted to the library: its outcome, how its answer is built, what it holds.
 *
 *  Every modality extends it with the outputs its verbs land, keeping this header first, so one
 *  @c CompiledDeferred serves them all. A blocking call lives on the stack of its verb and holds
 *  nothing, since its arguments are reachable for as long as it runs; a deferred one lives on the
 *  heap and holds strong references to every slot until it is final and settled.
 */
typedef struct usearch_js_call_t {

    /** Where the library reports the call, final once its status leaves @c deferred. */
    usearch_outcome_t outcome;

    /** Builds the answer of a call that succeeded; @c NULL once the answer was built. */
    usearch_js_finish_t finish;

    /** Strong references to the slots a deferred call borrows; @c NULL for an empty slot. */
    napi_ref held[usearch_js_slots_k];

    /** Heap blocks the call reads or writes, freed once it is final. */
    void* owned[usearch_js_owned_k];

    /** Traversal telemetry, published into the stats slot once the call succeeded. */
    usearch_traversal_stats_t stats;
} usearch_js_call_t;

/**
 *  @brief An index call: the shared header, then every output an index verb of any modality lands
 *      before its answer.
 *
 *  The modalities share one layout, so one finisher serves the verbs whose outputs agree: a report,
 *  an audit, a ranked run, a ragged run, a sample, a cursor pull and a monolith.
 */
typedef struct usearch_js_index_call_t {

    /** The outcome, the answer builder, the held values and the telemetry every modality shares. */
    usearch_js_call_t call;

    /** The audit @c health reports. */
    usearch_reachability_t reachability;

    /** What a refine or a reshard did to the graph. */
    usearch_refine_report_t report;

    /** The report @c describe fills, in the modality's own shape. */
    union {
        usearch_dense_index_description_t dense;
        usearch_sequence_index_description_t sequence;
        usearch_shapes_index_description_t shapes;
        usearch_geospatial_index_description_t geospatial;
    } description;

    /** The extent of a key's entries a basic modality's @c count reports. */
    union {
        usearch_sequence_key_stats_t sequence;
        usearch_shapes_key_stats_t shapes;
        usearch_geospatial_key_stats_t geospatial;
    } key_stats;

    /** The rows on a verb's batch axis. */
    size_t count;

    /** Entries under a key or a cursor pull wrote, nodes a refinement changed, or the rows or
     *  pairs a call wrote. */
    size_t written;

    /** Pairs a product scored before its bound cut any, or keys a late interaction wrote. */
    size_t required;

    /** The ranked run a search or a stream pull wrote into its arena. */
    usearch_matches_view_t matches;

    /** The pairs a product, the keys a late-interaction scoring, or the entries a containment walk
     *  wrote into its arena. */
    usearch_matches_tape_view_t products;

    /** The rows a retrieval, a cursor pull or an attention wrote into its arena. */
    usearch_rows_view_t vectors;

    /** The sequences or shapes a basic retrieval wrote into its arena. */
    usearch_tape_view_t tape;

    /** The observations a geospatial retrieval wrote into its arena. */
    usearch_geospatial_observation_t const* observations;

    /** The identities a cursor pull or a sample wrote into its arena. */
    usearch_keys_view_t keys;

    /** Which of its key's entries each draw of a sample is, inside its arena. */
    size_t const* entry_ordinals;

    /** The keys each of a cursor pull's entries links to, inside its arena. */
    usearch_keys_tape_view_t links;

    /** A monolith inside its arena. */
    usearch_bytes_view_t monolith;

    /** The kind the rows were asked in, which picks their element type even with none written. */
    usearch_scalar_t scalar;

    /** The placements a reshard lends, over the ordinals and paths in the call's owned blocks. */
    usearch_ram_options_t ram;
    usearch_vram_options_t vram;
    usearch_nvme_options_t nvme;
} usearch_js_index_call_t;

/** A plan's storage, which the library writes in place: its handle, and where it stopped. */
typedef struct usearch_js_plan_t {

    /** The refine plan a refine opened; @c NULL before the first call and once finished. */
    union {
        usearch_dense_refine_plan_t dense;
        usearch_sequence_refine_plan_t sequence;
        usearch_shapes_refine_plan_t shapes;
        usearch_geospatial_refine_plan_t geospatial;
    } refine;

    /** The reshard plan a reshard opened; @c NULL before the first call and once finished. */
    union {
        usearch_dense_reshard_plan_t dense;
        usearch_sequence_reshard_plan_t sequence;
        usearch_shapes_reshard_plan_t shapes;
        usearch_geospatial_reshard_plan_t geospatial;
    } reshard;

    /** Where the last call stopped. */
    usearch_phase_progress_t progress;
} usearch_js_plan_t;

/** Where a load reads its checkpoint from. */
typedef enum usearch_js_source_t {
    usearch_js_source_path_k,
    usearch_js_source_buffer_k,
} usearch_js_source_t;

/** A cursor walk or a resumable search, and the index and executor every pull names again. */
typedef struct usearch_js_walk_t {

    /** The engine cursor or stream, @c NULL once released. */
    void* handle;

    /** A strong reference to the compiled index, so the index outlives the walk it borrows. */
    napi_ref index;

    /** A strong reference to the @c Executor every pull runs on; @c NULL for the calling thread. */
    napi_ref executor;
} usearch_js_walk_t;

#pragma endregion Vocabulary

#pragma region Argument Parsing

/** Throws @p message with the canonical name of @p status as its @c code; always returns false, so
 *  a caller returns straight through. */
bool usearch_js_throw(napi_env env, usearch_status_t status, char const* message);

/** Throws an @c invalid-argument error saying what @p field expected; always returns false. */
bool usearch_js_throw_field(napi_env env, char const* field, char const* expectation);

/**
 *  @brief Throws the mapped JavaScript error when @p outcome carries a failure.
 *  @return @c true when the call succeeded, @c false with an exception pending otherwise.
 */
bool usearch_js_succeeded(napi_env env, usearch_outcome_t const* outcome);

/** Whether @p value is @c undefined or @c null, which every optional argument derives from. */
bool usearch_js_value_absent(napi_env env, napi_value value);

/** Reads a named property of an @b optional options object; an absent one reads @c undefined. */
napi_value usearch_js_property(napi_env env, napi_value object, char const* name);

/** Reads a non-negative integer from a number or a @c BigInt; absent arguments read as zero. */
bool usearch_js_size_from_value(napi_env env, napi_value value, char const* field, size_t* result);

/** Reads a finite real; absent arguments read as zero, which every knob derives from. */
bool usearch_js_double_from_value(napi_env env, napi_value value, char const* field, double* result);

/**
 *  @brief Copies a UTF-8 string into a @c malloc-ed block, so a deferred call can borrow it.
 *  @param[out] view The bytes, without the terminator the block also carries.
 *  @return The block the caller frees, or @c NULL with an exception pending.
 */
char* usearch_js_string_from_value(napi_env env, napi_value value, char const* field, usearch_bytes_view_t* view);

/** Reads the C @c completion every verb takes beside its executor. */
bool usearch_js_completion_from_value(napi_env env, napi_value value, usearch_completion_t* result);

/** Points @p stats at @p call's telemetry when the caller passed a reading, nowhere otherwise. */
bool usearch_js_stats_from_value(napi_env env, napi_value sink, usearch_js_call_t* call,
                                 usearch_traversal_stats_t** stats);

/** Reads a metric kind from its canonical spelling, through the C name tables. */
bool usearch_js_metric_from_value(napi_env env, napi_value value, char const* field, usearch_dense_metric_t* result);

/** Reads a sequence scoring kind from its canonical spelling, through the C name tables. */
bool usearch_js_sequence_metric_from_value(napi_env env, napi_value value, char const* field,
                                           usearch_sequence_metric_t* result);

/** Reads a shape scoring kind from its canonical spelling, through the C name tables. */
bool usearch_js_shapes_metric_from_value(napi_env env, napi_value value, char const* field,
                                         usearch_shapes_metric_t* result);

/** Reads a CPU class ceiling from its canonical spelling through the C name tables; absent sets
 *  none. */
bool usearch_js_cpu_class_from_value(napi_env env, napi_value value, usearch_cpu_class_t* result);

/** Reads a GPU class ceiling from its canonical spelling through the C name tables; absent sets
 *  none. */
bool usearch_js_gpu_class_from_value(napi_env env, napi_value value, usearch_gpu_class_t* result);

/** Reads a traversal from its canonical spelling, through the C name tables; absent walks. */
bool usearch_js_traversal_from_value(napi_env env, napi_value value, char const* field, usearch_traversal_t* result);

/** Reads a scalar kind as the code @c usearch_js_scalars published for its canonical spelling. */
bool usearch_js_scalar_from_value(napi_env env, napi_value value, char const* field, usearch_scalar_t* result);

/** Maps every scalar kind's canonical spelling to its code once, through the C name tables. */
napi_value usearch_js_scalars(napi_env env);

/** Bytes a row of @p columns values of @p scalar takes, sub-byte packing included; else zero. */
size_t usearch_js_row_bytes(usearch_scalar_t scalar, size_t columns);

/**
 *  @brief Reads NUMA node IDs or CUDA ordinals from an array of numbers.
 *
 *  A residency arm borrows its array only for the length of one factory call, so the
 *  allocation lives no longer than the constructor that made it.
 *
 *  @param[out] ordinals Receives a @c free -able array; NULL when the value is absent or empty.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool usearch_js_ordinals_from_value(napi_env env, napi_value value, char const* field, int32_t** ordinals,
                                    size_t* count);

/**
 *  @brief Reads filesystem or transport identifiers from an array of strings.
 *
 *  A path arrives as text and the ABI reads bytes, so the encoded runs have to outlive the
 *  call; they trail the views in the same block, since the ABI reads the views contiguously.
 *
 *  @param[out] arena The run to hand the arm; release it with @c usearch_js_paths_release.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool usearch_js_paths_from_value(napi_env env, napi_value value, char const* field, usearch_js_paths_arena_t* arena);

/** Releases the encoded paths and the block they share. */
void usearch_js_paths_release(usearch_js_paths_arena_t* arena);

/**
 *  @brief Borrows the bytes of a numeric @c TypedArray, whatever its element type.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool usearch_js_bytes_from_value(napi_env env, napi_value value, char const* field, void** data, size_t* bytes);

/**
 *  @brief Borrows a run of rows as five values: the @c TypedArray, its rows, its columns, the bytes
 *      between rows, and the scalar kind - the shape every row argument crosses in.
 *  @param[out] count The rows the run holds, which every batch verb takes as its @c count.
 *  @return @c true on success, @c false with an exception pending otherwise.
 *
 *  The engine reads no further than its @c count says, so the run is checked against the bytes the
 *  array holds here, the one side that sees both.
 */
bool usearch_js_rows_from_value(napi_env env, napi_value const* arguments, char const* field, size_t* count,
                                usearch_rows_view_t* view);

/**
 *  @brief Borrows a ragged batch as three values: the elements' @c TypedArray, its Apache Arrow
 *      offsets as a @c Uint32Array or a @c BigUint64Array, and the elements' scalar kind.
 *  @param[in] scalars_per_element Scalars one element holds: one byte of a sequence, three
 *      coordinates of a point.
 *  @param[out] count The entries the tape holds, one fewer than its offsets.
 *  @return @c true on success, @c false with an exception pending otherwise.
 *
 *  The engine reads up to the last offset, so it is checked against the bytes the array holds here,
 *  the one side that sees both.
 */
bool usearch_js_tape_from_value(napi_env env, napi_value const* arguments, char const* field,
                                size_t scalars_per_element, size_t* count, usearch_tape_view_t* view);

/** The scalar kind a @c TypedArray's elements spell on their own; unknown for byte carriers. */
usearch_scalar_t usearch_js_scalar_from_typedarray(napi_typedarray_type type);

/**
 *  @brief Reads a run of identities: a @c Uint8Array of sixteen bytes per key, borrowed in place,
 *      or a @c BigUint64Array of integer ones, gathered into @p gathered.
 *  @param[out] keys The run the engine reads.
 *  @param[out] gathered A @c malloc-ed block when the keys had to be gathered, else @c NULL.
 *  @return @c true on success, @c false with an exception pending otherwise.
 */
bool usearch_js_keys_from_value(napi_env env, napi_value value, char const* field, usearch_key_t const** keys,
                                size_t* count, usearch_key_t** gathered);

/** Reads one identity, which the argument spells the same way a batch of one would. */
bool usearch_js_key_from_value(napi_env env, napi_value value, char const* field, usearch_key_t* key);

/**
 *  @brief Allocates an @c ArrayBuffer of @p bytes and reports where its storage begins.
 *  @return The buffer, or @c NULL with an exception pending.
 */
napi_value usearch_js_buffer(napi_env env, size_t bytes, void** data);

/** Wraps a count the engine reported, once its error has been checked. */
napi_value usearch_js_number(napi_env env, size_t value);

/** Wraps a static null-terminated string the engine owns. */
napi_value usearch_js_static_string(napi_env env, char const* text);

/** The @c undefined every void verb hands back. */
napi_value usearch_js_undefined(napi_env env);

#pragma endregion Argument Parsing

#pragma region Classes and Probes

/** Defines the @c Executor class; the module glue keeps the reference. */
napi_value usearch_js_executor_define(napi_env env);

/** Unwraps the @c executor argument every fanned-out verb takes; an absent one reads as @c NULL. */
bool usearch_js_executor_from_value(napi_env env, napi_value value, usearch_executor_t* handle);

/** Defines the @c Arena class; the module glue keeps the reference. */
napi_value usearch_js_arena_define(napi_env env);

/** Unwraps the @c arena a verb's outputs land in; a freed one reads as @c NULL, for the library to
 *  refuse. */
bool usearch_js_arena_from_value(napi_env env, napi_value value, usearch_arena_t* handle);

/**
 *  @brief Views @p bytes an arena holds as a @c TypedArray of @p type, without copying.
 *
 *  The backing @c ArrayBuffer references @p arena, so the collector keeps the memory for as long as
 *  any view of it lives; the view ends at the arena's next call, and reads device memory as is for
 *  a VRAM arena, which @c Arena.describe names.
 *
 *  @return The view, or @c NULL with an exception pending.
 */
napi_value usearch_js_arena_view(napi_env env, napi_value arena, napi_typedarray_type type, void* data, size_t bytes,
                                 size_t elements);

/** Defines the @c CompiledDeferred class; the module glue keeps the reference. */
napi_value usearch_js_deferred_define(napi_env env);

/** Unwraps an instance of the class @p reference holds, or throws naming @p field and answers
 *  @c NULL. */
void* usearch_js_unwrap_instance(napi_env env, napi_value value, napi_ref reference, char const* field,
                                 char const* expectation);

/** The token constructors check, so only this addon instantiates its own classes. */
bool usearch_js_sealed(napi_env env, napi_callback_info info, napi_value* self);

/** Instantiates one of the addon's classes, handing its constructor the seal. */
napi_value usearch_js_instantiate(napi_env env, napi_ref constructor);

/**
 *  @brief Opens a call: @p local when blocking, or a zeroed heap block of @p bytes when deferred.
 *  @return The call, or @c NULL with an exception pending.
 */
usearch_js_call_t* usearch_js_call_open(napi_env env, usearch_completion_t completion, usearch_js_call_t* local,
                                        size_t bytes);

/**
 *  @brief Readies a deferred call: holds @p slots and wraps the call in a @c CompiledDeferred.
 *  @param[out] deferred The wrapper the verb answers once submitted; untouched for a blocking call.
 *  @return @c true on success, @c false with an exception pending and the call released otherwise.
 *
 *  Runs before the submission, so a failure here leaves nothing in flight.
 */
bool usearch_js_call_hold(napi_env env, usearch_js_call_t* call, usearch_js_call_t const* local,
                          napi_value const* slots, napi_value* deferred);

/** Answers a submitted call: a blocking one's finished result, a deferred one's wrapper. */
napi_value usearch_js_call_answer(napi_env env, usearch_js_call_t* call, usearch_js_call_t const* local,
                                  napi_value const* slots, napi_value deferred);

/** Releases a call that never reached the library, after its arguments failed to parse. */
void usearch_js_call_abandon(usearch_js_call_t* call, usearch_js_call_t const* local);

/** Defines the @c CompiledIndex class, its cursors and its search streams included. */
napi_value usearch_js_index_define(napi_env env);

/** Gives @p holder, an object or a prototype, a @c [Symbol.dispose] calling its @p method, as
 *  @c using expects. */
bool usearch_js_disposable(napi_env env, napi_value holder, char const* method);

/** Answers with the library's semantic version. */
napi_value usearch_js_version(napi_env env, napi_callback_info info);

/** Answers with the kernel families this build carries, whatever the machine offers. */
napi_value usearch_js_capabilities_compiled(napi_env env, napi_callback_info info);

/** Answers with the kernel families this machine can run, and dispatch picks from. */
napi_value usearch_js_capabilities_available(napi_env env, napi_callback_info info);

#pragma endregion Classes and Probes

#pragma region Index Verbs

/** Defines the @c CompiledSequenceIndex class and its cursors. */
napi_value usearch_js_sequence_index_define(napi_env env);

/** Defines the @c CompiledShapesIndex class and its cursors. */
napi_value usearch_js_shapes_index_define(napi_env env);

/** Defines the @c CompiledGeospatialIndex class and its cursors. */
napi_value usearch_js_geospatial_index_define(napi_env env);

/** Reads the receiver and @p wanted arguments beside it, missing ones reading as @c undefined; a
 *  freed handle throws @c uninitialized. */
usearch_js_index_t* usearch_js_index_self(napi_env env, napi_callback_info info, size_t wanted, napi_value* arguments,
                                          napi_value* self);

/** Reads @p wanted arguments of a static verb, which has no receiver to unwrap. */
bool usearch_js_static_arguments(napi_env env, napi_callback_info info, size_t wanted, napi_value* arguments);

/**
 *  @brief Opens the call every instance verb runs through, reading the executor and the completion.
 *  @return The call, or @c NULL with an exception pending.
 */
usearch_js_index_call_t* usearch_js_index_open(napi_env env, napi_value const* arguments,
                                               usearch_js_index_call_t* local, usearch_executor_t* executor,
                                               usearch_completion_t* completion);

/** Reads the @c ArrayBuffer a plan lives in; an absent one reads as @c NULL. */
bool usearch_js_plan_from_value(napi_env env, napi_value value, usearch_js_plan_t** plan);

/** Reads the length of an array of indexes, or throws naming @p field. */
bool usearch_js_handles_count(napi_env env, napi_value value, char const* field, uint32_t* count);

/** Unwraps every instance of the class @p index_class holds in the array @p value into @p handles;
 *  a freed one reads as @c NULL, for the library to refuse. */
bool usearch_js_handles_from_value(napi_env env, napi_value value, char const* field, uint32_t count,
                                   napi_ref index_class, void** handles);

/** Views a run of keys in place, gathering only integer keys into block @p block of @p call; an
 *  absent run reads as none. */
bool usearch_js_optional_keys(napi_env env, napi_value value, char const* field, usearch_js_call_t* call, size_t block,
                              usearch_keys_view_t* view, size_t* count);

/** Builds an object of @p count named values, or answers @c NULL with an exception pending. */
napi_value usearch_js_object(napi_env env, size_t count, char const* const* names, napi_value const* values);

/** Views a ranked run of @p count queries inside @p arena as @c {keys, scores, counts, columns}. */
napi_value usearch_js_matches_object(napi_env env, napi_value arena, usearch_matches_view_t const* matches,
                                     size_t count);

/** The @c TypedArray a kind reads back as; kinds JavaScript spells no element for stay bytes. */
napi_typedarray_type usearch_js_typedarray_of(usearch_scalar_t scalar, size_t* unit_bytes);

/** Wraps one reachability audit. */
napi_value usearch_js_reachability_object(napi_env env, usearch_reachability_t const* reachability);

/** Answers @c undefined. */
napi_value usearch_js_finish_nothing(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Answers the count the call wrote. */
napi_value usearch_js_finish_written(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Wraps what a refine or a reshard did to the graph. */
napi_value usearch_js_finish_report(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Wraps the audit a health check took. */
napi_value usearch_js_finish_health(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views the ranked run a search wrote inside its arena. */
napi_value usearch_js_finish_matches(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views a ragged run of pairs and its Arrow offsets inside the arena, as
 *  @c {offsets, keys, scores, required}. */
napi_value usearch_js_finish_multiply(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views the draws a sample wrote as @c {keys, entryOrdinals, parents, coordinates, draws}. */
napi_value usearch_js_finish_sample(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views the tape a sequence or shape retrieval wrote as @c {data, offsets, entries}, its elements
 *  @c count scalars of the call's kind apiece. */
napi_value usearch_js_finish_tape(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views the monolith inside its arena. */
napi_value usearch_js_finish_save_buffer(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views the keys a cursor pull wrote as @c {keys, entries}. */
napi_value usearch_js_finish_cursor_keys(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Views the keys and links a cursor pull wrote as @c {keys, entries, links: {offsets, keys}}. */
napi_value usearch_js_finish_cursor_links(napi_env env, usearch_js_call_t* call, napi_value const* slots);

/** Releases the engine walk through @p release, then the references keeping its index and executor
 *  alive. */
void usearch_js_walk_release(napi_env env, usearch_js_walk_t* walk, void (*release)(void*));

/** Reads the walk behind the receiver, and the index handle and executor its pull names. */
usearch_js_walk_t* usearch_js_walk_self(napi_env env, napi_callback_info info, size_t wanted, napi_value* arguments,
                                        void** index, usearch_executor_t* executor);

/**
 *  @brief Wraps an engine walk into an object exposing @p next, @c free and @c [Symbol.dispose],
 *      holding the index the walk borrows and the executor every pull runs on.
 *  @return The object, or @c NULL with an exception pending and the walk released.
 */
napi_value usearch_js_walk_object(napi_env env, void* handle, napi_value index, napi_value executor, napi_callback next,
                                  napi_callback free_method, napi_finalize finalize, void (*release)(void*));

#pragma endregion Index Verbs

#endif // UNUM_USEARCH_JAVASCRIPT_LIB_H
