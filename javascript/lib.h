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
 *  borrows the caller's @c TypedArray storage, hands the engine one strided matrix, and lets
 *  the engine land its outputs straight in the @c ArrayBuffer-s handed back. The workers live
 *  in an @c Executor the caller constructs and passes in, which is what lets several indexes
 *  share one pool.
 *
 *  No verb takes a JavaScript callback. The engine calls a predicate or a progress handler
 *  from the threads it fanned out over, while the thread owning the JavaScript context sits
 *  blocked inside the verb - and a thread-safe function would queue onto that same blocked
 *  loop. Lazy pulling through @c usearch_dense_find is the filtered path that stays on the
 *  caller's thread.
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

#include <usearch/dense_index.h>
#include <usearch/types.h>

#pragma region Vocabulary

/** Compile-time constants of the binding layer. */
enum {

    /** Bytes a filesystem path may occupy, terminator included. */
    usearch_js_path_capacity_k = 4096,

    /** Bytes an option name may occupy, terminator included. */
    usearch_js_name_capacity_k = 64,

    /** Bytes a formatted diagnostic may occupy, terminator included. */
    usearch_js_message_capacity_k = 256,

    /** Bytes one identity occupies, on the wire and in the engine alike. */
    usearch_js_key_bytes_k = 16,
};

/** Element count of a statically-sized table, which every naming table asks for. */
#define usearch_js_table_size_(table) (sizeof(table) / sizeof((table)[0]))

/** Everything the addon owns per environment: the constructors its verbs type-check
 *  against, reached through @c napi_get_instance_data rather than a namespace-scope global. */
typedef struct usearch_js_addon_t {

    /** The @c Executor constructor, kept alive for @c napi_instanceof. */
    napi_ref executor_class;

    /** The @c CompiledIndex constructor, kept alive for symmetry with the above. */
    napi_ref index_class;
} usearch_js_addon_t;

/** The workers a fanned-out verb runs on, owned apart from any index so one pool serves many. */
typedef struct usearch_js_executor_t {

    /** The engine handle, freed when the JavaScript wrapper is collected. */
    usearch_executor_t handle;
} usearch_js_executor_t;

/** An owned index handle, one per @c CompiledIndex instance. */
typedef struct usearch_js_index_t {

    /** The engine handle, freed when the JavaScript wrapper is collected. */
    usearch_dense_index_t handle;
} usearch_js_index_t;

/**
 *  @brief One borrowed @c TypedArray: its bytes and the scalar kind they spell.
 *
 *  Rows are never counted here, since a sub-byte kind packs several dimensions per byte and
 *  only the index's frozen width says how many bytes one row takes.
 */
typedef struct usearch_js_matrix_t {

    /** Pointer to the first scalar, offset into the backing @c ArrayBuffer already applied. */
    void* data;

    /** Bytes the view spans. */
    size_t bytes;

    /** The scalar type the bytes hold. */
    usearch_scalar_t scalar_kind;
} usearch_js_matrix_t;

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

#pragma endregion Vocabulary

#pragma region Error Reporting

/** Throws @p message under @p code; always returns false, so a caller returns straight through. */
bool usearch_js_throw(napi_env env, char const* code, char const* message);

/** Throws a @c TypeError saying what @p field expected; always returns false. */
bool usearch_js_throw_field(napi_env env, char const* field, char const* expectation);

/**
 *  @brief Throws the mapped JavaScript error when @p error carries a failure.
 *  @return @c true when the call succeeded, @c false with an exception pending otherwise.
 */
bool usearch_js_succeeded(napi_env env, usearch_error_t const* error);

#pragma endregion Error Reporting

#pragma region Argument Parsing

/** Whether @p value is @c undefined or @c null, which every optional argument derives from. */
bool usearch_js_value_absent(napi_env env, napi_value value);

/** Reads a non-negative count from a number or a @c BigInt; absent arguments read as zero. */
bool usearch_js_size_from_value(napi_env env, napi_value value, char const* field, size_t* result);

/** Reads a finite real; absent arguments read as zero, which every knob derives from. */
bool usearch_js_double_from_value(napi_env env, napi_value value, char const* field, double* result);

/** Copies a UTF-8 string into @p buffer, refusing anything that would not fit. */
bool usearch_js_string_from_value(napi_env env, napi_value value, char const* field, char* buffer, size_t capacity,
                                  size_t* length);

/** Reads a metric kind from its canonical spelling. */
bool usearch_js_metric_from_value(napi_env env, napi_value value, char const* field, usearch_dense_metric_t* result);

/** Reads a scalar kind from its canonical spelling; absent arguments read as unknown. */
bool usearch_js_scalar_from_value(napi_env env, napi_value value, char const* field, usearch_scalar_t* result);

/** Reads a residency from its canonical spelling. */
bool usearch_js_residency_from_value(napi_env env, napi_value value, char const* field, usearch_residency_t* result);

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
 *  @brief Bytes one row of @p columns scalars occupies in @p scalar_kind; zero when unknown.
 *
 *  Sub-byte kinds round the row up to whole bytes, which is what makes a packed @c u1x8 row of
 *  a hundred dimensions thirteen bytes rather than twelve and a half.
 */
size_t usearch_js_scalar_row_bytes(size_t columns, usearch_scalar_t scalar_kind);

/**
 *  @brief Borrows a numeric @c TypedArray - the shape every vector argument arrives in.
 *
 *  @param[in] env The environment the call runs in.
 *  @param[in] value The @c TypedArray to borrow.
 *  @param[in] field The argument's name, as a diagnostic spells it.
 *  @param[in] declared The scalar kind the caller named; unknown infers it from the array type.
 *  @param[out] matrix The borrowed bytes and the kind they hold.
 *  @return @c true on success, @c false with an exception pending otherwise.
 *
 *  A declared kind reinterprets the bytes rather than converting them, which is how packed
 *  @c u1x8 or @c bf16 rows reach the engine through a @c Uint8Array that spells neither.
 */
bool usearch_js_matrix_from_value(napi_env env, napi_value value, char const* field, usearch_scalar_t declared,
                                  usearch_js_matrix_t* matrix);

/**
 *  @brief Gathers a @c Uint8Array of sixteen-byte identities, or a @c BigUint64Array of integer
 *      ones, into engine keys.
 *  @return A @c malloc-ed array the caller frees, or @c NULL with an exception pending.
 */
usearch_key_t* usearch_js_keys_from_value(napi_env env, napi_value value, char const* field, size_t* count);

/** Reads one identity, which the argument spells the same way a batch of one would. */
bool usearch_js_key_from_value(napi_env env, napi_value value, char const* field, usearch_key_t* key);

/**
 *  @brief Allocates an @c ArrayBuffer of @p bytes and reports where its storage begins, so the
 *      engine lands its outputs there rather than in an arena a second pass would copy out of.
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

#pragma region Classes

/** Defines the @c Executor class; the module glue keeps the reference. */
napi_value usearch_js_executor_define(napi_env env);

/** Unwraps the @c executor argument every fanned-out verb takes. */
bool usearch_js_executor_from_value(napi_env env, napi_value value, usearch_executor_t* handle);

/** Defines the @c CompiledIndex class, its cursors and its finders included. */
napi_value usearch_js_index_define(napi_env env);

#pragma endregion Classes

#pragma region Library Probes

/** Answers with the library's semantic version. */
napi_value usearch_js_version(napi_env env, napi_callback_info info);

/** Answers with the kernel families this build carries, whatever the machine offers. */
napi_value usearch_js_capabilities_compiled(napi_env env, napi_callback_info info);

/** Answers with the kernel families this machine can run, and dispatch picks from. */
napi_value usearch_js_capabilities_available(napi_env env, napi_callback_info info);

#pragma endregion Library Probes

#endif // UNUM_USEARCH_JAVASCRIPT_LIB_H
