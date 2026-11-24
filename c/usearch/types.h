/**
 *  @file c/usearch/types.h
 *  @author Ash Vardanian
 *  @date August 14, 2026
 *  @brief Shared vocabulary of the USearch C interface: scalar types, keys, distances, shapes,
 *      outcomes, executors, arenas, and library introspection.
 *
 *  Everything the modality headers have in common - @c dense_index.h, @c sequence_index.h,
 *  @c shapes_index.h, and @c geospatial_index.h - and nothing they don't. Include @c usearch.h for
 *  all of them, or a modality header alone, since each pulls this one in. The C vocabulary sits
 *  beside its C++ twin @c types.hpp, so one installed tree serves both languages and the headers
 *  pair by stem.
 *
 *  @section types_verbs Verbs
 *
 *  Every verb that takes options, outputs, or callbacks takes one argument structure named after
 *  it - @c usearch_dense_search takes a @c usearch_dense_search_t - so optional arguments are
 *  skipped by name and their order never matters. Name lookups and the @c _free family stay
 *  positional. Every structure opens in the same order: the handle, @c executor, @c outcome, then
 *  @c completion. Inputs follow, then outputs and the @c arena they land in, then knobs, and last
 *  @c progress and @c stats.
 *
 *  Structures hold no size or version field, so they grow source-compatibly only: initialize
 *  them by designator, which zeroes every field left out, and expect the major version to change
 *  whenever one does.
 *
 *  @section types_asynchrony Asynchrony
 *
 *  Every call names its @c completion, and blocking and deferred calls share one shape. A blocking
 *  call, the default, returns with its final status stored and every output readable. A @c NULL
 *  executor runs the call on the calling thread. It creates no pool, thread, ring, SPDK queue pair
 *  or stream and probes no topology; its scratch and blocking queues live for that call only. Name
 *  an executor for calls that allocate nothing once warm. Whatever its completion, such a call is
 *  final on return.
 *
 *  An executor runs one call at a time. A call that finds another one holding its executor, from
 *  any thread, finishes @c usearch_status_contended_k without running.
 *
 *  A deferred call on a caller-exclusive executor returns once handed to the executor's thread 0,
 *  which runs it there, forks included, and finishes it, so polling alone sees it final. A deferred
 *  call on a caller-inclusive executor, whose threads need their caller, or on a VRAM executor,
 *  whose caller stages and feeds its devices, runs at submission and is final on return. Submission
 *  copies the structure; everything it points to must stay allocated and unchanged until the call
 *  is final, and @c outcome->status reads @c usearch_status_deferred_k until then.
 *
 *  A call lets go of its arena and of the handles it names, then of its executor, and only then
 *  stores the message and the final status with release ordering. A caller that reads the status
 *  final with acquire ordering may reuse the arena, submit the next call or free any handle at
 *  once. Waits, drains and frees only wait: the executor's own threads finish every call. A free
 *  waits for the calls naming its handle, so one issued from a callback of such a call waits on
 *  itself. A call stops early only when its progress handler returns @c usearch_cancel_k.
 *
 *  One thread keeps several executors busy by giving each a deferred call and polling the statuses.
 *  A VRAM call is final when it returns, so a manager submits it last:
 *
 *  @code{.c}
 *  usearch_outcome_t ram_outcome = {0}, nvme_outcome = {0};
 *  ram_search.outcome = &ram_outcome, ram_search.completion = usearch_completion_deferred_k;
 *  nvme_search.outcome = &nvme_outcome, nvme_search.completion = usearch_completion_deferred_k;
 *  usearch_dense_search(&ram_search);  // on a RAM executor
 *  usearch_dense_search(&nvme_search); // on an NVMe executor
 *  while (usearch_outcome_status(&ram_outcome) == usearch_status_deferred_k ||
 *         usearch_outcome_status(&nvme_outcome) == usearch_status_deferred_k)
 *      prepare_next_batch();
 *  @endcode
 *
 *  @section types_memory Memory
 *
 *  Two complications shape memory, and nothing else may: outputs whose size only the backend
 *  knows, and memory a GPU should place itself. Arenas answer the first, executors the second, and
 *  every binding stays one verb per operation rather than a twin per output style. Every byte has
 *  one owner, and each owner picks where its bytes live once, when it is created:
 *
 *  - The caller allocates every buffer a call reads or writes, and frees it after the call is
 *    final; no allocator callback crosses the interface, and every pointer a caller passes is an
 *    input, classified through the driver once per call.
 *  - Payload belongs to the handle, placed by its constructor's residency, as @c usearch_dense_init_ram
 *    places it in host memory.
 *  - Scratch and staging belong to the executor, placed at its creation and grown to a high-water mark.
 *  - Every output whose length scales with the call is written in an arena: matches, rows,
 *    products, monoliths, offsets, cursor and stream batches, and attention outputs alike. Only
 *    fixed-size reports, such as stats, descriptions, metadata and the scalar @c written and
 *    @c required, stay caller fields, and so does state the caller holds between calls, like
 *    attention's @c carry.
 *  - Executors make arenas. @c usearch_arena_init places one where its executor computes, host pages
 *    on its memory domains or the memory of its GPU, decided once; a @c NULL executor makes host
 *    pages for calls on the calling thread. Every arena is freed before its executor.
 *  - A call rewinds the arena it is given, invalidating the previous call's outputs, and writes views
 *    into it. Any verb may reuse any arena of its residency, one call at a time.
 *  - A call holds the arena from its start until just before its final status.
 *  - An Arrow export reads the arena's outputs until the arena's next call.
 *
 *  Outputs are views the call writes, so none holds a capacity or a stride for the caller to
 *  pick, and an arena keeps its high-water mark, so a steady loop allocates once. The library
 *  stages only to change a layout or to make pageable memory DMA-able. On a VRAM executor the
 *  outputs are device memory, which the caller copies out itself when it wants them elsewhere.
 *
 *  @code{.c}
 *  usearch_arena_t arena = NULL;
 *  usearch_arena_init(&(usearch_arena_init_t){.arena = &arena, .executor = executor, .outcome = &outcome});
 *  for (size_t i = 0; i != keys_count; ++i) {
 *      get.key = keys[i], get.vectors = &rows, get.written = &rows_count, get.arena = arena;
 *      usearch_dense_get(&get);
 *      if (get.outcome->status != usearch_status_success_k) break; // blocking, so already final
 *      consume(rows, rows_count); // valid until the next call given `arena`
 *  }
 *  usearch_arena_free(arena);
 *  @endcode
 *
 *  @section types_shapes Shapes
 *
 *  A verb's @c count is its batch axis and nothing else. Inputs are views, where a zero stride
 *  repeats the first element as BLAS does; outputs are views too, written packed into the arena,
 *  so none holds a capacity. Operands off the batch axis keep their own shape, and ragged inputs
 *  and outputs are tapes with Apache Arrow offsets: one more of them than entries, around the
 *  elements laid end to end.
 */
#ifndef UNUM_USEARCH_TYPES_H
#define UNUM_USEARCH_TYPES_H

#include <stddef.h> // `size_t`
#include <stdint.h> // `uint8_t`, `int32_t`, `uint32_t`

#pragma region Platform Identity

/*  Derived once, here, from the compiler's predefines; no other file reads them, and no build sets
 *  them. */
#if defined(USEARCH_OS_LINUX_) || defined(USEARCH_OS_APPLE_) || defined(USEARCH_OS_WINDOWS_) || \
    defined(USEARCH_OS_FREEBSD_) || defined(USEARCH_OS_WASI_) || defined(USEARCH_OS_POSIX_)
#error "USEARCH_OS_*_ are identity, derived from the compiler; set a USEARCH_WITH_* facility instead"
#endif

/** Whether the kernel is Linux, Android included, as 1 or 0. */
#if defined(__linux__)
#define USEARCH_OS_LINUX_ 1
#else
#define USEARCH_OS_LINUX_ 0
#endif

/** Whether the kernel is Darwin: macOS, iOS, tvOS, watchOS or visionOS, as 1 or 0. */
#if defined(__APPLE__)
#define USEARCH_OS_APPLE_ 1
#else
#define USEARCH_OS_APPLE_ 0
#endif

/** Whether the target is Windows, MinGW included and Cygwin not, as 1 or 0. */
#if defined(_WIN32)
#define USEARCH_OS_WINDOWS_ 1
#else
#define USEARCH_OS_WINDOWS_ 0
#endif

/** Whether the kernel is FreeBSD, as 1 or 0. */
#if defined(__FreeBSD__)
#define USEARCH_OS_FREEBSD_ 1
#else
#define USEARCH_OS_FREEBSD_ 0
#endif

/** Whether the target is the WebAssembly System Interface, which maps no files, as 1 or 0. */
#if defined(__wasi__)
#define USEARCH_OS_WASI_ 1
#else
#define USEARCH_OS_WASI_ 0
#endif

/** Whether the POSIX interfaces exist: Linux, Darwin, FreeBSD or Cygwin, as 1 or 0. */
#if USEARCH_OS_LINUX_ || USEARCH_OS_APPLE_ || USEARCH_OS_FREEBSD_ || defined(__CYGWIN__)
#define USEARCH_OS_POSIX_ 1
#else
#define USEARCH_OS_POSIX_ 0
#endif

#if USEARCH_OS_LINUX_ + USEARCH_OS_APPLE_ + USEARCH_OS_WINDOWS_ + USEARCH_OS_FREEBSD_ + USEARCH_OS_WASI_ > 1
#error "Two operating-system identities hold at once"
#endif
#if USEARCH_OS_POSIX_ && USEARCH_OS_WINDOWS_
#error "POSIX and Windows identities hold at once"
#endif

#pragma endregion Platform Identity

#pragma region Facilities

/** Whether the io_uring engines build, as 1 or 0: on Linux, unless a build sets 0. */
#if !defined(USEARCH_WITH_LINUX_URING) && USEARCH_OS_LINUX_
#define USEARCH_WITH_LINUX_URING 1
#elif !defined(USEARCH_WITH_LINUX_URING)
#define USEARCH_WITH_LINUX_URING 0
#endif
#if USEARCH_WITH_LINUX_URING && !USEARCH_OS_LINUX_
#error "USEARCH_WITH_LINUX_URING needs Linux"
#endif

/** Whether the Linux AIO engine builds, as 1 or 0: on Linux with the kernel's AIO header, unless a
 *  build sets 0. */
#if !defined(USEARCH_WITH_LINUX_AIO) && USEARCH_OS_LINUX_
#if __has_include(<linux/aio_abi.h>)
#define USEARCH_WITH_LINUX_AIO 1
#endif
#endif // !defined(USEARCH_WITH_LINUX_AIO) && USEARCH_OS_LINUX_
#if !defined(USEARCH_WITH_LINUX_AIO)
#define USEARCH_WITH_LINUX_AIO 0
#endif
#if USEARCH_WITH_LINUX_AIO && !USEARCH_OS_LINUX_
#error "USEARCH_WITH_LINUX_AIO needs Linux"
#endif

/** Whether the I/O Rings engine builds, as 1 or 0: on Windows with an SDK that ships
 *  @c ioringapi.h, unless a build sets 0. */
#if !defined(USEARCH_WITH_WINDOWS_IORING) && USEARCH_OS_WINDOWS_ && defined(__has_include)
#if __has_include(<ioringapi.h>)
#define USEARCH_WITH_WINDOWS_IORING 1
#endif
#endif // !defined(USEARCH_WITH_WINDOWS_IORING) && USEARCH_OS_WINDOWS_ && defined(__has_include)
#if !defined(USEARCH_WITH_WINDOWS_IORING)
#define USEARCH_WITH_WINDOWS_IORING 0
#endif
#if USEARCH_WITH_WINDOWS_IORING && !USEARCH_OS_WINDOWS_
#error "USEARCH_WITH_WINDOWS_IORING needs Windows"
#endif

/** Whether SPDK's user-space NVMe driver is linked, as 1 or 0, which the build passes on. */
#if !defined(USEARCH_WITH_LINUX_SPDK)
#define USEARCH_WITH_LINUX_SPDK 0
#endif
#if USEARCH_WITH_LINUX_SPDK && !USEARCH_OS_LINUX_
#error "USEARCH_WITH_LINUX_SPDK needs Linux"
#endif

/** Whether this unit builds the shared library, as 1 or 0. */
#if !defined(USEARCH_EXPORTING)
#define USEARCH_EXPORTING 0
#endif

/** Whether this unit links the shared library rather than the archive, as 1 or 0. */
#if !defined(USEARCH_SHARED)
#define USEARCH_SHARED 0
#endif

/**
 *  @brief Marks every function the library exports, and nothing else.
 *
 *  On Windows it exports from the shared library, which builds with @c USEARCH_EXPORTING, imports
 *  where a consumer of that library sets @c USEARCH_SHARED, as linking @c usearch::shared does, and
 *  expands to nothing for the static library. GCC and Clang give the function default visibility,
 *  so it survives @c -fvisibility=hidden. Define it to override.
 */
#if !defined(USEARCH_EXPORT)
#if USEARCH_OS_WINDOWS_ && USEARCH_EXPORTING
#define USEARCH_EXPORT __declspec(dllexport)
#elif USEARCH_OS_WINDOWS_ && USEARCH_SHARED
#define USEARCH_EXPORT __declspec(dllimport)
#elif USEARCH_OS_WINDOWS_
#define USEARCH_EXPORT
#elif defined(__GNUC__) || defined(__clang__)
#define USEARCH_EXPORT __attribute__((visibility("default")))
#else
#define USEARCH_EXPORT
#endif
#endif // !defined(USEARCH_EXPORT)

#pragma endregion Facilities

#ifdef __cplusplus
extern "C" {
#endif

/** The library's version, which the release tooling rewrites in place. */
#define USEARCH_VERSION_MAJOR 3
#define USEARCH_VERSION_MINOR 0
#define USEARCH_VERSION_PATCH 0

/** One byte of raw storage, read or written without interpretation: a key's identity, a caller's
 *  run's contents, the buffers callers hand in. */
typedef uint8_t usearch_byte_t;

/**
 *  @brief User-facing 128-bit key identifying one entry of any index - sixteen opaque bytes,
 *      rendered canonically as a lowercase dashed UUID, matching SafeGraphs' @c uuid key column.
 *
 *  Keys need not be unique: several entries may share one, and count-like functions report how
 *  many. Integer identities sit in the low bytes with the rest zeroed.
 */
typedef struct usearch_key_t {
    usearch_byte_t bytes[16];
} usearch_key_t;

/** Distance or similarity score between two entries, in the metric of the index that
 *  produced it. Wide enough for every modality's scoring range. */
typedef double usearch_distance_t;

/**
 *  @brief A caller-owned run of bytes and its length, never null-terminated by contract - the shape
 *      every modality's string arguments arrive in.
 *
 *  Callers holding a Rust @c str, a Go string, or a Java byte array pass their own storage without
 *  manufacturing a terminator none of them keeps.
 */
typedef struct usearch_bytes_view_t {

    /** First byte of the run; text is bytes that happen to decode. */
    usearch_byte_t const* data;

    /** Bytes the run spans, with no terminator assumed or written. */
    size_t length;
} usearch_bytes_view_t;

/**
 *  @brief A strided run of keys - how a batch names the identities it holds, one per row of the
 *      call's @c count, or one per entry a cursor pull wrote into its arena.
 *
 *  A zero @c stride repeats the first key for every row, which is how a batch of rows is written
 *  under one key without a caller materializing the copies.
 *  An output's keys are packed, so the call writes one key's size as @c stride.
 */
typedef struct usearch_keys_view_t {

    /** First key of the run. */
    usearch_key_t const* data;

    /** Bytes between consecutive keys; zero repeats the first. */
    size_t stride;
} usearch_keys_view_t;

/**
 *  @brief What went wrong, in a vocabulary shared by the engine and by every binding that
 *      validates before calling it - so one failure holds one identity, whoever catches it.
 *
 *  Some enumerators name failures this library never returns itself: a binding holding richer
 *  layout information than the ABI can accept rejects the call before it crosses, and reports the
 *  same status the engine would if it could see the same thing. The set describes the project's
 *  failures, not this implementation's exits, so unused enumerators are deliberate.
 *
 *  Statuses fall into five categories: resources, for memory, I/O and devices; input, for shapes,
 *  layouts, kinds, malformed data and invalid arguments; state, for missing reservations, frozen
 *  configurations, read-only handles, exhausted ID spaces, held executors and arenas, and
 *  unsupported verbs; interruptions, for cancellation and deferral; and defects. Any verb that
 *  allocates, binds a device or touches a file may also return a resources status, and any verb
 *  taking a progress callback may be cancelled. Each verb's own block lists only the statuses its
 *  contract adds.
 */
typedef enum usearch_status_t {

    /** The call succeeded; @c message holds nothing. */
    usearch_status_success_k = 0,

    /** The call was submitted deferred and is not final yet; the one non-final status. */
    usearch_status_deferred_k = 1,

    /** An allocation or reservation could not be satisfied. */
    usearch_status_out_of_memory_k = 2,

    /** Dimensionality or row counts disagree. */
    usearch_status_shape_mismatch_k = 3,

    /** A stride or packing the engine cannot traverse. */
    usearch_status_unsupported_layout_k = 4,

    /** A scalar, metric, residency, traversal or relation tag this build does not know. */
    usearch_status_unknown_kind_k = 5,

    /** A mutating verb on a handle that serves its checkpoint in place. */
    usearch_status_read_only_k = 6,

    /** A file could not be opened, read, written, or parsed. */
    usearch_status_io_k = 7,

    /** A device could not be selected, or its memory could not be claimed. */
    usearch_status_device_k = 8,

    /** A caller-supplied representation could not be decoded - a malformed key, a truncated
     *  buffer or checkpoint. */
    usearch_status_malformed_k = 9,

    /** The library reported neither success nor a failure - a contract violation rather
     *  than a condition, so it names a defect and not a state. */
    usearch_status_internal_k = 10,

    /** A hard design ceiling was reached, like the 32-bit ID space; a smaller request
     *  cannot help where a smaller allocation would. */
    usearch_status_overflow_k = 11,

    /** The verb needs a prior successful reservation that never happened. */
    usearch_status_uninitialized_k = 12,

    /** The request contradicts an already-frozen configuration, like re-reserving with
     *  a different geometry. */
    usearch_status_config_mismatch_k = 13,

    /** The call was refused rather than waited for: another call holds its executor or its arena,
     *  or a drain came from inside a callback of the call it would wait on. */
    usearch_status_contended_k = 14,

    /** A progress handler asked the operation to stop; it unwound and published nothing, so the
     *  handle holds whatever it did before the call. */
    usearch_status_cancelled_k = 15,

    /** The verb is declared for every residency but this one does not implement it, its traversal
     *  or its predicate, or the build left the requested quantization out; no retry, option, or
     *  smaller request can help, so a caller picks another residency, traversal or build. */
    usearch_status_unsupported_k = 16,

    /** An argument breaks the verb's contract - a required pointer left @c NULL, inputs inside the
     *  arena the call rewinds, a name no table spells. The outcome itself is never @c NULL: it is
     *  where this status lands. */
    usearch_status_invalid_argument_k = 17,
} usearch_status_t;

/** Outcome of one call: the status word a deferred call completes through and every binding
 *  branches on, and a static message naming what broke, a library invariant or a rule of this
 *  call, else the status's own @ref usearch_status_name. Messages are never freed, and are
 *  @c NULL exactly when the status is @c usearch_status_success_k. */
typedef struct usearch_outcome_t {

    /** What happened, for the caller to branch on. */
    usearch_status_t status;

    /** The static detail behind a failing @c status, @c NULL on success. */
    char const* message;
} usearch_outcome_t;

/** Scalar types the indexes store and exchange, named by width and encoding.
 *  Sub-byte types pack several dimensions per byte, as their @c xN suffixes spell. */
typedef enum usearch_scalar_t {

    /** No scalar named - an unset field, or a custom metric owning its own storage. */
    usearch_scalar_unknown_k = 0,

    // Floating point
    /** IEEE-754 binary64: ±1.80e308, epsilon 2.22e-16, subnormals to 4.94e-324,
     *  with infinities and NaNs. */
    usearch_scalar_f64_k = 1,

    /** IEEE-754 binary32: ±3.40e38, epsilon 1.19e-7, subnormals to 1.40e-45, with
     *  infinities and NaNs. */
    usearch_scalar_f32_k = 2,

    /** Truncated f32 - 8-bit exponent, 7-bit mantissa: f32's range at ±3.39e38 and
     *  epsilon 7.81e-3, with infinities and NaNs. */
    usearch_scalar_bf16_k = 3,

    /** IEEE-754 binary16: ±65504, epsilon 9.77e-4, subnormals to 5.96e-8, with
     *  infinities and NaNs. */
    usearch_scalar_f16_k = 4,

    /** FP8 with a 5-bit exponent: ±57344, epsilon 0.25, with infinities and NaNs -
     *  f16's dynamic range held on two mantissa bits. */
    usearch_scalar_e5m2_k = 5,

    /** FP8 with a 4-bit exponent: ±448, epsilon 0.125, no infinities, and two NaN
     *  codes at 0x7F and 0xFF. */
    usearch_scalar_e4m3_k = 6,

    /** FP6 with a 3-bit exponent, byte-aligned with two padding bits: ±28, no
     *  infinities or NaNs, so every one of its 64 codes is a number. */
    usearch_scalar_e3m2_k = 7,

    /** FP6 with a 2-bit exponent, byte-aligned with two padding bits: ±7.5, no
     *  infinities or NaNs, and 72% of its codes outside [-1, 1]. */
    usearch_scalar_e2m3_k = 8,

    // Signed integers
    /** 8-bit signed integer spanning -128 to 127, every code a number. */
    usearch_scalar_i8_k = 10,

    /** Two signed nibbles per byte spanning -8 to 7, even dimensions in the low
     *  nibble and odd ones in the high. */
    usearch_scalar_i4x2_k = 11,

    /** 32-bit signed integer spanning ±2.15e9, every code a number. */
    usearch_scalar_i32_k = 12,

    // Unsigned integers
    /** 8-bit unsigned integer spanning 0 to 255, every code a number. */
    usearch_scalar_u8_k = 20,

    /** Two unsigned nibbles per byte spanning 0 to 15, even dimensions in the low
     *  nibble and odd ones in the high. */
    usearch_scalar_u4x2_k = 21,

    /** Eight single-bit dimensions per byte - the form the Hamming & Jaccard metrics
     *  score, where only the set-bit count of a XOR or AND holds meaning. */
    usearch_scalar_u1x8_k = 22,

} usearch_scalar_t;

/**
 *  @brief A run of rows along a call's batch axis - the currency of every verb whose rows share one
 *      width, read from the caller as an input and written packed into an arena as an output.
 *
 *  An input's row count is the call's @c count, and the call checks @c columns against the frozen
 *  geometry. A zero @c stride repeats the first row.
 *  An output's rows are packed: the call writes one row's bytes as @c stride, and @c columns and
 *  @c scalar as it wrote them, so neither a repeat nor a geometry check applies.
 */
typedef struct usearch_rows_view_t {

    /** Base pointer of the first row. */
    void const* data;

    /** Bytes between the starts of consecutive rows; zero repeats the first row. */
    size_t stride;

    /** Values per row; an input's must match the frozen geometry, and sub-byte rows of either kind
     *  must fill whole bytes, or they are refused. */
    size_t columns;

    /** The scalar type the rows hold, converted on the way in where it differs. */
    usearch_scalar_t scalar;
} usearch_rows_view_t;

/**
 *  @brief A caller-owned strided matrix off the batch axis - an operand whose rows and columns are
 *      its own, like the value rows attention gathers from.
 *
 *  @c rows and @c columns are counts and @c stride alone is in bytes. A zero @c stride repeats the
 *  first row.
 */
typedef struct usearch_matrix_view_t {

    /** Base pointer of the first row. */
    void const* data;

    /** Rows the matrix holds. */
    size_t rows;

    /** Logical scalars per row, as the caller counts them. */
    size_t columns;

    /** Bytes between the starts of consecutive rows; zero repeats the first row. */
    size_t stride;

    /** The scalar type the rows hold. */
    usearch_scalar_t scalar;
} usearch_matrix_view_t;

/** How wide a tape's offsets are, naming the member of its @c offsets union to read. */
typedef enum usearch_tape_layout_t {

    /** Apache Arrow @c Binary or @c List, with 32-bit offsets. */
    usearch_tape_u32_k = 0,

    /** Apache Arrow @c LargeBinary or @c LargeList, the same with 64-bit offsets. */
    usearch_tape_u64_k = 1,
} usearch_tape_layout_t;

/**
 *  @brief A ragged batch as Apache Arrow lays one out: elements end to end, and one more offset
 *      than entries around them, read as an input or written into an arena as an output.
 *
 *  Entry @c i spans elements @c offsets[i] up to @c offsets[i+1], so the last offset closes the
 *  final entry. What an element is belongs to the modality: a byte of a sequence, a point of a
 *  shape or of a geospatial set. As an output the call writes offsets as wide as @c size_t.
 */
typedef struct usearch_tape_view_t {

    /** First element of the tape, which every offset counts from. */
    void const* data;

    /** Which member of @c offsets the tape spells them in. */
    usearch_tape_layout_t layout;

    /** Entries + 1 non-decreasing offsets, in elements: the call's @c count + 1 for an input, its
     *  @c written + 1 for an output. */
    union {

        /** The offsets of a @c usearch_tape_u32_k tape. */
        uint32_t const* u32;

        /** The offsets of a @c usearch_tape_u64_k tape. */
        uint64_t const* u64;
    } offsets;

    /** The scalar type the elements hold: @c u8 for sequences, @c f64 degrees for geospatial
     *  inputs, and any kind for shape points, converted on the way in where it differs from
     *  storage. */
    usearch_scalar_t scalar;
} usearch_tape_view_t;

/**
 *  @brief A ranked run a call wrote into its arena - @c count rows of @c columns packed columns,
 *      where @c count is the call's own.
 *
 *  Holds no query count of its own: the call already fixed it, and a second copy is a second
 *  thing to disagree. Query @c i holds @c counts[i] matches from @c keys[i · columns] and
 *  @c scores[i · columns] on, and the columns past them are unwritten.
 *
 *  What @c scores hold is each verb's own: a search writes distances nearest-first, whichever kind
 *  the index froze - the inner kind as @c 1 @c - @c dot - while the linear-algebra verbs write
 *  similarities, as their own docs state.
 */
typedef struct usearch_matches_view_t {

    /** First row of the keys, @c columns per row. */
    usearch_key_t const* keys;

    /** First row of the scores, positionally paired with @c keys. */
    usearch_distance_t const* scores;

    /** Matches written per query, one entry per query. */
    size_t const* counts;

    /** Columns per row the call wrote, at most its @c wanted - the "k" in "kANN". */
    size_t columns;
} usearch_matches_view_t;

/**
 *  @brief A ragged run of matches a call wrote into its arena, for verbs whose queries hold as many
 *      matches as their walks found: keys and scores end to end, and one more offset than queries.
 *
 *  Query @c i spans @c offsets[i] up to @c offsets[i+1] in both arrays, so the last offset is the
 *  total. Scores follow the same per-verb meaning as in @c usearch_matches_view_t.
 */
typedef struct usearch_matches_tape_view_t {

    /** Every query's keys, end to end. */
    usearch_key_t const* keys;

    /** Every query's scores, positionally paired with @c keys. */
    usearch_distance_t const* scores;

    /** The call's @c count + 1 non-decreasing offsets into @c keys and @c scores. */
    size_t const* offsets;
} usearch_matches_tape_view_t;

/**
 *  @brief A ragged run of keys a call wrote into its arena, for verbs whose entries name as many
 *      keys as each one links to: keys end to end, and one more offset than entries.
 *
 *  Entry @c i spans @c offsets[i] up to @c offsets[i+1], so the last offset is the total.
 */
typedef struct usearch_keys_tape_view_t {

    /** Every entry's keys, end to end. */
    usearch_key_t const* keys;

    /** The call's @c written + 1 non-decreasing offsets into @c keys. */
    size_t const* offsets;
} usearch_keys_tape_view_t;

/** The CPU families USearch ships tuned variants for - dispatch reports its pick in this
 *  vocabulary, and callers may cap it with a ceiling. Each class bundles the compute features, wait
 *  instruction, and atomics its silicon generation guarantees. */
typedef enum usearch_cpu_class_t {

    /** No class detected. */
    usearch_cpu_class_unknown_k = 0,

    // Portable

    /** The portable floor: no SIMD assumptions, standard yields. */
    usearch_cpu_class_serial_k = 1,

    /** The one class of a monoversioned build: every capability its floor implies or its preset
     *  vouches for, as the kernel libraries compiled them. */
    usearch_cpu_class_monoversioned_k = 2,

    // x86

    /** Haswell-class x86: AVX2 & FMA, the 256-bit floor. */
    usearch_cpu_class_intel_haswell_k = 3,

    /** Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI and its successors without
     *  AVX-512, TPAUSE waits, CMPCCXADD conditional adds. */
    usearch_cpu_class_intel_sierra_k = 4,

    /** Skylake-class x86: the AVX-512 foundation tier. */
    usearch_cpu_class_intel_skylake_k = 5,

    /** Icelake-class x86: AVX-512 VNNI & VPOPCNT - the integer & bit workhorse. */
    usearch_cpu_class_intel_icelake_k = 6,

    /** Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512. */
    usearch_cpu_class_amd_genoa_k = 7,

    /** Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 &
     *  AMX-INT8 tiles. */
    usearch_cpu_class_intel_sapphire_k = 8,

    /** Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles. */
    usearch_cpu_class_intel_granite_k = 9,

    /** Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions,
     *  TPAUSE waits, CMPCCXADD conditional adds. */
    usearch_cpu_class_intel_diamond_k = 10,

    // Arm

    /** Graviton2-class Arm - Neoverse N1, the bare NEON floor; phones predating
     *  Armv8.6 land here too. */
    usearch_cpu_class_aws_g2_k = 12,

    /** Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2,
     *  Cortex-A710 & X2 class silicon. */
    usearch_cpu_class_apple_m2_k = 13,

    /** Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles. */
    usearch_cpu_class_apple_m4_k = 14,

    /** Graviton5-class Arm - Neoverse V3, Armv9.2: monitored waits; no build compiles it yet. */
    usearch_cpu_class_aws_g5_k = 15,

    /** Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products. */
    usearch_cpu_class_nvidia_vera_k = 16,

    // RISC-V

    /** Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23: vectors with Zvbb, Zawrs
     *  monitored waits, Zacas single-instruction compare-and-swap. */
    usearch_cpu_class_alibaba_c930_k = 17,

    // Power

    /** IBM POWER10-class Power, ISA 3.1: VSX vectors, the MMA outer-product accumulators and
     *  pause_short waits. */
    usearch_cpu_class_ibm_power10_k = 18,

    // WebAssembly

    /** WebAssembly every engine loads: 32-bit, single-threaded, the fixed-width simd128 tier. */
    usearch_cpu_class_wasm32_v128_k = 19,

    /** WebAssembly with relaxed SIMD, shared memory for real thread pools, and the 64-bit
     *  addressing that alone lets one module's memory pass 4 GB. */
    usearch_cpu_class_wasm64_extended_k = 20,
} usearch_cpu_class_t;

/** The GPU families USearch ships tuned variants for, the device twin of
 *  @c usearch_cpu_class_t: one class per generation of kernel kits. */
typedef enum usearch_gpu_class_t {

    /** No class detected. */
    usearch_gpu_class_unknown_k = 0,

    // NVIDIA

    /** Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30. */
    usearch_gpu_class_nvidia_ampere_k = 1,

    /** Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions. */
    usearch_gpu_class_nvidia_ada_k = 2,

    /** Hopper-class NVIDIA - H100 & H200: no TMA gather, DPX instructions. */
    usearch_gpu_class_nvidia_hopper_k = 3,

    /** Blackwell-class NVIDIA - B200, B300 & GB300: TMA gather4 scattered fetches. */
    usearch_gpu_class_nvidia_blackwell_k = 4,

    /** Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark. */
    usearch_gpu_class_nvidia_blackwellrtx_k = 5,

    /** Rubin-class NVIDIA - the Vera-Rubin platform's device half. */
    usearch_gpu_class_nvidia_rubin_k = 6,

    // AMD

    /** CDNA3 AMD - MI300X & MI325X: gfx942, 64-wide waves, FNUZ fp8. */
    usearch_gpu_class_amd_cdna3_k = 7,

    /** CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8. */
    usearch_gpu_class_amd_cdna4_k = 8,

    /** CDNA5 AMD - MI400: gfx1250, 32-wide waves. */
    usearch_gpu_class_amd_cdna5_k = 9,

    // Apple
    /** Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2. */
    usearch_gpu_class_apple_metal7_k = 10,

    /** Apple9-family GPUs - M3 & M4. */
    usearch_gpu_class_apple_metal9_k = 11,

    /** Apple10-family GPUs - M5: the Neural Accelerator matrix units. */
    usearch_gpu_class_apple_metal10_k = 12,
} usearch_gpu_class_t;

/** The GPU vendors one binary drives at once, each through a runtime loaded on first use. */
typedef enum usearch_gpu_vendor_t {
    usearch_gpu_vendor_unknown_k = 0,

    /** NVIDIA, through the CUDA driver when one is installed. */
    usearch_gpu_vendor_nvidia_k,

    /** AMD, through the HIP runtime when one is installed. */
    usearch_gpu_vendor_amd_k,

    /** Apple, through Metal, present on every Apple silicon Mac. */
    usearch_gpu_vendor_apple_k,
} usearch_gpu_vendor_t;

/** One GPU as USearch numbers it: its position in @c usearch_gpu_devices is the USearch device
 *  ordinal every VRAM call names. */
typedef struct usearch_gpu_device_t {

    /** The vendor whose runtime drives the device. */
    usearch_gpu_vendor_t vendor;

    /** The device's ordinal in its vendor's runtime, as @c nvidia-smi or @c rocm-smi list it. */
    uint32_t vendor_ordinal;

    /** The class whose kernels run on it. */
    usearch_gpu_class_t gpu_class;

    /** Streaming multiprocessors, compute units or GPU cores. */
    uint32_t compute_units;

    /** Device memory in bytes, or the working set Metal recommends for unified memory. */
    uint64_t memory_bytes;

    /** The vendor's own name for the device, NUL-terminated and truncated to fit. */
    char name[64];
} usearch_gpu_device_t;

/**
 *  @brief Where an index's payload lives and which engine traverses it - the vocabulary the
 *      lifecycle verbs spell in their names, as @c usearch_dense_init_vram does.
 *
 *  Each modality names a verb per action and residency rather than taking this as a tag, so a
 *  residency a modality lacks has no verb to call. The enumerators remain for the name tables
 *  bindings parse user text with.
 */
typedef enum usearch_residency_t {

    /** No residency named. */
    usearch_residency_unknown_k = 0,

    /** Volatile-memory index traversed by CPU threads - the concurrent baseline. */
    usearch_residency_ram_k = 1,

    /** Accelerator-memory index traversed by GPU warps on a chosen device. */
    usearch_residency_vram_k = 2,

    /** Persisted index traversed through pluggable I/O engines, sized past RAM. */
    usearch_residency_nvme_k = 3,
} usearch_residency_t;

/** How a search reaches the indexed entries it scores. */
typedef enum usearch_traversal_t {

    /** No traversal named, which derives the walk. */
    usearch_traversal_unknown_k = 0,

    /** Walk the graph, scoring the entries the expansion reaches - approximate, the default. */
    usearch_traversal_walk_k = 1,

    /** Score every entry published when the query's scan starts - exact, the ground truth a walk
     *  approximates. */
    usearch_traversal_scan_k = 2,
} usearch_traversal_t;

/** Traversal telemetry, shared by every verb that walks a graph in every modality - a pure output,
 *  fully overwritten by the call it is passed to. Passing @c NULL skips reporting. */
typedef struct usearch_traversal_stats_t {

    /** Graph members visited while resolving the batch. */
    size_t visited_members;

    /** Distances computed while resolving the batch, diversity re-selections included. */
    size_t computed_distances;

    /** Starvation restarts - re-seedings after the frontier drained prematurely,
     *  and a graph-connectivity health signal rather than a cost counter. */
    size_t restarts;

    /** Hint keys no shard held an entry under; zero unless a search was handed hints. */
    size_t missing_hints;
} usearch_traversal_stats_t;

/** One reachability audit: the population it judged, and how much of it a search reaches. */
typedef struct usearch_reachability_t {

    /** Entries published when the audit started - the population it judges; entries inserted
     *  while it ran are not counted, so compare @c reachable against this, never a later size. */
    size_t audited;

    /** Of those, the entries a search reaches from the root; the rest no search finds until a
     *  refinement re-links them. */
    size_t reachable;

    /** Depths the sweep from the root took; on a sharded handle, the deepest shard's. */
    size_t depth_from_root;

    /** Depths the sweep toward the root took; on a sharded handle, the deepest shard's. */
    size_t depth_to_root;

    /** Entries no reached entry links to. */
    size_t unlinked;

    /** Entries linked to by fewer entries than they link to. */
    size_t underlinked;

    /** Entries deeper than the depth holding the most entries. */
    size_t beyond_most_populated_depth;

    /** Islands no entry outside them links into, so no search from the root reaches them. */
    size_t unreached_islands;

    /** Islands that link to no entry outside them, so no search from inside them leaves. */
    size_t stranded_islands;

    /** Components when direction is ignored. */
    size_t components;
} usearch_reachability_t;

/** The kind of work a plan is doing, shared by refine and reshard in every residency: a reshard's
 *  moves first, then a round's phases in their order, then the closing write. */
typedef enum usearch_phase_t {

    /** No plan, or one that has not started. */
    usearch_phase_none_k = 0,

    /** A reshard copying source entries into their target with their links shifted. */
    usearch_phase_copy_k,

    /** A reshard building graphs over entries that arrived with no links, as NVMe parts do. */
    usearch_phase_build_k,

    /** A reshard cutting a source into one part per target. */
    usearch_phase_partition_k,

    /** A reshard moving each part into its target. */
    usearch_phase_scatter_k,

    /** A round's health check, opening or closing it. */
    usearch_phase_health_k,

    /** A round linking each island with the root's component. */
    usearch_phase_heal_k,

    /** A round drawing the hubs its gather streams entries past. */
    usearch_phase_hubs_k,

    /** A round gathering every entry into a neighborhood. */
    usearch_phase_gather_k,

    /** A round re-stitching each neighborhood. */
    usearch_phase_restitch_k,

    /** A plan filing what its moves or rounds produced where it lands: a cut's keys, or NVMe
     *  parts. */
    usearch_phase_write_k,

    /** The plan ended its round, and nothing is pending. */
    usearch_phase_done_k,
} usearch_phase_t;

/** Where a plan stands: its phase, that phase's position in the plan, and its own count. */
typedef struct usearch_phase_progress_t {

    /** The kind of work in hand. */
    usearch_phase_t phase;

    /** This phase's position among the ones the plan runs. */
    size_t ordinal;

    /** Units of this phase done, entries or islands. */
    size_t completed;

    /** Units of this phase in all. */
    size_t total;
} usearch_phase_progress_t;

/** What a round concluded, so no caller picks a change rate. */
typedef enum usearch_refine_verdict_t {

    /** No round concluded: a report no round filled yet. */
    usearch_refine_none_k = 0,

    /** The round changed some entry's link set, so another round can help. */
    usearch_refine_improved_k,

    /** The round changed no entry's link set. */
    usearch_refine_converged_k,

    /** The round lowered nothing while something stands, as full entries leave no slot. */
    usearch_refine_stalled_k,

    /** The caller stopped the round; the phase names where. */
    usearch_refine_stopped_k,
} usearch_refine_verdict_t;

/**
 *  @brief What one refining call did to the graph, counted alike in every modality and residency:
 *      @c refine and @c reshard land it, so far when their plan stays live, and each operation's
 *      @c plan_finish lands the whole plan's.
 *
 *  Every one of those verbs ends in rounds, each a health check, a heal, a draw of hubs, a gather
 *  and a restitch, closed by a second health check, and the report counts them. @c changed never
 *  exceeds @c reselected, and a round over a settled graph changes nothing. A call that fails
 *  before its closing health check reports zeros; one its progress handler stopped closes its round
 *  as a stop does, so its report stands beside the @c usearch_status_cancelled_k it answers.
 */
typedef struct usearch_refine_report_t {

    /** Entries the restitch selected neighbors for, every entry of each neighborhood it reached. */
    size_t reselected;

    /** Of those, the entries whose neighbor set changed by any link added or dropped; order never
     *  counts. */
    size_t changed;

    /** Entries the first audit found not reached both ways, counted once. */
    size_t islands_found;

    /** Links written between an island and an entry the audit reached both ways, summed across
     *  the heal's audits. */
    size_t islands_reconnected;

    /** Neighborhoods the round re-stitched, and the widest one's entries. */
    size_t neighborhoods;
    size_t widest_neighborhood;

    /** Links the round added, and links it dropped. */
    size_t added_links;
    size_t dropped_links;

    /** What the round concluded. */
    usearch_refine_verdict_t verdict;

    /** The health check opening the round. */
    usearch_reachability_t opening;

    /** The health check closing the heal, as a fresh @c health call right after would take it. */
    usearch_reachability_t reachability;
} usearch_refine_report_t;

/**
 *  @brief How a refining verb may spend, shared by every verb that refines.
 *
 *  A round sizes its hubs and its neighborhoods from the cache of each core it runs on: a CPU
 *  core's share of its cache, a GPU multiprocessor's shared memory, an Apple core's threadgroup
 *  memory. Nothing here tunes it; only what reaches past the round is named.
 */
typedef struct usearch_refine_options_t {

    /** The @b optional breadth of any walk a refining verb takes beside its rounds; zero derives. A
     *  round itself walks nothing. */
    size_t expansion_factor;

    /** The @b optional bytes a loaded NVMe handle holds for the one part it rewrites: zero derives
     *  what rounds that part in RAM, and a smaller one streams the part's rounds from the drives.
     *  RAM and VRAM rounds never read it. */
    size_t memory_limit;
} usearch_refine_options_t;

/**
 *  @brief The host memory domains a RAM-resident index places its shards in - one shard per
 *      named NUMA node, so a dual-socket box names two and gets two graphs whose traversals
 *      stay local to the cores that walk them.
 *
 *  Names where the index lives, never who walks it: threads belong to the executor the verbs
 *  take. Every modality's RAM lifecycle verbs and the RAM executor take this struct.
 *
 *  The library's own payload blocks of 1 GiB or more claim 1 GiB huge pages while the host's 1 GiB
 *  pool covers them, and every other block base pages; 2 MiB pages are never claimed.
 */
typedef struct usearch_ram_options_t {

    /** NUMA node IDs to place shards on, read only during the call;
     *  @c NULL derives every node the process may allocate on. */
    int32_t const* memory_domains;

    /** Number of entries in @c memory_domains; zero derives. */
    size_t memory_domains_count;

    /** Entries one shard may hold before the index opens another shard generation on the same
     *  domains; zero, or anything past the 32-bit ID space, means 2^32 - 2. */
    size_t shard_capacity;

    /** The newest class dispatch may pick, intersected with what the CPU runs, so a class the CPU
     *  lacks narrows the pick rather than refusing it; @c usearch_cpu_class_unknown_k sets none. */
    usearch_cpu_class_t cpu_class_ceiling;

    /** Bytes an NVMe view's key table may hold, on the first named domain or the heap without one;
     *  zero sets no ceiling. */
    size_t memory_limit;
} usearch_ram_options_t;

/**
 *  @brief The devices a VRAM-resident index places its shards on - one shard per named ordinal,
 *      sharded and never replicated, so eight devices host eight slices of one graph rather than
 *      eight copies of it.
 *
 *  Names where the index lives, never who walks it: streams belong to the executor the verbs
 *  take, one per device. A checkpoint of more parts than named devices keeps one shard per part,
 *  part @c p on device @c p modulo their count, and shards sharing a device queue on its stream
 *  in order.
 *
 *  Flat matrices are device-resident, never migratable: an index outgrowing the named devices
 *  fails rather than paging to the host, and the answer is to name more devices.
 *
 *  One index spans one vendor and runs the best class all its devices load, never above
 *  @c gpu_class_ceiling; devices of two vendors, or with no class in common, refuse. Separate
 *  indexes may sit on different vendors in one process.
 */
typedef struct usearch_vram_options_t {

    /** USearch device ordinals to place shards on, NVIDIA devices first, then AMD, then Apple, as
     *  @c usearch_gpu_devices lists them, read only during the call; @c NULL picks device zero. */
    int32_t const* devices;

    /** Number of entries in @c devices; zero derives. */
    size_t devices_count;

    /** Entries one shard may hold before the index opens another shard generation on the same
     *  devices, while their memory lasts; zero, or anything past the 32-bit ID space, means
     *  2^32 - 2. */
    size_t shard_capacity;

    /** The newest class dispatch may pick, capping the best class every named device loads, so a
     *  ceiling no named device loads is refused; @c usearch_gpu_class_unknown_k sets none. */
    usearch_gpu_class_t gpu_class_ceiling;
} usearch_vram_options_t;

/**
 *  @brief The devices an NVMe-resident index serves its shards from - one shard per named path,
 *      so striping across four drives names four.
 *
 *  Names where the index lives, never who walks it: submission depth belongs to the executor the
 *  verbs take. Naming nothing serves every shard where its manifest already points.
 *
 *  A path is whatever identifies a device to the engine that can reach it, so the string is
 *  the extension point rather than a field per transport. A filesystem path names a kernel
 *  device; a transport identifier like @c trtype:PCIe @c traddr:0000:5e:00.0 names a
 *  namespace the kernel no longer owns and selects a userspace engine. A FUSE mount is just
 *  a path too, which is how object storage stages in without an HTTP stack in this library -
 *  point @c mountpoint-s3 at a bucket and the ordinary engines read it.
 *
 *  Transfer geometry is never an input: alignment is probed from the device, and the record
 *  granularity a checkpoint was written at is read back from its own header.
 *
 *  Reads always bypass the page cache where the filesystem admits @c O_DIRECT.
 */
typedef struct usearch_nvme_options_t {

    /** Devices or directories to serve shards from, read only during the
     *  call; @c NULL leaves every shard where its manifest points. */
    usearch_bytes_view_t const* paths;

    /** Number of entries in @c paths; zero derives. */
    size_t paths_count;

    /** The newest class dispatch may pick, intersected with what the CPU runs, so a class the CPU
     *  lacks narrows the pick rather than refusing it; @c usearch_cpu_class_unknown_k sets none. */
    usearch_cpu_class_t cpu_class_ceiling;
} usearch_nvme_options_t;

/** Whether a predicate admits one candidate key or passes over it. */
typedef enum usearch_admission_t {

    /** Skip the key; the walk still routes through it. */
    usearch_reject_k = 0,

    /** Report the key among the matches. */
    usearch_admit_k = 1,
} usearch_admission_t;

/** Admits or rejects @p count candidate keys at once, never zero, writing one admission per key
 *  into @p admissions, a buffer the library owns for the call and fills with @c usearch_reject_k
 *  before every call, so a key the handler leaves unanswered stays rejected. Called concurrently
 *  from the executor's threads. */
typedef void (*usearch_predicate_callback_t)(usearch_key_t const* keys, usearch_admission_t* admissions, size_t count,
                                             void* state);

/** What one call costs the predicate's handler, which decides when the library asks it. */
typedef enum usearch_predicate_cost_t {

    /** Asked as candidates are scored, a few keys a call, with nothing buffered. */
    usearch_predicate_cost_cheap_k = 0,

    /** Asked once per round a thread settles, about every candidate it gathered, for handlers
     *  that pay per call, such as a lookup in an external database. */
    usearch_predicate_cost_expensive_k = 1,
} usearch_predicate_cost_t;

/** Which candidate keys a search may report: a handler, the state it is called with, and what a
 *  call costs it. */
typedef struct usearch_predicate_t {

    /** The @b optional handler; @c NULL admits every key. */
    usearch_predicate_callback_t callback;

    /** Opaque pointer forwarded into every @c callback call. */
    void* state;

    /** What one @c callback call costs the handler; zero is cheap. */
    usearch_predicate_cost_t cost;
} usearch_predicate_t;

/** Whether the operation a handler is watching continues or unwinds. */
typedef enum usearch_continuation_t {

    /** Carry on to the next unit of work. */
    usearch_continue_k = 0,

    /** Stop and report @c usearch_status_cancelled_k; whether finished increments stand
     *  is each verb's own contract. */
    usearch_cancel_k = 1,
} usearch_continuation_t;

/**
 *  @brief Periodic report of a long operation's advance, and the caller's chance to stop it.
 *      Counts are in whatever units the verb consumes - entries for an insertion, queries for a
 *      search, shards for a checkpoint - so one handler serves every one of them.
 *
 *  Never invoked concurrently, however many threads perform the work, so the handler needs no
 *  synchronization of its own - and never after the call it was passed to stores its final status.
 *
 *  Arguments, in order:
 *  - @c completed: units finished so far, never decreasing within one phase.
 *  - @c total: units the call finishes unless cancelled.
 *  - @c state: the opaque pointer registered alongside the handler, forwarded verbatim.
 */
typedef usearch_continuation_t (*usearch_progress_callback_t)( //
    size_t completed, size_t total, void* state);

/**
 *  @brief How a long call reports its advance: a handler, its state, and how many reports it hears.
 *
 *  A batch verb splits its units into @c steps nearly equal slices, hands each to its engine whole
 *  and reports after each; a verb whose engine counts its own units, a checkpoint's shards or each
 *  phase of a round, reports at every barrier its engine reaches, so a long phase stays watchable.
 *  Without a handler the call runs as one slice and reports nothing.
 */
typedef struct usearch_progress_t {

    /** The @b optional handler; @c NULL skips every report. */
    usearch_progress_callback_t callback;

    /** Opaque pointer forwarded into every @c callback call. */
    void* state;

    /** Slices a batch verb splits its units into, reporting after each; zero derives to ten. */
    size_t steps;
} usearch_progress_t;

/**
 *  @brief Blocks on an atomic wait until @p outcome is final, with acquire ordering.
 *  @param[in] outcome An outcome passed to a call, read until its status leaves @c usearch_status_deferred_k.
 *  @return The final status.
 *
 *  Never wait on a call's own outcome from inside its progress handler or predicate: the call
 *  finishes only after they return.
 */
USEARCH_EXPORT usearch_status_t usearch_outcome_wait(usearch_outcome_t* outcome);

/**
 *  @brief Reads the status of @p outcome without waiting, with the acquire ordering its final
 *      store pairs with, so a final status also publishes every output of the call.
 *  @param[in] outcome An outcome passed to a call.
 *  @return The final status once the call is final.
 *
 *  The one way to poll: the status word is atomic only by this contract, so a caller reading the
 *  field directly needs an atomic load its language may not offer on foreign memory.
 */
USEARCH_EXPORT usearch_status_t usearch_outcome_status(usearch_outcome_t* outcome);

/** Whether a call returns with its final status, or once submitted to its executor. */
typedef enum usearch_completion_t {

    /** Return with the final status stored and every output readable; the default. */
    usearch_completion_blocking_k = 0,

    /** Return once submitted, the status deferred until final; the caller waits or polls. */
    usearch_completion_deferred_k = 1,
} usearch_completion_t;

/**
 *  @brief Everything a call needs to run but an index does not own: the host threads or device
 *      streams that advance the work, the traversal scratch each of them walks with, and the
 *      pinned bounce buffer staging passes through.
 *
 *  One executor serves any number of index handles, so separate categories of indexes share one
 *  machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
 *  constructor that produced it and must match the residency of every handle it drives; a mismatch
 *  reports @c usearch_status_device_k.
 *
 *  Scratch is carved one slice per thread or stream, sized from geometry rather than
 *  from the scalar type, which is what lets a single executor drive indexes of different
 *  quantizations and metrics. It grows to the widest geometry asked of it, is rewound per call,
 *  and is kept between calls, so a steady workload allocates once.
 *
 *  One call holds an executor at a time, blocking or deferred, and handles stay freely concurrent -
 *  it is the executor that serializes, never the index. A call submitted while another holds the
 *  executor finishes @c usearch_status_contended_k at once, and nothing queues, so submitting never
 *  allocates. A caller-exclusive pool runs a deferred call on its own thread 0, which forks the
 *  other threads as the caller would, so the call finishes without its caller. An executor without
 *  threads of its own - caller-inclusive, single-thread, or WebAssembly - runs a deferred call at
 *  submission, final when it returns, and so does every VRAM verb.
 *
 *  Progress handlers and predicates of a deferred call run on the pool's threads. A blocking verb
 *  called from inside one of them on the same executor reports @c usearch_status_contended_k, and
 *  so does @c usearch_executor_drain on the thread running the call's body; a drain from any other
 *  of its threads would wait on itself.
 *
 *  Every verb accepts a @c NULL executor and then runs on the calling thread, blocking whatever
 *  its @c completion says: RAM work runs single-threaded, VRAM work runs on each device's null
 *  legacy stream and synchronizes before returning, and NVMe reads block through the files the
 *  index already holds. An index reading through SPDK answers @c usearch_status_invalid_argument_k
 *  to a @c NULL executor, since no blocking read reaches its namespaces. Its outputs land in any
 *  arena of the handle's residency.
 *
 *  The library stages only to change layout on the side that computes, or to make pageable memory
 *  DMA-able, and never routes device bytes through the host:
 *
 *      Case                                          Action
 *      Device inputs on the executor's device        read in place
 *      Device inputs needing padding or a cast       a device-to-device copy or a cast kernel
 *      Pinned or managed host inputs                 a direct asynchronous upload
 *      Pageable host inputs                          the pinned bounce buffer, then an upload
 *      Outputs in an arena on the executor's device  the kernel writes them directly
 */
typedef struct usearch_executor_handle_t* usearch_executor_t;

/** The arguments of @c usearch_executor_for_ram. */
typedef struct usearch_executor_for_ram_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The memory domains of the indexes it will drive; copied at creation, so the caller may free
     *  it once the call returns. */
    usearch_ram_options_t ram;

    /** Threads to spawn; zero takes 7/8 of the cores those domains hold. */
    size_t threads;

    /** The @b optional ceiling on each thread's scratch, which every call's walks and staging carve
     *  from, past which a call reports @c usearch_status_out_of_memory_k; zero takes 1 GiB. */
    size_t scratch_limit;
} usearch_executor_for_ram_t;

/**
 *  @brief Creates an executor of host threads for RAM-resident indexes, interleaved across the
 *      compute domains local to the named memory domains.
 *
 *  Passing the very placement the indexes were built with is what makes compute and memory domains
 *  agree, leaving no second place to name hardware and no way for the two to disagree.
 *
 *  Interleaving is why a half-size executor takes half of every socket rather than all of one, and
 *  the default stops at 7/8 of the cores because full occupancy starves the ordered publication
 *  path. Threads pin to the first cores the creating thread may run on, so a caller making several
 *  executors narrows its affinity mask before each, giving every executor its own core set.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c cpu_class_ceiling no class takes.
 */
USEARCH_EXPORT void usearch_executor_for_ram(usearch_executor_for_ram_t* arguments);

/** The arguments of @c usearch_executor_for_vram. */
typedef struct usearch_executor_for_vram_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The devices of the indexes it will drive; copied at creation, so the caller may free it once
     *  the call returns. */
    usearch_vram_options_t vram;

    /** The @b optional CUDA streams of the caller, one per device in @c vram order, which are not
     *  wired yet; @c NULL lets the executor open its own. */
    void* const* streams;

    /** The @b optional ceiling on each stream's run, which every call's scratch and outputs carve
     *  from, past which a call reports @c usearch_status_out_of_memory_k; zero lets the run grow as
     *  the geometry asks. */
    size_t scratch_limit;
} usearch_executor_for_vram_t;

/**
 *  @brief Opens one submission stream on every distinct device @c vram names, which it destroys
 *      when freed, all of them driven by whichever host thread submits.
 *
 *  One stream per device is the whole design: every shard on a device queues on its stream in
 *  order, a single host thread keeps every device fed, and adding host threads would buy nothing,
 *  since the work has already left for the accelerator by the time one of them would return. A
 *  GPU executor never changes process-wide device flags.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c gpu_class_ceiling no class takes.
 *  - @c usearch_status_unsupported_k Caller @c streams: the executor drives only its own.
 *  - @c usearch_status_device_k An ordinal past the listing, no device named, or a device refused
 *    its stream.
 *  - @c usearch_status_config_mismatch_k Devices of more than one vendor.
 */
USEARCH_EXPORT void usearch_executor_for_vram(usearch_executor_for_vram_t* arguments);

/** The arguments of @c usearch_gpu_devices. */
typedef struct usearch_gpu_devices_t {

    /** Output for the outcome, final when the call returns. */
    usearch_outcome_t* outcome;

    /** Output for up to @c devices_capacity devices, in USearch ordinal order; may be @c NULL. */
    usearch_gpu_device_t* devices;

    /** Entries @c devices holds. */
    size_t devices_capacity;

    /** Output for every device this process can drive, which may exceed @c devices_capacity. */
    size_t* devices_count;
} usearch_gpu_devices_t;

/**
 *  @brief Lists every GPU this process can drive, NVIDIA devices first, then AMD, then Apple, so
 *      the position of a device is the USearch ordinal @c usearch_vram_options_t names it by.
 *
 *  Each vendor's runtime loads on first use, and a vendor whose runtime is absent lists no devices,
 *  so the call succeeds with a zero count on a machine without GPUs. The listing is taken once per
 *  process and never changes after.
 */
USEARCH_EXPORT void usearch_gpu_devices(usearch_gpu_devices_t* arguments);

/** The I/O engines USearch drives external memory through, newest first; an executor pinned to
 *  one tries it and degrades down the ladder the media admit. */
typedef enum usearch_nvme_class_t {

    /** Derives the newest engine the host and the media admit: io_uring, then Linux AIO, then
     *  blocking reads on kernel paths, and SPDK on transport identifiers. */
    usearch_nvme_class_unknown_k = 0,

    /** @c pread or @c ReadFile on the calling thread. */
    usearch_nvme_class_posix_k = 1,

    /** Linux AIO over @c io_submit and @c io_getevents, with @c O_DIRECT. */
    usearch_nvme_class_linux_aio_k = 2,

    /** io_uring reads on files and block devices. */
    usearch_nvme_class_linux_uring_k = 3,

    /** io_uring NVMe passthrough on namespace character devices, such as @c /dev/ng0n1, over the
     *  128-byte entries Linux 5.19 introduced. */
    usearch_nvme_class_linux_uring_passthrough_k = 4,

    /** The SPDK user-space driver, over namespaces named by transport identifiers. */
    usearch_nvme_class_linux_spdk_k = 5,

    /** I/O Rings, on Windows 11 22H2 and newer. */
    usearch_nvme_class_windows_ioring_k = 6,
} usearch_nvme_class_t;

/** The arguments of @c usearch_executor_for_nvme. */
typedef struct usearch_executor_for_nvme_t {

    /** Output for the new executor. */
    usearch_executor_t* executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** The devices of the indexes it will drive; copied at creation, so the caller may free it once
     *  the call returns. */
    usearch_nvme_options_t nvme;

    /** The memory domains its threads and their scratch sit on, whose first entry wins over the
     *  domain the devices sit on; @c NULL takes the devices' domain. Only @c memory_domains is
     *  read: @c shard_capacity, @c memory_limit and @c cpu_class_ceiling belong to RAM indexes, and
     *  @c nvme.cpu_class_ceiling caps the threads. */
    usearch_ram_options_t ram;

    /** Threads to spawn, one submission ring apiece; zero derives from the devices' queue capabilities. */
    size_t threads;

    /** The I/O engine every ring submits through, opened exactly or refused with the reason its
     *  setup gave; @c usearch_nvme_class_unknown_k tries io_uring, then AIO, then blocking reads.
     *  Passthrough and SPDK open only when named, since one needs character devices and the other
     *  needs transport identifiers. */
    usearch_nvme_class_t nvme_class;

    /** The @b optional ceiling on each thread's scratch, which every call's walks and staging carve
     *  from, past which a call reports @c usearch_status_out_of_memory_k; zero takes 1 GiB. */
    size_t scratch_limit;
} usearch_executor_for_nvme_t;

/**
 *  @brief Creates an executor of host threads for NVMe-resident indexes, one submission ring
 *      apiece, near the first memory domain @c ram names, else the one the devices sit on.
 *
 *  One ring per thread is what keeps submission lock-free: a thread owns its ring outright, so
 *  an external-memory walk never coordinates with a sibling to issue a read. A ring reports its
 *  reads only to the thread that owns it, so deferred calls finish on those threads. The rings
 *  open as deep as the devices' queues admit and poll when every device polls; a later call
 *  needing more depth reopens them once. The same threads serve RAM-resident indexes too.
 *
 *  Threads pin to the first cores the creating thread may run on, so a caller making several
 *  executors narrows its affinity mask before each, giving every executor its own core set.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k @c NULL paths for a nonzero count, an unknown
 *    @c nvme_class, or a @c cpu_class_ceiling no class takes.
 *  - @c usearch_status_config_mismatch_k Kernel paths mixed with transport identifiers, SPDK
 *    pinned over kernel paths or a kernel engine over transport identifiers, or more threads than
 *    an SPDK controller granted I/O queue pairs.
 *  - @c usearch_status_unsupported_k Transport identifiers in a build without SPDK, or a pinned
 *    engine this build lacks.
 */
USEARCH_EXPORT void usearch_executor_for_nvme(usearch_executor_for_nvme_t* arguments);

/** How a load from a path read the parts onto the devices, as @c describe reports. */
typedef enum usearch_load_path_t {

    /** Never loaded from a path, or an index on RAM or NVMe. */
    usearch_load_path_none_k = 0,

    /** Through host memory, since some part's file took no cuFile registration. */
    usearch_load_path_host_copy_k = 1,

    /** Every part straight onto the device through cuFile. */
    usearch_load_path_cufile_k = 2,
} usearch_load_path_t;

/** What an executor's threads do between a drain and its next call. */
typedef enum usearch_mood_t {

    /** Sleep in the kernel until the next call, freeing the cores. */
    usearch_mood_parked_k = 0,

    /** Keep polling, trading the cores for the lowest wake-up latency. */
    usearch_mood_spinning_k = 1,
} usearch_mood_t;

/** The arguments of @c usearch_executor_drain. */
typedef struct usearch_executor_drain_t {

    /** The executor whose call is awaited. */
    usearch_executor_t executor;

    /** Output for an executor-level failure, such as a device fault; never one of
     *  a drained call. */
    usearch_outcome_t* outcome;

    /** What the executor's threads do until its next call. */
    usearch_mood_t mood;
} usearch_executor_drain_t;

/** Waits until no call holds the executor and its devices finished their work, then applies the
 *  mood. Reports only executor-level failures, and @c usearch_status_contended_k from a callback of
 *  the call it would wait on. */
USEARCH_EXPORT void usearch_executor_drain(usearch_executor_drain_t* arguments);

/** Waits until no call holds the executor and its devices finished their work, releases its
 *  scratch, and frees the executor; @c NULL is safe, and it cannot fail, so it takes no
 *  outcome. Every arena it made must be freed first. */
USEARCH_EXPORT void usearch_executor_free(usearch_executor_t executor);

/**
 *  @brief Memory the library writes every output whose length scales with a call into, reused
 *      across calls. @ref types_memory
 *
 *  Made by @c usearch_arena_init where its executor computes: host pages on its memory domains, or
 *  the memory of its GPU, and a GPU that runs out reports @c usearch_status_out_of_memory_k rather
 *  than falling back to the host. Each call given the arena rewinds it, which invalidates every
 *  output of the previous call, and refills it, growing only when the new outputs outsize
 *  everything it held, so a steady loop allocates once. To keep one call's outputs alive past the
 *  next call, pass the next call another arena.
 *
 *  - A @c NULL arena is refused with @c usearch_status_invalid_argument_k.
 *  - A call whose inputs point into the arena it is about to rewind reports
 *    @c usearch_status_invalid_argument_k.
 *  - An arena serves one call at a time; a second call naming it finishes
 *    @c usearch_status_contended_k.
 *  - A call on another residency or device than the arena's, or a call naming a GPU arena with
 *    another executor than the one that made it, @c NULL included, reports
 *    @c usearch_status_device_k.
 *  - A call holds the arena from its start until just before its final status.
 *  - A call that fails, is cancelled, or has nothing to report leaves its outputs @c NULL.
 *
 *  @note A GPU arena grows, fills and frees on the first stream of the executor that made it, so it
 *      serves only calls on that executor.
 */
typedef struct usearch_arena_handle_t* usearch_arena_t;

/** The arguments of @c usearch_arena_init. */
typedef struct usearch_arena_init_t {

    /** Output for the new arena. */
    usearch_arena_t* arena;

    /** The executor whose calls the arena serves, which decides where it lives; @c NULL serves calls
     *  on the calling thread from host pages. */
    usearch_executor_t executor;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;
} usearch_arena_init_t;

/**
 *  @brief Creates an empty arena where @c executor computes, so its placement is decided once and
 *      GPU code keeps its outputs on the device without asking.
 *
 *  Nothing is allocated until a call writes into it. An arena must not outlive the executor that
 *  made it: free every arena before its executor. An arena made with a @c NULL executor has none,
 *  and lives until it is freed.
 */
USEARCH_EXPORT void usearch_arena_init(usearch_arena_init_t* arguments);

/**
 *  @brief Frees the arena and every output in it; @c NULL is safe, and it cannot fail, so takes no
 *      outcome.
 *  @param[in,out] arena The arena to be freed; invalid afterwards.
 *
 *  Waits out every executor call naming the handle; never call it from their callbacks.
 *  Free a GPU arena once the caller's own device work on its outputs has finished, keeping the
 *  rule @ref usearch_arena_t notes.
 */
USEARCH_EXPORT void usearch_arena_free(usearch_arena_t arena);

/** Where an arena's outputs live. */
typedef struct usearch_arena_description_t {

    /** RAM for host pages, VRAM for one device's memory. */
    usearch_residency_t residency;

    /** The device's position in @c usearch_gpu_devices, on VRAM; 0 for host pages. */
    size_t device;
} usearch_arena_description_t;

/** The arguments of @c usearch_arena_describe. */
typedef struct usearch_arena_describe_t {

    /** The arena to describe. */
    usearch_arena_t arena;

    /** Output for the outcome, final once its status leaves @c usearch_status_deferred_k. */
    usearch_outcome_t* outcome;

    /** Output for where the arena's outputs live. */
    usearch_arena_description_t* description;
} usearch_arena_describe_t;

/**
 *  @brief Reports where an arena's outputs live, as the executor that made it decided, so a caller
 *      reads them as host memory or hands them to device code without remembering that executor.
 *
 *  @par Statuses:
 *  - @c usearch_status_invalid_argument_k A @c NULL arena or description.
 */
USEARCH_EXPORT void usearch_arena_describe(usearch_arena_describe_t* arguments);

/**
 *  @brief The canonical lowercase, dash-separated spelling of each closed set's enumerators,
 *      and the inverse that reads one back.
 *
 *  Every binding that touches user text otherwise writes these tables itself, against a header
 *  that nothing checks it against. The names are static storage and are never deallocated.
 *  A value no status holds spells "unrecognized".
 */
USEARCH_EXPORT char const* usearch_status_name(usearch_status_t status);

/** Reads a status back into @p status, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_status_from_name(usearch_bytes_view_t name, usearch_status_t* status);

/** The canonical spelling of a scalar kind, as @c usearch_scalar_f32_k spells "f32". */
USEARCH_EXPORT char const* usearch_scalar_name(usearch_scalar_t scalar);

/** Reads a scalar kind back into @p scalar, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_scalar_from_name(usearch_bytes_view_t name, usearch_scalar_t* scalar);

/** The canonical spelling of a residency, as @c usearch_residency_ram_k spells "ram". */
USEARCH_EXPORT char const* usearch_residency_name(usearch_residency_t residency);

/** Reads a residency back into @p residency, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_residency_from_name(usearch_bytes_view_t name, usearch_residency_t* residency);

/** The canonical spelling of a traversal, as @c usearch_traversal_scan_k spells "scan". */
USEARCH_EXPORT char const* usearch_traversal_name(usearch_traversal_t traversal);

/** Reads a traversal back into @p traversal, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_traversal_from_name(usearch_bytes_view_t name, usearch_traversal_t* traversal);

/** The canonical spelling of a mood, as @c usearch_mood_parked_k spells "parked". */
USEARCH_EXPORT char const* usearch_mood_name(usearch_mood_t mood);

/** Reads a mood back into @p mood, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_mood_from_name(usearch_bytes_view_t name, usearch_mood_t* mood);

/** The canonical spelling of a predicate cost, as @c usearch_predicate_cost_expensive_k spells
 *  "expensive". */
USEARCH_EXPORT char const* usearch_predicate_cost_name(usearch_predicate_cost_t cost);

/** Reads a predicate cost back into @p cost, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_predicate_cost_from_name(usearch_bytes_view_t name,
                                                                 usearch_predicate_cost_t* cost);

/** The canonical spelling of a refine verdict, as @c usearch_refine_stalled_k spells "stalled". */
USEARCH_EXPORT char const* usearch_refine_verdict_name(usearch_refine_verdict_t verdict);

/** Reads a refine verdict back into @p verdict, or fails with
 *  @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_refine_verdict_from_name(usearch_bytes_view_t name,
                                                                 usearch_refine_verdict_t* verdict);

/** The canonical spelling of a plan's phase, as @c usearch_phase_restitch_k spells "restitch". */
USEARCH_EXPORT char const* usearch_phase_name(usearch_phase_t phase);

/** Reads a plan's phase back into @p phase, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_phase_from_name(usearch_bytes_view_t name, usearch_phase_t* phase);

/** The canonical spelling of a load path, as @c usearch_load_path_cufile_k spells "cufile". */
USEARCH_EXPORT char const* usearch_load_path_name(usearch_load_path_t load_path);

/** The canonical spelling of a CPU class, as the dispatch ladder names its rungs. */
USEARCH_EXPORT char const* usearch_cpu_class_name(usearch_cpu_class_t cpu_class);

/** Reads a CPU class back into @p cpu_class, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_cpu_class_from_name(usearch_bytes_view_t name, usearch_cpu_class_t* cpu_class);

/** The canonical spelling of a GPU class, as the dispatch ladder names its rungs. */
USEARCH_EXPORT char const* usearch_gpu_class_name(usearch_gpu_class_t gpu_class);

/** Reads a GPU class back into @p gpu_class, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_gpu_class_from_name(usearch_bytes_view_t name, usearch_gpu_class_t* gpu_class);

/** The canonical spelling of an NVMe class, as the I/O engine ladder names its rungs. */
USEARCH_EXPORT char const* usearch_nvme_class_name(usearch_nvme_class_t nvme_class);

/** Reads an NVMe class into @p nvme_class, or fails with @c usearch_status_invalid_argument_k. */
USEARCH_EXPORT usearch_status_t usearch_nvme_class_from_name(usearch_bytes_view_t name,
                                                             usearch_nvme_class_t* nvme_class);

/**
 *  @brief Reports the version of the library this process linked, which may differ from the
 *      @c USEARCH_VERSION_MAJOR family of macros it was compiled against.
 *  @param[out] major The major version, which changes whenever an argument structure does.
 *  @param[out] minor The minor version.
 *  @param[out] patch The patch version.
 *
 *  None of the three outputs may be @c NULL.
 */
USEARCH_EXPORT void usearch_version(int* major, int* minor, int* patch);

/** The version of the library as a static null-terminated string, a candidate's suffix included,
 *  like "3.0.0" or "3.0.0-rc.1". */
USEARCH_EXPORT char const* usearch_version_name(void);

/**
 *  @brief Retrieves the kernel families the CPU classes this binary compiled reach.
 *  @return A comma-separated static string with capability names.
 */
USEARCH_EXPORT char const* usearch_capabilities_compiled(void);

/**
 *  @brief Retrieves the kernel families of the newest CPU class the current machine runs.
 *  @return A comma-separated static string with capability names, empty where the machine runs
 *      no class this build compiled, as a monoversioned build on an older CPU.
 */
USEARCH_EXPORT char const* usearch_capabilities_available(void);

#ifdef __cplusplus
}
#endif

#endif // UNUM_USEARCH_TYPES_H
