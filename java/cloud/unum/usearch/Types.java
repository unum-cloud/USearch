package cloud.unum.usearch;

import java.io.Serial;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;

import org.jspecify.annotations.Nullable;

/**
 * The vocabulary every modality shares - the Java twin of {@code include/usearch/types.h}: entry
 * identities, failure reporting, scalar kinds, the executor every fanned-out verb runs on, deferred
 * calls, and the strided descriptors batches travel in.
 *
 * <p>
 * The class itself is the library's introspection: its version, and the hardware it can and does
 * use. Everything else it holds is a nested type, so one file mirrors one header.
 */
public final class Types {

    private Types() {}

    /** The version of the linked library. Calls {@code usearch_version_name}. */
    public static String version() {
        return Library.text(VERSION);
    }

    /**
     * The hardware capabilities the linked binary was compiled with, comma separated. Calls
     * {@code usearch_capabilities_compiled}.
     */
    public static String capabilitiesCompiled() {
        return Library.text(CAPABILITIES_COMPILED);
    }

    /**
     * The hardware capabilities the current machine offers at runtime, comma separated. Calls
     * {@code usearch_capabilities_available}.
     */
    public static String capabilitiesAvailable() {
        return Library.text(CAPABILITIES_AVAILABLE);
    }

    /**
     * A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
     *
     * <p>
     * The sixteen bytes are a {@link UUID}'s two big-endian longs, so {@link #toUUID()} and
     * {@link #fromUUID(UUID)} convert without reordering anything.
     *
     * <p>
     * Keys need not be unique: several entries may share one, and {@link DenseVerbs#count(Key)}
     * reports how many. Integer identities park in the low eight bytes, so {@code Key.of(42)} prints
     * as {@code 00000000-0000-0000-0000-00000000002a}.
     */
    public static final class Key implements Comparable<Key> {

        /** The width of a key in bytes, and the row pitch of every key matrix crossing the boundary. */
        public static final int BYTES = 16;

        /** The all-zero key. */
        public static final Key NIL = new Key(new UUID(0, 0));

        /** One half of a key as the boundary lays it out, most significant byte first. */
        private static final ValueLayout.OfLong HALF = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.BIG_ENDIAN);

        private final UUID identity;

        private Key(UUID identity) {
            this.identity = identity;
        }

        /** Parks an integer identity in the low eight bytes, zeroing the rest. */
        public static Key of(long value) {
            return new Key(new UUID(0, value));
        }

        /** Adopts the identity a {@link UUID} spells, byte for byte. */
        public static Key fromUUID(UUID identity) {
            return new Key(identity);
        }

        /**
         * Copies sixteen bytes, most significant first.
         *
         * @throws USearchException INVALID_ARGUMENT when the slab is shorter than a key.
         */
        public static Key ofBytes(byte[] source) {
            return ofBytes(source, 0);
        }

        /**
         * Copies sixteen bytes starting at an offset, most significant first.
         *
         * @throws USearchException INVALID_ARGUMENT when no key fits at that offset.
         */
        public static Key ofBytes(byte[] source, int offset) {
            if (offset < 0 || source.length - BYTES < offset) {
                throw new USearchException(Status.INVALID_ARGUMENT, "the key does not fit at that offset");
            }
            return read(MemorySegment.ofArray(source), offset);
        }

        /**
         * Reads the rendering {@link #toString()} produces. Dashes may sit anywhere, or nowhere,
         * which is neither narrower nor wider than {@link UUID#fromString(String)} but different:
         * that one demands the five dashed groups and lets each be short.
         *
         * @throws USearchException MALFORMED when the text is not thirty-two hexadecimal digits.
         */
        public static Key parse(String text) {
            byte[] digits = new byte[BYTES * 2];
            int written = 0;
            for (int position = 0; position < text.length(); ++position) {
                char character = text.charAt(position);
                if (character == '-') {
                    continue;
                }
                if (written == digits.length) {
                    throw new USearchException(Status.MALFORMED, "the key is not thirty-two hexadecimal digits");
                }
                digits[written++] = hexadecimalDigit(character);
            }
            if (written != digits.length) {
                throw new USearchException(Status.MALFORMED, "the key is not thirty-two hexadecimal digits");
            }
            long high = 0;
            long low = 0;
            for (int position = 0; position < BYTES; ++position) {
                high = (high << 4) | digits[position];
                low = (low << 4) | digits[position + BYTES];
            }
            return new Key(new UUID(high, low));
        }

        private static byte hexadecimalDigit(char character) {
            if (character >= '0' && character <= '9') {
                return (byte) (character - '0');
            }
            if (character >= 'a' && character <= 'f') {
                return (byte) (character - 'a' + 10);
            }
            if (character >= 'A' && character <= 'F') {
                return (byte) (character - 'A' + 10);
            }
            throw new USearchException(Status.MALFORMED, "the key is not thirty-two hexadecimal digits");
        }

        /** The identity as a {@link UUID}, which is the same sixteen bytes. */
        public UUID toUUID() {
            return identity;
        }

        /** Copies the sixteen bytes out, most significant first. */
        public byte[] toBytes() {
            byte[] bytes = new byte[BYTES];
            writeInto(bytes, 0);
            return bytes;
        }

        /** Writes the sixteen bytes into a slab at an offset, most significant first. */
        public void writeInto(byte[] destination, int offset) {
            write(MemorySegment.ofArray(destination), offset);
        }

        /** Reports whether the high eight bytes are all zero, so the key names an integer identity. */
        public boolean isIntegral() {
            return identity.getMostSignificantBits() == 0;
        }

        /**
         * Reports the integer identity parked in the low eight bytes.
         *
         * @throws USearchException MALFORMED when the high eight bytes carry anything, so no
         *     integer describes the key; {@link #isIntegral()} asks first.
         */
        public long toLong() {
            if (!isIntegral()) {
                throw new USearchException(Status.MALFORMED, "the key carries more than an integer identity");
            }
            return identity.getLeastSignificantBits();
        }

        /** Renders the key as a lowercase dashed UUID. */
        @Override
        public String toString() {
            return identity.toString();
        }

        /** Two keys are equal when their sixteen bytes are. */
        @Override
        public boolean equals(Object other) {
            return other instanceof Key sibling && identity.equals(sibling.identity);
        }

        /** Hashes the same sixteen bytes {@link #equals(Object)} compares, so the two agree. */
        @Override
        public int hashCode() {
            return identity.hashCode();
        }

        /**
         * Orders keys by their bytes, most significant first - the order their rendering sorts in.
         * {@link UUID#compareTo(UUID)} reads those bytes as signed longs and is not that order.
         */
        @Override
        public int compareTo(Key other) {
            int high = Long.compareUnsigned(identity.getMostSignificantBits(), other.identity.getMostSignificantBits());
            return high != 0
                    ? high
                    : Long.compareUnsigned(
                            identity.getLeastSignificantBits(), other.identity.getLeastSignificantBits());
        }

        /** Reads the sixteen bytes at an offset of any segment, most significant first. */
        static Key read(MemorySegment source, long offset) {
            return new Key(new UUID(source.get(HALF, offset), source.get(HALF, offset + 8)));
        }

        /** Writes the sixteen bytes at an offset of any segment, most significant first. */
        void write(MemorySegment target, long offset) {
            target.set(HALF, offset, identity.getMostSignificantBits());
            target.set(HALF, offset + 8, identity.getLeastSignificantBits());
        }
    }

    /**
     * What went wrong, in a vocabulary shared by the engine and by this binding - one enumerator per
     * {@code usearch_status_*_k}, the deferred one included, which only a deferred call reports
     * before it is final. Validation performed here before a call crosses reports the status the
     * engine would have reported for the same fault.
     */
    public enum Status {

        /** The call succeeded and carries no message. */
        SUCCESS(0),
        /** The call was submitted deferred and is not final yet; the one non-final status. */
        DEFERRED(1),
        /** An allocation or reservation could not be satisfied. */
        OUT_OF_MEMORY(2),
        /** Dimensionality, row counts, or output capacity disagree. */
        SHAPE_MISMATCH(3),
        /** A stride or packing the engine cannot traverse. */
        UNSUPPORTED_LAYOUT(4),
        /** A scalar, metric, or residency tag this build does not know. */
        UNKNOWN_KIND(5),
        /** A mutating verb reached a handle that serves its checkpoint in place. */
        READ_ONLY(6),
        /** A file could not be opened, read, written, or parsed. */
        IO(7),
        /** A device could not be selected, or its memory could not be claimed. */
        DEVICE(8),
        /** A caller-supplied representation could not be decoded. */
        MALFORMED(9),
        /** The library reported neither success nor a failure. */
        INTERNAL(10),
        /** A hard design ceiling was reached, like the 32-bit identity space. */
        OVERFLOW(11),
        /** The verb needs a prior successful reservation that never happened. */
        UNINITIALIZED(12),
        /** The request contradicts an already-frozen configuration. */
        CONFIG_MISMATCH(13),
        /** A synchronization contract was violated - a full deferred table, or a busy arena. */
        CONTENDED(14),
        /**
         * A progress handler asked the operation to stop; it unwound and published nothing, so the
         * handle carries whatever it did before the call.
         */
        CANCELLED(15),
        /** The verb is declared for every residency but this one does not implement it. */
        UNSUPPORTED(16),
        /** An argument breaks the verb's contract, like a required one left null, or a status no build names. */
        INVALID_ARGUMENT(17);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the status a code carries; one this build cannot name reads as INVALID_ARGUMENT, as in C. */
        public static Status of(int code) {
            for (Status candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return INVALID_ARGUMENT;
        }

        /** Renders the status as its C enumerator's stem. Calls {@code usearch_status_name}. */
        @Override
        public String toString() {
            return Library.text(STATUS_NAME, code);
        }
    }

    /**
     * One failed call: a status every caller can branch on, and the detail the status cannot carry.
     * Nearly every verb can report one, so it is unchecked.
     */
    public static final class USearchException extends RuntimeException {

        @Serial
        private static final long serialVersionUID = 1L;

        private final Status status;

        /** Builds the failure a status and its detail describe. */
        public USearchException(Status status, String message) {
            super("usearch: " + status + ": " + message);
            this.status = status;
        }

        /** The status this failure carries. */
        public Status status() {
            return status;
        }
    }

    /**
     * A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
     * dimensions per byte, as their {@code xN} suffixes spell.
     *
     * <p>
     * The enumerators are ordered by shrinking dynamic range, and their codes carry the numbering gaps
     * the C enum reserves for kinds that have not landed yet.
     */
    public enum Scalar {

        /** A type the library could not name. */
        UNKNOWN(0, 0, 1),
        /** 64-bit double precision. */
        F64(1, 8, 1),
        /** 32-bit single precision. */
        F32(2, 4, 1),
        /** The brain floating-point format. */
        BF16(3, 2, 1),
        /** IEEE half precision. */
        F16(4, 2, 1),
        /** 8-bit floating point with five exponent bits. */
        E5M2(5, 1, 1),
        /** 8-bit floating point with four exponent bits. */
        E4M3(6, 1, 1),
        /** 6-bit floating point with three exponent bits, one per byte. */
        E3M2(7, 1, 1),
        /** 6-bit floating point with two exponent bits, one per byte. */
        E2M3(8, 1, 1),
        /** An 8-bit signed integer. */
        I8(10, 1, 1),
        /** Two 4-bit signed integers per byte. */
        I4X2(11, 1, 2),
        /** An 8-bit unsigned integer. */
        U8(20, 1, 1),
        /** Two 4-bit unsigned integers per byte. */
        U4X2(21, 1, 2),
        /** Eight single-bit dimensions per byte. */
        U1X8(22, 1, 8);

        private final int code;
        private final int valueBytes;
        private final int dimensionsPerValue;

        Scalar(int code, int valueBytes, int dimensionsPerValue) {
            this.code = code;
            this.valueBytes = valueBytes;
            this.dimensionsPerValue = dimensionsPerValue;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** The byte width of one stored value. */
        public int valueBytes() {
            return valueBytes;
        }

        /** How many logical dimensions one stored value carries. */
        public int dimensionsPerValue() {
            return dimensionsPerValue;
        }

        /** Names the kind a code carries; a code this build cannot name becomes UNKNOWN. */
        public static Scalar of(int code) {
            for (Scalar candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return UNKNOWN;
        }

        /**
         * The bytes one row of the given width occupies, a partly filled trailing value included.
         *
         * @throws USearchException OVERFLOW when the product does not fit in sixty-four bits.
         */
        public long rowBytes(long dimensions) {
            long values = dimensions / dimensionsPerValue + (dimensions % dimensionsPerValue != 0 ? 1 : 0);
            return Extents.product(values, valueBytes, "the row width");
        }

        /** Renders the scalar kind as its C enumerator's stem. Calls {@code usearch_scalar_name}. */
        @Override
        public String toString() {
            return Library.text(SCALAR_NAME, code);
        }
    }

    /** What an executor does with its threads or streams once it goes idle. */
    public enum Mood {

        /** Idle threads sleep; the next call pays a wake-up. */
        PARKED(0),
        /** Idle threads stay hot, trading power for the next call's latency. */
        SPINNING(1);

        private final int code;

        Mood(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Renders the mood as its C enumerator's stem. Calls {@code usearch_mood_name}. */
        @Override
        public String toString() {
            return Library.text(MOOD_NAME, code);
        }
    }

    /**
     * The workers every fanned-out verb runs on, shared across indexes: an executor owns host threads
     * or device streams and their scratch, an index owns rows.
     *
     * <p>
     * One factory per C constructor, each taking every argument its own kind takes: memory domains
     * and a thread count for RAM, a borrowed ForkUnion pool, device ordinals and optional streams for
     * VRAM, where a single host thread drives one stream per device, and device paths and a thread
     * count for NVMe. Naming no hardware derives it, a zero deferred limit takes the default size of
     * the deferred-call table, and the kind must match the residency of every handle it drives.
     *
     * <p>
     * A receiver names the executor its verbs run on, and the index alone runs them on the caller's
     * thread. Calls on one executor run in submission order; handles stay freely concurrent.
     *
     * <pre>{@code
     * try (Executor pool = Executor.forRam(new int[] {0, 1}, 16, 0);
     *         DenseIndex index = DenseIndex.initRam(new DenseOptions().dimensions(256), new int[0])) {
     *     index.on(pool).insert(keys, vectors);
     *     pool.drain(Mood.PARKED);
     * }
     * }</pre>
     */
    public static final class Executor implements AutoCloseable {

        private final Handle handle;

        private Executor(MemorySegment spawned) {
            handle = new Handle(this, spawned, EXECUTOR_FREE);
        }

        /**
         * Host workers for RAM-resident indexes, interleaved across the compute domains local to the
         * named NUMA nodes, so a dual-socket box names two and gets workers local to the shards they
         * walk.
         *
         * <p>
         * Calls {@code usearch_executor_for_ram}.
         *
         * @param memoryDomains NUMA node ids to place shards on; an empty list derives every node.
         * @param threads Workers to spawn; zero takes 7/8 of the cores those domains carry, since full
         *     occupancy starves the ordered publication path.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         * @throws USearchException INVALID_ARGUMENT on a negative count, and OUT_OF_MEMORY when the
         *     scratch cannot be claimed.
         */
        public static Executor forRam(int[] memoryDomains, long threads, long deferredLimit) {
            return spawn(FOR_RAM, call -> {
                call.ordinals("ram", memoryDomains);
                call.size("threads", unsigned(threads));
                call.size("deferred_limit", unsigned(deferredLimit));
            });
        }

        /**
         * A RAM executor over a borrowed ForkUnion pool, so several libraries in one process share one
         * set of threads while each keeps its own scratch. The pool keeps its own threads and its
         * lender's mood.
         *
         * <p>
         * Calls {@code usearch_executor_on_pool}.
         *
         * @param pool The {@code fu_pool_t} to borrow, which must outlive the executor.
         * @param memoryDomains NUMA node ids of the indexes it will drive; an empty list derives them.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         */
        public static Executor onPool(MemorySegment pool, int[] memoryDomains, long deferredLimit) {
            return spawn(ON_POOL, call -> {
                call.ordinals("ram", memoryDomains);
                call.pointer("pool", pool);
                call.size("deferred_limit", unsigned(deferredLimit));
            });
        }

        /**
         * One submission stream on every named device, all of them driven by the one thread that calls
         * the verbs, which is why this takes no thread count: the work has already left for the
         * accelerator by the time a second host thread would return.
         *
         * <p>
         * Calls {@code usearch_executor_for_vram}.
         *
         * @param devices CUDA device ordinals to place shards on; an empty list derives one free device.
         * @param streams No streams, or one borrowed CUDA stream per device in {@code devices} order,
         *     each outliving the executor.
         * @param scratchLimit The ceiling on scratch bytes per device; zero lets scratch grow.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         * @throws USearchException SHAPE_MISMATCH when the streams are neither absent nor one per
         *     device, DEVICE when a named device does not answer, and OUT_OF_MEMORY when the scratch
         *     cannot be claimed.
         */
        public static Executor forVram(int[] devices, MemorySegment[] streams, long scratchLimit, long deferredLimit) {
            // The C side reads one stream per device from any non-null pointer, so only this side sees a short list.
            if (streams.length != 0 && streams.length != devices.length) {
                throw new USearchException(Status.SHAPE_MISMATCH, "pass no streams or one per device");
            }
            return spawn(FOR_VRAM, call -> {
                call.ordinals("vram", devices);
                if (streams.length != 0) {
                    MemorySegment borrowed = call.memory.allocate(Library.POINTER, streams.length);
                    for (int order = 0; order != streams.length; ++order) {
                        borrowed.setAtIndex(Library.POINTER, order, streams[order]);
                    }
                    call.pointer("streams", borrowed);
                }
                call.size("scratch_limit", unsigned(scratchLimit));
                call.size("deferred_limit", unsigned(deferredLimit));
            });
        }

        /**
         * Host workers for NVMe-resident indexes, one submission ring apiece, across the compute
         * domains local to the devices the paths name - so striping across four drives names four.
         *
         * <p>
         * A path is whatever identifies a device to the engine that can reach it: a filesystem path
         * names a kernel device, and a transport identifier like {@code trtype:PCIe traddr:0000:5e:00.0}
         * names a namespace the kernel no longer owns.
         *
         * <p>
         * Calls {@code usearch_executor_for_nvme}.
         *
         * @param paths Devices or directories to place shards on; an empty list leaves every shard
         *     where its manifest points.
         * @param threads Workers to spawn; zero derives from those devices' queue capabilities.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         * @throws USearchException INVALID_ARGUMENT on a negative count, and OUT_OF_MEMORY when the
         *     scratch cannot be claimed.
         */
        public static Executor forNvme(String[] paths, long threads, long deferredLimit) {
            return spawn(FOR_NVME, call -> {
                call.paths("nvme", paths);
                call.size("threads", unsigned(threads));
                call.size("deferred_limit", unsigned(deferredLimit));
            });
        }

        /**
         * Waits for every call submitted before the drain began, then sets what the executor does
         * whenever it next goes idle; the last drain wins. Every call reports its own failure, so this
         * reports only what befell the executor itself.
         *
         * <p>
         * Calls {@code usearch_executor_drain}.
         *
         * @throws USearchException DEVICE on a lost device.
         * @throws IllegalStateException on a closed executor.
         */
        public void drain(Mood mood) {
            Call.block(DRAIN, null, this, call -> {
                call.enumeration("mood", mood.code());
                return () -> null;
            });
        }

        /**
         * Drains every thread or stream and releases them. It is idempotent, and a freed executor must
         * not be used again. One the program drops without closing is released once it is unreachable.
         *
         * <p>
         * Calls {@code usearch_executor_free}.
         */
        @Override
        public void close() {
            handle.close();
        }

        /** Refuses a negative count, which C would read as a size past any machine's. */
        private static long unsigned(long count) {
            if (count < 0) {
                throw new USearchException(Status.INVALID_ARGUMENT, "a count is negative");
            }
            return count;
        }

        /** Runs one executor factory and adopts what it spawned. */
        private static Executor spawn(Verb verb, java.util.function.Consumer<Call> fill) {
            return Call.block(verb, null, null, call -> {
                MemorySegment spawned = call.output("executor", Library.POINTER);
                fill.accept(call);
                return () -> new Executor(spawned.get(Library.POINTER, 0));
            });
        }

        /**
         * The handle a verb hands across.
         *
         * @throws IllegalStateException on a closed executor.
         */
        MemorySegment address() {
            return handle.address();
        }

        /** Renders the executor as the handle it wraps. */
        @Override
        public String toString() {
            return "Executor(" + handle + ")";
        }
    }

    /**
     * Memory the library allocates for the outputs only a call can size - the rows of
     * {@link DenseVerbs#get}, the products of {@link DenseVerbs#multiply}, the monolith of
     * {@link DenseVerbs#saveBuffer} - reused across calls.
     *
     * <p>
     * A fresh arena holds nothing; the first call handed it creates the memory where that call's
     * executor computes, and every later call rewinds and refills it, so a steady loop allocates once.
     * A result read out of an arena is valid until the arena is handed to another call or closed;
     * reading it after that raises {@link IllegalStateException} rather than touching freed memory.
     * To keep one call's result alive past the next, pass the next call another arena, or none, in
     * which case the result owns a fresh arena of its own and frees it when closed.
     *
     * <p>
     * An arena serves one call at a time and takes no lock: a second call submitted while the first
     * runs fails with CONTENDED. One the program drops without closing is freed once neither it nor
     * any result read out of it is reachable.
     */
    public static final class Arena implements AutoCloseable {

        /** The {@code usearch_arena_t} slot every call creates or rewinds the arena through. */
        private final MemorySegment handle = java.lang.foreign.Arena.ofAuto().allocate(Library.POINTER);

        /** The scope the previous call's results were read through, closed once they go stale. */
        private volatile java.lang.foreign.Arena outputs = java.lang.foreign.Arena.ofShared();

        /** Frees the library's memory once, on close or once nothing reaches the arena. */
        private final Cleaner.Cleanable release;

        /** An empty arena, which the first call handed it creates. */
        public Arena() {
            MemorySegment slot = handle;
            release = CLEANER.register(this, () -> {
                MemorySegment created = slot.get(Library.POINTER, 0);
                slot.set(Library.POINTER, 0, MemorySegment.NULL);
                Library.invoke(ARENA_FREE, created);
            });
        }

        /**
         * Invalidates the previous call's results and opens the scope the next call's results are
         * read through, which the call after it closes in turn.
         *
         * @throws IllegalStateException on a closed arena.
         */
        java.lang.foreign.Arena rewind() {
            outputs.close();
            outputs = java.lang.foreign.Arena.ofShared();
            return outputs;
        }

        /** The {@code usearch_arena_t} slot calls create or rewind the arena through. */
        MemorySegment slot() {
            return handle;
        }

        /**
         * Widens a pointer one call wrote to the {@code length} bytes it addresses, valid while the
         * scope that call's rewind opened stays open. The view keeps this arena reachable, so the
         * library's memory outlives it.
         *
         * @throws IllegalStateException once a later call rewound the arena, or it was closed.
         */
        @SuppressWarnings("restricted")
        MemorySegment reach(java.lang.foreign.Arena scope, MemorySegment pointer, long length) {
            return length == 0
                    ? MemorySegment.NULL
                    : pointer.reinterpret(length, scope, ignored -> Reference.reachabilityFence(this));
        }

        /**
         * Frees the memory and every result read out of it. It is idempotent.
         *
         * <p>
         * Calls {@code usearch_arena_free}.
         */
        @Override
        public void close() {
            java.lang.foreign.Arena stale = outputs;
            if (stale.scope().isAlive()) {
                stale.close();
            }
            release.clean();
        }
    }

    /**
     * The monolith a {@code saveBuffer} of any modality wrote into an arena, as
     * {@link DenseVerbs#saveBuffer} does. Valid until the arena is handed to another call or closed;
     * closing this frees the arena only when the call created it.
     */
    public static final class Monolith implements AutoCloseable {

        private final @Nullable Arena owned;
        private final MemorySegment bytes;

        Monolith(@Nullable Arena owned, MemorySegment bytes) {
            this.owned = owned;
            this.bytes = bytes;
        }

        /** The bytes in place, which {@link DenseIndex#loadRamBuffer} reads back with no copy. */
        public MemorySegment segment() {
            return bytes;
        }

        /** Frees the arena the call created; a borrowed one stays the caller's. */
        @Override
        public void close() {
            if (owned != null) {
                owned.close();
            }
        }
    }

    /**
     * A slab viewed as rows of equal width: a base, a row count, a logical column count, and a byte
     * pitch between consecutive row starts. It is the shape every batch argument takes, from
     * {@link DenseVerbs#insert} to {@link DenseVerbs#attend}.
     *
     * <p>
     * The slab is a {@link MemorySegment}: a heap one, as the array factories wrap, is copied into
     * native memory for the duration of one call and never retained, and a native one - a mapped
     * file, a direct buffer through {@link MemorySegment#ofBuffer}, an arena allocation - is
     * addressed in place with no copy. A view borrows and never owns, so a native slab must stay
     * open until every call it crosses is final.
     */
    public static final class RowsView {

        private final MemorySegment slab;
        private final long rows;
        private final long columns;
        private final long strideBytes;
        private final Scalar scalar;

        private RowsView(MemorySegment slab, long rows, long columns, long strideBytes, Scalar scalar) {
            this.slab = slab;
            this.rows = rows;
            this.columns = columns;
            this.strideBytes = strideBytes;
            this.scalar = scalar;
        }

        /**
         * Views a packed F64 slab as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static RowsView of(double[] data, long columns) {
            return of(MemorySegment.ofArray(data), columns, Scalar.F64);
        }

        /**
         * Views a packed F32 slab as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static RowsView of(float[] data, long columns) {
            return of(MemorySegment.ofArray(data), columns, Scalar.F32);
        }

        /**
         * Views a packed slab of sixteen-bit values as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static RowsView of(short[] data, long columns, Scalar scalar) {
            return of(MemorySegment.ofArray(data), columns, scalar);
        }

        /**
         * Views a packed slab of single-byte values as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static RowsView of(byte[] data, long columns, Scalar scalar) {
            return of(MemorySegment.ofArray(data), columns, scalar);
        }

        /**
         * Views a packed segment as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH on a zero width or a slab that does not divide into
         *     whole rows, and UNKNOWN_KIND on an unnamed kind.
         */
        public static RowsView of(MemorySegment data, long columns, Scalar scalar) {
            long rowBytes = rowBytesOf(columns, scalar);
            if (data.byteSize() % rowBytes != 0) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, data.byteSize() + " bytes do not divide into rows of " + rowBytes);
            }
            return new RowsView(data, data.byteSize() / rowBytes, columns, rowBytes, scalar);
        }

        /**
         * Views a slab whose pitch exceeds its row width - the padded layout a memory-mapped dataset
         * arrives in.
         *
         * @throws USearchException UNSUPPORTED_LAYOUT for a stride too narrow for one row, and
         *     SHAPE_MISMATCH when the rows overrun the slab.
         */
        public static RowsView strided(MemorySegment slab, long rows, long columns, long strideBytes, Scalar scalar) {
            long rowBytes = rowBytesOf(columns, scalar);
            if (strideBytes < rowBytes) {
                throw new USearchException(
                        Status.UNSUPPORTED_LAYOUT,
                        "a stride of " + strideBytes + " bytes cannot hold a row of " + rowBytes);
            }
            long span = rows == 0
                    ? 0
                    : Extents.sum(
                            Extents.product(rows - 1, strideBytes, "the matrix span"), rowBytes, "the matrix span");
            if (span > slab.byteSize()) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, rows + " rows of stride " + strideBytes + " overrun the slab");
            }
            return new RowsView(slab, rows, columns, strideBytes, scalar);
        }

        /**
         * Allocates a packed, zeroed native slab the garbage collector frees, which calls write into
         * in place.
         *
         * @throws USearchException SHAPE_MISMATCH on a zero width, UNKNOWN_KIND on an unnamed kind,
         *     and OVERFLOW when the slab would not fit in sixty-four bits.
         */
        public static RowsView allocate(long rows, long columns, Scalar scalar) {
            long rowBytes = rowBytesOf(columns, scalar);
            MemorySegment slab =
                    java.lang.foreign.Arena.ofAuto().allocate(Extents.product(rows, rowBytes, "the matrix"), 8);
            return new RowsView(slab, rows, columns, rowBytes, scalar);
        }

        private static long rowBytesOf(long columns, Scalar scalar) {
            if (columns <= 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "a matrix row cannot be zero columns wide");
            }
            if (scalar == Scalar.UNKNOWN) {
                throw new USearchException(Status.UNKNOWN_KIND, "the matrix names no scalar kind");
            }
            return scalar.rowBytes(columns);
        }

        /** How many rows the view spans. */
        public long rows() {
            return rows;
        }

        /** The logical width of one row, which the engine checks the frozen geometry against. */
        public long columns() {
            return columns;
        }

        /** The byte pitch between consecutive row starts. */
        public long strideBytes() {
            return strideBytes;
        }

        /** The scalar type the rows hold. */
        public Scalar scalar() {
            return scalar;
        }

        /** The slab the view borrows. */
        public MemorySegment segment() {
            return slab;
        }

        /** The bytes from the first row's start to the last row's end, which is all the slab must hold. */
        long spanBytes() {
            return rows == 0 ? 0 : (rows - 1) * strideBytes + scalar.rowBytes(columns);
        }

        /** Renders the view as the geometry it describes. */
        @Override
        public String toString() {
            return "RowsView(rows=" + rows + ", columns=" + columns + ", strideBytes=" + strideBytes + ", " + scalar
                    + ")";
        }
    }

    /**
     * Where a ranked run lands - three slabs holding a queries-by-wanted matrix of keys, the scores
     * positionally paired with them, and one match count per query.
     *
     * <p>
     * The keys are one flat byte slab, and {@link #key(int, int)} materializes a {@link Key} only for
     * the slots a caller reads. Slots past a query's count hold memory the engine never wrote, so
     * every accessor here refuses them.
     *
     * <p>
     * The wanted count is the "k" in kANN and the row width of both matrices, and it is read off the
     * slab lengths rather than asked for, so the room reserved and the count asked for cannot
     * disagree. Lending slabs the caller already owns is what lets a recall sweep allocate once and
     * reuse them across every point.
     */
    public static final class Matches {

        private final MemorySegment keys;
        private final MemorySegment scores;
        private final MemorySegment counts;
        private final int wanted;

        private Matches(MemorySegment keys, MemorySegment scores, MemorySegment counts, int wanted) {
            this.keys = keys;
            this.scores = scores;
            this.counts = counts;
            this.wanted = wanted;
        }

        /**
         * Adopts three slabs the caller already owns, deriving the wanted count from their lengths. A
         * native slab is written in place; a heap one is staged and copied back once the call is final.
         *
         * @throws USearchException SHAPE_MISMATCH when the batch holds no query, the key and score
         *     slabs are not the same shape, or the slabs do not divide into one row per query.
         */
        public static Matches of(MemorySegment keys, MemorySegment scores, MemorySegment counts) {
            long queries = counts.byteSize() / Long.BYTES;
            long slots = scores.byteSize() / Double.BYTES;
            if (queries == 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "a run must hold at least one query");
            }
            if (keys.byteSize() != slots * Key.BYTES) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the key and score slabs are not the same shape");
            }
            if (slots == 0 || slots % queries != 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the slabs do not divide into one row per query");
            }
            Extents.toIndex(queries, "the query count");
            return new Matches(keys, scores, counts, Extents.toIndex(slots / queries, "the wanted count"));
        }

        /**
         * Sizes three fresh native slabs for a batch of queries, each reporting up to wanted matches,
         * which the garbage collector frees.
         *
         * @throws USearchException SHAPE_MISMATCH when no query or no match is asked for, and
         *     OVERFLOW when the queries-by-wanted product would not fit in sixty-four bits.
         */
        public static Matches allocate(long queries, long wanted) {
            if (wanted <= 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the outputs reserve no room for a single match");
            }
            long slots = Extents.product(queries, wanted, "the match matrix");
            java.lang.foreign.Arena memory = java.lang.foreign.Arena.ofAuto();
            return of(
                    memory.allocate(Extents.product(slots, Key.BYTES, "the key matrix"), 8),
                    memory.allocate(ValueLayout.JAVA_DOUBLE, slots),
                    memory.allocate(ValueLayout.JAVA_LONG, queries));
        }

        /** The upper bound on matches per query, and the row width of both matrices. */
        public int wanted() {
            return wanted;
        }

        /** How many queries the slabs were sized for. */
        public int queries() {
            return (int) (counts.byteSize() / Long.BYTES);
        }

        /**
         * How many matches of a query the engine wrote.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch.
         */
        public int count(int query) {
            if (query < 0 || query >= queries()) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the query is outside the batch");
            }
            return (int) Math.min(counts.getAtIndex(ValueLayout.JAVA_LONG, query), wanted);
        }

        /**
         * The key of one match, in nearest-first order.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank
         *     past the matches that query reported.
         */
        public Key key(int query, int rank) {
            return Key.read(keys, slotOf(query, rank) * Key.BYTES);
        }

        /**
         * The score of one match, in the metric of the index that produced it.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank
         *     past the matches that query reported.
         */
        public double score(int query, int rank) {
            return scores.getAtIndex(ValueLayout.JAVA_DOUBLE, slotOf(query, rank));
        }

        private long slotOf(int query, int rank) {
            int readable = count(query);
            if (rank < 0 || rank >= readable) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, "the query reported " + readable + " readable matches");
            }
            return (long) query * wanted + rank;
        }

        /** Fills the {@code usearch_matches_span_t} at {@code field} for a call landing here. */
        void land(Call call, String field) {
            call.pointer(field + ".keys", call.output(keys));
            call.size(field + ".keys_stride", (long) wanted * Key.BYTES);
            call.pointer(field + ".scores", call.output(scores));
            call.size(field + ".scores_stride", (long) wanted * Double.BYTES);
            call.pointer(field + ".counts", call.output(counts));
            call.size(field + ".capacity", queries());
            call.size(field + ".wanted", wanted);
        }

        /** Renders the batch shape, not its contents. */
        @Override
        public String toString() {
            return "Matches(queries=" + queries() + ", wanted=" + wanted + ")";
        }
    }

    /**
     * A submitted call, which owns its outcome, argument structure, inputs, outputs and arena until
     * the call is final.
     *
     * <p>
     * {@link #status()} polls without blocking and {@link #join()} blocks until the call is final,
     * then answers what it produced or raises what it reported; the first join frees the outcome and
     * every later one repeats its answer. Closing an unfinished one, or dropping it, waits for the
     * call before freeing what it lent. It starts no thread: waiting off the caller's thread is the
     * caller's own, as {@code withContext(Dispatchers.IO) { call.join() }} in a Kotlin coroutine or a
     * virtual thread in Java.
     *
     * <p>
     * Binding-only; joining and closing call {@code usearch_outcome_wait}.
     *
     * @param <Result> what the call answers once final.
     */
    public static final class Deferred<Result> implements AutoCloseable {

        private final Call call;
        private final Supplier<Result> finish;
        private final Cleaner.Cleanable release;

        /** The final status, kept by the first join before it frees the outcome. */
        private volatile @Nullable Status settled;

        private volatile boolean closed;

        /** What the first join answered or raised, which every later join repeats. */
        private @Nullable Result result;

        private @Nullable Throwable failure;

        Deferred(Call call, Supplier<Result> finish) {
            this.call = call;
            this.finish = finish;
            this.release = CLEANER.register(this, call::release);
        }

        /**
         * DEFERRED until the call is final, then its final status, read without blocking.
         *
         * @throws IllegalStateException once closed.
         */
        public Status status() {
            if (closed) {
                throw new IllegalStateException("usearch: the deferred call is closed");
            }
            Status known = settled;
            if (known != null) {
                return known;
            }
            try {
                return Status.of((int) OUTCOME_STATUS.getAcquire(call.outcome, 0L));
            } catch (IllegalStateException freed) {
                // A join on another thread may have kept the status and freed the outcome in between.
                known = settled;
                if (known == null) {
                    throw freed;
                }
                return known;
            }
        }

        /**
         * Blocks until the call is final, then answers what it produced; a later join answers or
         * raises the same without waiting again.
         *
         * @throws USearchException the failure the call reported.
         * @throws IllegalStateException once closed.
         */
        public synchronized Result join() {
            if (closed) {
                throw new IllegalStateException("usearch: the deferred call is closed");
            }
            if (settled == null) {
                await(call.outcome);
                try {
                    call.raise();
                    Library.check(call.outcome);
                    result = finish.get();
                } catch (RuntimeException | Error raised) {
                    failure = raised;
                } finally {
                    settled = Status.of((int) OUTCOME_STATUS.getAcquire(call.outcome, 0L));
                    release.clean();
                }
            }
            if (failure instanceof RuntimeException raised) {
                throw raised;
            }
            if (failure instanceof Error raised) {
                throw raised;
            }
            return result;
        }

        /** Waits for the call if it is not final yet, then frees what it lent. It is idempotent. */
        @Override
        public synchronized void close() {
            closed = true;
            release.clean();
        }

        /** Renders the call as its status, without blocking. */
        @Override
        public String toString() {
            return "Deferred(" + (closed ? "closed" : status()) + ")";
        }
    }

    /**
     * Overflow-checked extent arithmetic, standing between every caller-supplied count and the
     * allocation or narrowing it drives. The C interface bounds neither the wanted count nor the row
     * count, so every product feeding an array length, a stride, or a boundary-crossing size passes
     * through here and a wrap becomes an OVERFLOW failure rather than a short slab.
     */
    static final class Extents {

        private Extents() {}

        /** Multiplies two non-negative extents, reporting OVERFLOW instead of wrapping. */
        static long product(long left, long right, String what) {
            nonNegative(left, what);
            nonNegative(right, what);
            if (left != 0 && right > Long.MAX_VALUE / left) {
                throw new USearchException(Status.OVERFLOW, what + " overflows sixty-four bits");
            }
            return left * right;
        }

        /** Adds two non-negative extents, reporting OVERFLOW instead of wrapping. */
        static long sum(long left, long right, String what) {
            nonNegative(left, what);
            nonNegative(right, what);
            if (left > Long.MAX_VALUE - right) {
                throw new USearchException(Status.OVERFLOW, what + " overflows sixty-four bits");
            }
            return left + right;
        }

        /**
         * Narrows an extent to the length of a Java array, reporting OVERFLOW when it does not fit.
         * The ceiling is a few slots below {@code Integer.MAX_VALUE}, which is where every JVM's array
         * header pushes it.
         */
        static int toArrayLength(long extent, String what) {
            nonNegative(extent, what);
            if (extent > Integer.MAX_VALUE - 8) {
                throw new USearchException(Status.OVERFLOW, what + " does not fit in a Java array");
            }
            return (int) extent;
        }

        /** Narrows an extent to an index, reporting OVERFLOW when it does not fit. */
        static int toIndex(long extent, String what) {
            nonNegative(extent, what);
            if (extent > Integer.MAX_VALUE) {
                throw new USearchException(Status.OVERFLOW, what + " does not fit in a Java index");
            }
            return (int) extent;
        }

        /** Refuses a negative extent, which no count the C interface takes may be. */
        private static void nonNegative(long extent, String what) {
            if (extent < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, what + " is negative");
            }
        }
    }

    /** One verb taking an argument structure: the structure's layout, and the downcall that takes it. */
    record Verb(StructLayout arguments, MethodHandle handle) {

        Verb(String name, StructLayout arguments) {
            this(arguments, Library.verb(name));
        }

        /** The offset of a field, nested fields named by their dotted path. */
        long at(String path) {
            return Library.offset(arguments, path.split("\\."));
        }
    }

    /**
     * A native handle freed once - on close, or once its owner is unreachable. Calls lease its
     * lifetime, so closing it while another thread's call names it raises rather than freeing memory
     * under that call.
     */
    static final class Handle {

        /** The scope calls lease; closing it is what retires the handle. */
        final java.lang.foreign.Arena lifetime = java.lang.foreign.Arena.ofShared();

        /** The address, nulled when another handle absorbs this one. */
        private final MemorySegment cell = java.lang.foreign.Arena.ofAuto().allocate(Library.POINTER);

        private final Cleaner.Cleanable release;

        Handle(Object owner, MemorySegment address, MethodHandle free) {
            cell.set(Library.POINTER, 0, address);
            java.lang.foreign.Arena scope = lifetime;
            MemorySegment slot = cell;
            release = CLEANER.register(owner, () -> {
                if (scope.scope().isAlive()) {
                    scope.close();
                }
                Library.invoke(free, slot.get(Library.POINTER, 0));
            });
        }

        /**
         * The address calls name.
         *
         * @throws IllegalStateException on a closed handle.
         */
        MemorySegment address() {
            if (!lifetime.scope().isAlive()) {
                throw new IllegalStateException("usearch: the handle is closed");
            }
            return cell.get(Library.POINTER, 0);
        }

        /**
         * Retires the handle without freeing it, answering the address for the call that absorbs it.
         *
         * @throws IllegalStateException on a closed handle, or one a running call leases.
         */
        MemorySegment surrender() {
            MemorySegment address = address();
            lifetime.close();
            cell.set(Library.POINTER, 0, MemorySegment.NULL);
            release.clean();
            return address;
        }

        /**
         * Frees the handle; idempotent.
         *
         * @throws IllegalStateException while another thread's call leases the handle.
         */
        void close() {
            if (lifetime.scope().isAlive()) {
                lifetime.close();
            }
            release.clean();
        }

        @Override
        public String toString() {
            return lifetime.scope().isAlive()
                    ? "at=" + Long.toHexString(cell.get(Library.POINTER, 0).address())
                    : "closed";
        }
    }

    /**
     * One call in flight: its argument structure, its outcome, and everything it lends the library,
     * in memory that lives until the call is final - confined to the caller when blocking, shared
     * when deferred.
     */
    static final class Call {

        final java.lang.foreign.Arena memory;
        final MemorySegment arguments;
        final MemorySegment outcome;
        private final Verb verb;

        /** Native memory the caller lends, kept reachable until the call is final. */
        private final List<Object> lent = new ArrayList<>();

        /** Copies of staged heap outputs back into the caller's slabs, run once the call succeeded. */
        private final List<Runnable> landings = new ArrayList<>();

        /** Runs once the call is final, joined or not, like freeing the donors a merge left. */
        private Runnable settle = () -> {};

        /** The first exception an upcall raised, which outranks the status it turned into. */
        final AtomicReference<@Nullable Throwable> raised = new AtomicReference<>();

        private Call(Verb verb, java.lang.foreign.Arena memory) {
            this.verb = verb;
            this.memory = memory;
            this.arguments = memory.allocate(verb.arguments());
            this.outcome = memory.allocate(Library.OUTCOME);
            arguments.set(Library.POINTER, verb.at("outcome"), outcome);
        }

        /**
         * Runs one verb blocking: fills its structure, calls, and answers what {@code request}'s
         * finisher reads once the call is final.
         *
         * @param index the handle the call leases, or {@code null} for a verb that names none.
         * @param executor the executor named in the structure, or {@code null} for the calling thread.
         */
        static <Result> Result block(
                Verb verb,
                @Nullable Handle index,
                @Nullable Executor executor,
                Function<Call, Supplier<Result>> request) {
            try (java.lang.foreign.Arena memory = java.lang.foreign.Arena.ofConfined()) {
                Call call = new Call(verb, memory);
                try {
                    Supplier<Result> finish = call.open(index, executor, request);
                    call.invoke(index);
                    call.raise();
                    Library.check(call.outcome);
                    call.landings.forEach(Runnable::run);
                    return finish.get();
                } finally {
                    call.settle.run();
                }
            }
        }

        /** Submits one verb deferred, answering a {@link Deferred} that owns the call until final. */
        static <Result> Deferred<Result> submit(
                Verb verb, Handle index, Executor executor, Function<Call, Supplier<Result>> request) {
            Call call = new Call(verb, java.lang.foreign.Arena.ofShared());
            try {
                call.enumeration("completion", 1);
                Supplier<Result> finish = call.open(index, executor, request);
                call.invoke(index);
                return new Deferred<>(call, () -> {
                    call.landings.forEach(Runnable::run);
                    return finish.get();
                });
            } catch (RuntimeException | Error refused) {
                try {
                    call.settle.run();
                } finally {
                    call.memory.close();
                }
                throw refused;
            }
        }

        private <Result> Supplier<Result> open(
                @Nullable Handle index, @Nullable Executor executor, Function<Call, Supplier<Result>> request) {
            if (index != null) {
                pointer("index", index.address());
            }
            if (executor != null) {
                pointer("executor", executor.address());
                lend(executor);
            }
            return request.apply(this);
        }

        /** Calls the verb, passing the structure in the index's lifetime so the call leases it. */
        @SuppressWarnings("restricted")
        private void invoke(@Nullable Handle index) {
            MemorySegment passed =
                    index == null ? arguments : arguments.reinterpret(arguments.byteSize(), index.lifetime, null);
            Library.invoke(verb.handle(), passed);
        }

        /** Waits until final, then settles and frees the call; what a {@link Deferred} releases. */
        void release() {
            await(outcome);
            try {
                settle.run();
            } finally {
                memory.close();
            }
        }

        /** Raises the first exception an upcall swallowed. */
        void raise() {
            Throwable first = raised.get();
            if (first instanceof RuntimeException failure) {
                throw failure;
            }
            if (first instanceof Error failure) {
                throw failure;
            }
            if (first != null) {
                throw new IllegalStateException("usearch: a callback raised", first);
            }
        }

        /** Keeps native memory reachable until the call is final. */
        void lend(Object borrowed) {
            lent.add(borrowed);
        }

        /** Runs once the call is final, whether it is joined or not. */
        void settle(Runnable action) {
            settle = action;
        }

        void size(String field, long value) {
            arguments.set(Library.SIZE, verb.at(field), value);
        }

        void enumeration(String field, int value) {
            arguments.set(Library.ENUM, verb.at(field), value);
        }

        void real(String field, float value) {
            arguments.set(ValueLayout.JAVA_FLOAT, verb.at(field), value);
        }

        void pointer(String field, MemorySegment value) {
            arguments.set(Library.POINTER, verb.at(field), value);
        }

        void key(String field, Key key) {
            key.write(arguments, verb.at(field));
        }

        /** Allocates a zeroed output of {@code layout} in the call's memory and points {@code field} at it. */
        MemorySegment output(String field, MemoryLayout layout) {
            MemorySegment allocated = memory.allocate(layout);
            pointer(field, allocated);
            return allocated;
        }

        /** Lends native bytes in place, or copies heap ones into the call's memory. */
        MemorySegment input(MemorySegment bytes) {
            if (bytes.isNative()) {
                lend(bytes);
                return bytes;
            }
            return memory.allocate(bytes.byteSize(), 8).copyFrom(bytes);
        }

        /** Lends native bytes for writing in place, or stages heap ones and copies them back once final. */
        MemorySegment output(MemorySegment bytes) {
            if (bytes.isNative()) {
                lend(bytes);
                return bytes;
            }
            MemorySegment staged = memory.allocate(bytes.byteSize(), 8).copyFrom(bytes);
            landings.add(() -> bytes.copyFrom(staged));
            return staged;
        }

        /** Fills the {@code usearch_bytes_view_t} at {@code field}. */
        void bytes(String field, MemorySegment bytes) {
            pointer(field + ".data", input(bytes));
            size(field + ".length", bytes.byteSize());
        }

        /** Fills the {@code usearch_bytes_view_t} at {@code field} with a path, never terminated. */
        void path(String field, String path) {
            bytes(field, MemorySegment.ofArray(path.getBytes(StandardCharsets.UTF_8)));
        }

        /** Fills a placement struct at {@code field}, leaving an empty run as the null C derives from. */
        void ordinals(String field, int[] ordinals) {
            if (ordinals.length != 0) {
                pointer(field + ".data", memory.allocateFrom(ValueLayout.JAVA_INT, ordinals));
                size(field + ".count", ordinals.length);
            }
        }

        /** Fills a {@code usearch_nvme_options_t} at {@code field}, leaving an empty run as the null C derives from. */
        void paths(String field, String[] paths) {
            if (paths.length == 0) {
                return;
            }
            MemorySegment views = memory.allocate(Library.BYTES_VIEW, paths.length);
            for (int order = 0; order != paths.length; ++order) {
                byte[] encoded = paths[order].getBytes(StandardCharsets.UTF_8);
                MemorySegment view = views.asSlice(order * Library.BYTES_VIEW.byteSize(), Library.BYTES_VIEW);
                view.set(Library.POINTER, 0, memory.allocateFrom(ValueLayout.JAVA_BYTE, encoded));
                view.set(Library.SIZE, Library.POINTER.byteSize(), encoded.length);
            }
            pointer(field + ".data", views);
            size(field + ".count", paths.length);
        }

        /** Fills a {@code usearch_keys_view_t} at {@code field} with one key per row. */
        void keys(String field, Key[] keys) {
            MemorySegment staged = memory.allocate((long) Key.BYTES * keys.length, 8);
            for (int row = 0; row != keys.length; ++row) {
                keys[row].write(staged, (long) row * Key.BYTES);
            }
            pointer(field + ".data", staged);
            size(field + ".stride", Key.BYTES);
        }

        /** Fills a {@code usearch_rows_view_t} at {@code field} over a matrix the call reads. */
        void rows(String field, RowsView view) {
            pointer(field + ".data", input(view.segment().asSlice(0, view.spanBytes())));
            size(field + ".stride", view.strideBytes());
            size(field + ".columns", view.columns());
            enumeration(field + ".scalar", view.scalar().code());
        }

        /** Fills a {@code usearch_rows_span_t} at {@code field} over a matrix the call writes. */
        void rowsOut(String field, RowsView view) {
            pointer(field + ".data", output(view.segment().asSlice(0, view.spanBytes())));
            size(field + ".stride", view.strideBytes());
            size(field + ".columns", view.columns());
            size(field + ".capacity", view.rows());
            enumeration(field + ".scalar", view.scalar().code());
        }

        /**
         * Points {@code field} at the slot of {@code arena}, rewinding it, and keeps it until final,
         * answering the scope this call's results are read through.
         */
        java.lang.foreign.Arena arena(String field, Arena arena) {
            java.lang.foreign.Arena scope = arena.rewind();
            pointer(field, arena.slot());
            lend(arena);
            return scope;
        }

        /** Points {@code field} at a fresh traversal-stats struct, copied out once the call succeeded. */
        void stats(String field, DenseIndex.@Nullable DenseTraversalStats stats) {
            if (stats != null) {
                MemorySegment reported = output(field, Library.TRAVERSAL_STATS);
                landings.add(() -> stats.read(reported));
            }
        }
    }

    /** Releases every handle, arena and deferred call the program dropped without closing. */
    static final Cleaner CLEANER = Cleaner.create();

    private static final MethodHandle VERSION =
            Library.downcall("usearch_version_name", FunctionDescriptor.of(Library.POINTER));

    private static final MethodHandle CAPABILITIES_COMPILED =
            Library.downcall("usearch_capabilities_compiled", FunctionDescriptor.of(Library.POINTER));

    private static final MethodHandle CAPABILITIES_AVAILABLE =
            Library.downcall("usearch_capabilities_available", FunctionDescriptor.of(Library.POINTER));

    private static final MethodHandle STATUS_NAME = Library.namer("usearch_status_name");
    private static final MethodHandle SCALAR_NAME = Library.namer("usearch_scalar_name");
    private static final MethodHandle MOOD_NAME = Library.namer("usearch_mood_name");

    private static final MethodHandle OUTCOME_WAIT =
            Library.downcall("usearch_outcome_wait", FunctionDescriptor.of(Library.ENUM, Library.POINTER));

    /** Blocks until an outcome is final. Calls {@code usearch_outcome_wait}. */
    private static void await(MemorySegment outcome) {
        try {
            int ignored = (int) OUTCOME_WAIT.invokeExact(outcome);
        } catch (Throwable failure) {
            throw Library.rethrow(failure);
        }
    }

    /** The status word of an outcome, read with acquire ordering while the library may still store it. */
    private static final VarHandle OUTCOME_STATUS = Library.ENUM.varHandle();

    /** {@code usearch_executor_for_ram_t}. */
    private static final Verb FOR_RAM = new Verb(
            "usearch_executor_for_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.PLACEMENT.withName("ram"),
                    Library.SIZE.withName("threads"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_on_pool_t}. */
    private static final Verb ON_POOL = new Verb(
            "usearch_executor_on_pool",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.PLACEMENT.withName("ram"),
                    Library.POINTER.withName("pool"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_for_vram_t}. */
    private static final Verb FOR_VRAM = new Verb(
            "usearch_executor_for_vram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.PLACEMENT.withName("vram"),
                    Library.POINTER.withName("streams"),
                    Library.SIZE.withName("scratch_limit"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_for_nvme_t}. */
    private static final Verb FOR_NVME = new Verb(
            "usearch_executor_for_nvme",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.PLACEMENT.withName("nvme"),
                    Library.SIZE.withName("threads"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_drain_t}. */
    private static final Verb DRAIN = new Verb(
            "usearch_executor_drain",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.ENUM.withName("mood"),
                    Library.padding()));

    private static final MethodHandle EXECUTOR_FREE =
            Library.downcall("usearch_executor_free", FunctionDescriptor.ofVoid(Library.POINTER));

    private static final MethodHandle ARENA_FREE =
            Library.downcall("usearch_arena_free", FunctionDescriptor.ofVoid(Library.POINTER));
}
