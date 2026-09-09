package cloud.unum.usearch;

import java.io.Serial;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.UUID;

/**
 * The vocabulary every modality shares - the Java twin of {@code include/usearch/types.h}: entry
 * identities, failure reporting, scalar kinds, residencies, the executor every fanned-out verb runs
 * on, and the strided descriptors batches travel in.
 *
 * <p>
 * The class itself is the library's introspection: its version, and the hardware it can and does
 * use. Everything else it holds is a nested type, so one file mirrors one header.
 */
public final class Types {

    static {
        Library.require();
    }

    private Types() {}

    /** The version of the linked library. */
    public static String version() {
        return nativeVersion();
    }

    /** The hardware capabilities the linked binary was compiled with, comma separated. */
    public static String capabilitiesCompiled() {
        return nativeCapabilitiesCompiled();
    }

    /** The hardware capabilities the current machine offers at runtime, comma separated. */
    public static String capabilitiesAvailable() {
        return nativeCapabilitiesAvailable();
    }

    /**
     * A 128-bit entry identity, most significant byte first, rendered as a lowercase dashed UUID.
     *
     * <p>
     * The sixteen bytes are a {@link UUID}'s two big-endian longs, so {@link #toUUID()} and
     * {@link #fromUUID(UUID)} convert without reordering anything.
     *
     * <p>
     * Keys need not be unique: several entries may share one, and {@link DenseIndex#count(Key)}
     * reports how many. Integer identities park in the low eight bytes, so {@code Key.of(42)} prints
     * as {@code 00000000-0000-0000-0000-00000000002a}.
     */
    public static final class Key implements Comparable<Key> {

        /** The width of a key in bytes, and the row pitch of every key matrix crossing the boundary. */
        public static final int BYTES = 16;

        /** The all-zero key. */
        public static final Key NIL = new Key(new UUID(0, 0));

        private final UUID identity;

        private Key(UUID identity) {
            this.identity = identity;
        }

        /** Parks an integer identity in the low eight bytes, zeroing the rest. */
        public static Key of(long value) {
            return new Key(new UUID(0, value));
        }

        /**
         * Adopts the identity a {@link UUID} spells, byte for byte.
         *
         * @throws USearchException MALFORMED_INPUT when the identity is null.
         */
        public static Key fromUUID(UUID identity) {
            if (identity == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the key borrows no identity");
            }
            return new Key(identity);
        }

        /**
         * Copies sixteen bytes, most significant first.
         *
         * @throws USearchException MALFORMED_INPUT when the slab is null or shorter than a key.
         */
        public static Key ofBytes(byte[] source) {
            return ofBytes(source, 0);
        }

        /**
         * Copies sixteen bytes starting at an offset, most significant first.
         *
         * @throws USearchException MALFORMED_INPUT when no key fits at that offset.
         */
        public static Key ofBytes(byte[] source, int offset) {
            if (source == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the key borrows no storage");
            }
            if (offset < 0 || source.length - BYTES < offset) {
                throw new USearchException(Status.MALFORMED_INPUT, "the key does not fit at that offset");
            }
            long high = 0;
            long low = 0;
            for (int position = 0; position < 8; ++position) {
                high = (high << 8) | (source[offset + position] & 0xffL);
                low = (low << 8) | (source[offset + 8 + position] & 0xffL);
            }
            return new Key(new UUID(high, low));
        }

        /**
         * Reads the rendering {@link #toString()} produces. Dashes may sit anywhere, or nowhere,
         * which is neither narrower nor wider than {@link UUID#fromString(String)} but different:
         * that one demands the five dashed groups and lets each be short.
         *
         * @throws USearchException MALFORMED_INPUT when the text is not thirty-two hexadecimal digits.
         */
        public static Key parse(String text) {
            if (text == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
            }
            byte[] digits = new byte[BYTES * 2];
            int written = 0;
            for (int position = 0; position < text.length(); ++position) {
                char character = text.charAt(position);
                if (character == '-') {
                    continue;
                }
                if (written == digits.length) {
                    throw new USearchException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
                }
                digits[written++] = hexadecimalDigit(character);
            }
            if (written != digits.length) {
                throw new USearchException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
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
            throw new USearchException(Status.MALFORMED_INPUT, "the key is not thirty-two hexadecimal digits");
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
            long high = identity.getMostSignificantBits();
            long low = identity.getLeastSignificantBits();
            for (int position = 0; position < 8; ++position) {
                destination[offset + position] = (byte) (high >>> (8 * (7 - position)));
                destination[offset + 8 + position] = (byte) (low >>> (8 * (7 - position)));
            }
        }

        /** Reports whether the high eight bytes are all zero, so the key names an integer identity. */
        public boolean isIntegral() {
            return identity.getMostSignificantBits() == 0;
        }

        /**
         * Reports the integer identity parked in the low eight bytes.
         *
         * @throws USearchException MALFORMED_INPUT when the high eight bytes carry anything, so no
         *     integer describes the key; {@link #isIntegral()} asks first.
         */
        public long toLong() {
            if (!isIntegral()) {
                throw new USearchException(Status.MALFORMED_INPUT, "the key carries more than an integer identity");
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

        /** Packs a run of keys into the contiguous byte matrix the boundary reads. */
        static byte[] flatten(Key[] keys) {
            byte[] slab = new byte
                    [Extents.toArrayLength(Extents.product(keys.length, BYTES, "the key matrix"), "the key matrix")];
            for (int row = 0; row < keys.length; ++row) {
                keys[row].writeInto(slab, row * BYTES);
            }
            return slab;
        }
    }

    /**
     * What went wrong, in a vocabulary shared by the engine and by this binding. Validation performed
     * here before a call crosses reports the status the engine would have reported for the same fault.
     */
    public enum Status {

        /** The call completed and carries no message. */
        SUCCESS(0, "success"),
        /** An allocation or reservation could not be satisfied. */
        OUT_OF_MEMORY(1, "out-of-memory"),
        /** Dimensionality, row counts, or output capacity disagree. */
        SHAPE_MISMATCH(2, "shape-mismatch"),
        /** A stride or packing the engine cannot traverse. */
        UNSUPPORTED_LAYOUT(3, "unsupported-layout"),
        /** A scalar, metric, or residency tag this build does not know. */
        UNKNOWN_KIND(4, "unknown-kind"),
        /** A mutating verb reached a handle that opens its checkpoint in place. */
        READ_ONLY(5, "read-only"),
        /** A file could not be opened, read, written, or parsed. */
        IO_FAILURE(6, "io"),
        /** A device could not be selected, or its memory could not be claimed. */
        DEVICE_FAILURE(7, "device"),
        /** A caller-supplied representation could not be decoded. */
        MALFORMED_INPUT(8, "malformed-input"),
        /** The library reported neither success nor a failure. */
        INTERNAL(9, "internal"),
        /** A hard design ceiling was reached, like the 32-bit identity space. */
        OVERFLOW(10, "overflow"),
        /** The verb needs a prior successful reservation that never happened. */
        UNINITIALIZED(11, "uninitialized"),
        /** The request contradicts an already-frozen configuration. */
        CONFIG_MISMATCH(12, "config-mismatch"),
        /**
         * A synchronization contract was violated by a concurrent operation; discard any produced
         * output and retry in a quieter phase.
         */
        CONTENDED(13, "contended"),
        /**
         * A progress handler asked the operation to stop; it unwound and published nothing, so the
         * handle carries whatever it did before the call.
         */
        CANCELLED(14, "cancelled"),
        /**
         * A status a newer library reported and this build cannot name. An unknown status is never
         * success.
         */
        UNRECOGNIZED(-1, "unrecognized");

        private final int code;
        private final String stem;

        Status(int code, String stem) {
            this.code = code;
            this.stem = stem;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the status a code carries; a code this build cannot name becomes UNRECOGNIZED. */
        public static Status of(int code) {
            for (Status candidate : values()) {
                if (candidate.code == code && candidate != UNRECOGNIZED) {
                    return candidate;
                }
            }
            return UNRECOGNIZED;
        }

        /** Renders the status as its C enumerator's stem. */
        @Override
        public String toString() {
            return stem;
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

        /**
         * Builds the failure a status code and its message describe - the shape the native layer
         * raises, which cannot name a Java enumerator itself.
         */
        static USearchException of(int code, String message) {
            return new USearchException(Status.of(code), message);
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
    public enum ScalarKind {

        /** A type the library could not name. */
        UNKNOWN(0, "unknown", 0, 1),
        /** 64-bit double precision. */
        F64(1, "f64", 8, 1),
        /** 32-bit single precision. */
        F32(2, "f32", 4, 1),
        /** The brain floating-point format. */
        BF16(3, "bf16", 2, 1),
        /** IEEE half precision. */
        F16(4, "f16", 2, 1),
        /** 8-bit floating point with five exponent bits. */
        E5M2(5, "e5m2", 1, 1),
        /** 8-bit floating point with four exponent bits. */
        E4M3(6, "e4m3", 1, 1),
        /** 6-bit floating point with three exponent bits, one per byte. */
        E3M2(7, "e3m2", 1, 1),
        /** 6-bit floating point with two exponent bits, one per byte. */
        E2M3(8, "e2m3", 1, 1),
        /** An 8-bit signed integer. */
        I8(10, "i8", 1, 1),
        /** Two 4-bit signed integers per byte. */
        I4X2(11, "i4x2", 1, 2),
        /** An 8-bit unsigned integer. */
        U8(20, "u8", 1, 1),
        /** Two 4-bit unsigned integers per byte. */
        U4X2(21, "u4x2", 1, 2),
        /** Eight single-bit dimensions per byte. */
        U1X8(22, "u1x8", 1, 8);

        private final int code;
        private final String stem;
        private final int valueBytes;
        private final int dimensionsPerValue;

        ScalarKind(int code, String stem, int valueBytes, int dimensionsPerValue) {
            this.code = code;
            this.stem = stem;
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
        public static ScalarKind of(int code) {
            for (ScalarKind candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return UNKNOWN;
        }

        /**
         * The bytes one row of the given width occupies.
         *
         * @throws USearchException SHAPE_MISMATCH when the width does not fill whole values of this
         *     kind, and OVERFLOW when the product does not fit in sixty-four bits.
         */
        public long rowBytes(long dimensions) {
            if (dimensions % dimensionsPerValue != 0) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, dimensions + " dimensions do not fill whole " + stem + " values");
            }
            return Extents.product(dimensions / dimensionsPerValue, valueBytes, "the row width");
        }

        /** Renders the scalar kind as its C enumerator's stem. */
        @Override
        public String toString() {
            return stem;
        }
    }

    /**
     * Where an index's payload lives and which engine traverses it. The kind decides what loading a
     * checkpoint means for the handle, and which {@link Executor} may drive it.
     *
     * <p>
     * Placement within a kind - the memory domains, the device ordinals, the device paths - is derived
     * rather than named here, so an index spreads over whatever the machine offers.
     */
    public enum Residency {

        /** A volatile-memory index traversed by CPU threads - the concurrent baseline. */
        RAM(0, "ram"),

        /** An accelerator-memory index traversed by GPU warps, sharded across the devices. */
        VRAM(1, "vram"),

        /**
         * A persisted index traversed through pluggable I/O engines, sized past RAM. External-memory
         * graphs are never modified in place, so the handle is read-only and every mutating verb fails
         * on it with READ_ONLY.
         */
        NVME(2, "nvme");

        private final int code;
        private final String stem;

        Residency(int code, String stem) {
            this.code = code;
            this.stem = stem;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /**
         * Names the kind a code carries.
         *
         * @throws USearchException UNKNOWN_KIND when no residency carries that code.
         */
        public static Residency of(int code) {
            for (Residency candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            throw new USearchException(Status.UNKNOWN_KIND, "the residency names neither RAM, VRAM, nor NVMe");
        }

        /** Renders the residency as its C enumerator's stem. */
        @Override
        public String toString() {
            return stem;
        }
    }

    /**
     * The workers every fanned-out verb runs on, shared across indexes: an executor owns host threads
     * or device streams, an index owns rows.
     *
     * <p>
     * One factory per residency, each taking the hardware its own kind places shards on and nothing
     * else: memory domains and a thread count for RAM, device paths and a thread count for NVMe, and
     * device ordinals alone for VRAM, where a single host thread drives one stream per device. Naming
     * no hardware derives it, and the kind must match the residency of every handle it drives.
     *
     * <p>
     * An executor admits one driver at a time, reporting CONTENDED rather than corrupting a dispatch
     * already in flight. Handles stay freely concurrent - it is the executor that serializes.
     *
     * <pre>{@code
     * try (Executor pool = Executor.forRam(new int[] {0, 1}, 16);
     *         DenseIndex index = new DenseIndex.DenseOptions().dimensions(256).open()) {
     *     index.add(keys, vectors, pool);
     *     pool.drain();
     * }
     * }</pre>
     */
    public static final class Executor implements AutoCloseable {

        private volatile long handle;

        /** Adopts the handle a factory spawned, refusing the one a silent failure left behind. */
        private Executor(long spawned) {
            if (spawned == 0) {
                throw new USearchException(Status.INTERNAL, "the library returned no handle and no error");
            }
            handle = spawned;
        }

        /**
         * Host workers for RAM-resident indexes, spread over every memory domain the process may
         * allocate on.
         *
         * @param threads Workers to spawn; zero takes 7/8 of the cores, since full occupancy starves
         *     the ordered publication path.
         * @throws USearchException SHAPE_MISMATCH on a negative thread count, and OUT_OF_MEMORY when
         *     the scratch arena cannot be claimed.
         */
        public static Executor forRam(long threads) {
            return new Executor(nativeExecutorForRam(null, spawnable(threads)));
        }

        /**
         * Host workers for RAM-resident indexes, interleaved across the compute domains local to the
         * named NUMA nodes, so a dual-socket box names two and gets workers local to the shards they
         * walk.
         *
         * @param memoryDomains NUMA node ids to place shards on; an empty list derives every node.
         * @param threads Workers to spawn; zero takes 7/8 of the cores those domains carry.
         * @throws USearchException MALFORMED_INPUT on a null list, SHAPE_MISMATCH on a negative
         *     thread count, and OUT_OF_MEMORY when the scratch arena cannot be claimed.
         */
        public static Executor forRam(int[] memoryDomains, long threads) {
            if (memoryDomains == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the memory domains are null");
            }
            return new Executor(nativeExecutorForRam(memoryDomains, spawnable(threads)));
        }

        /**
         * One submission stream on one free device, driven by the thread that calls the verbs.
         *
         * @throws USearchException DEVICE_FAILURE when no device answers, and OUT_OF_MEMORY when the
         *     scratch arena cannot be claimed.
         */
        public static Executor forVram() {
            return new Executor(nativeExecutorForVram(null));
        }

        /**
         * One submission stream on every named device, all of them driven by the one thread that calls
         * the verbs, which is why this takes no thread count: the work has already left for the
         * accelerator by the time a second host thread would return.
         *
         * @param devices CUDA device ordinals to place shards on; an empty list derives one free
         *     device.
         * @throws USearchException MALFORMED_INPUT on a null list, DEVICE_FAILURE when a named device
         *     does not answer, and OUT_OF_MEMORY when the scratch arena cannot be claimed.
         */
        public static Executor forVram(int[] devices) {
            if (devices == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the devices are null");
            }
            return new Executor(nativeExecutorForVram(devices));
        }

        /**
         * Host workers for NVMe-resident indexes, one submission ring apiece, leaving every shard where
         * its manifest already points.
         *
         * @param threads Workers to spawn; zero derives from the devices' queue capabilities.
         * @throws USearchException SHAPE_MISMATCH on a negative thread count, and OUT_OF_MEMORY when
         *     the scratch arena cannot be claimed.
         */
        public static Executor forNvme(long threads) {
            return new Executor(nativeExecutorForNvme(null, spawnable(threads)));
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
         * @param paths Devices or directories to place shards on; an empty list leaves every shard
         *     where its manifest points.
         * @param threads Workers to spawn; zero derives from those devices' queue capabilities.
         * @throws USearchException MALFORMED_INPUT on a null list or an empty path, SHAPE_MISMATCH on
         *     a negative thread count, and OUT_OF_MEMORY when the scratch arena cannot be claimed.
         */
        public static Executor forNvme(String[] paths, long threads) {
            if (paths == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the device paths are null");
            }
            byte[][] encoded = new byte[paths.length][];
            for (int order = 0; order != paths.length; ++order) {
                String path = paths[order];
                if (path == null || path.isEmpty()) {
                    throw new USearchException(Status.MALFORMED_INPUT, "a device path is empty");
                }
                encoded[order] = path.getBytes(StandardCharsets.UTF_8);
            }
            return new Executor(nativeExecutorForNvme(encoded, spawnable(threads)));
        }

        /**
         * Waits for every deferred call on this executor to finish.
         *
         * @throws USearchException UNINITIALIZED on a closed executor, and whatever status the first
         *     drained call reported.
         */
        public void drain() {
            nativeExecutorDrain(live());
        }

        /**
         * Drains every thread or stream and releases them. It is idempotent, and a freed executor must
         * not be used again.
         */
        @Override
        public void close() {
            long released = handle;
            handle = 0;
            if (released != 0) {
                nativeExecutorFree(released);
            }
        }

        /** Refuses a negative thread count, which no factory may spawn. */
        private static long spawnable(long threads) {
            if (threads < 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the thread count is negative");
            }
            return threads;
        }

        /** Reports the handle, refusing a closed executor. */
        private long live() {
            long current = handle;
            if (current == 0) {
                throw new USearchException(Status.UNINITIALIZED, "the executor is freed or was never initialized");
            }
            return current;
        }

        /**
         * The handle a verb hands across. The engine never derives one: a null handle reads there as a
         * residency mismatch, so the absence is named here, at the argument that is missing.
         */
        static long handleOf(Executor executor) {
            if (executor == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the executor is null");
            }
            return executor.live();
        }

        /** Renders the executor as the handle it wraps. */
        @Override
        public String toString() {
            return "Executor(at=" + Long.toHexString(handle) + ")";
        }
    }

    /**
     * A caller-owned slab viewed as rows of equal width: a base, a row count, a logical column count,
     * and a byte pitch between consecutive row starts. It is the shape every batch argument and every
     * batch output takes, so one type serves {@link DenseIndex#add} and {@link DenseIndex#get} alike.
     *
     * <p>
     * The slab is either a primitive array, borrowed for the duration of one call and never retained,
     * or a direct {@link ByteBuffer}, addressed in place with no copy. A heap buffer has no stable
     * address and is refused rather than silently copied. A matrix borrows and never owns, so the
     * slab must outlive every call it crosses.
     */
    public static final class Matrix {

        /**
         * How a slab is addressed at the boundary, closing the set of what a matrix may hold: one
         * enumerator per accessor the native layer reaches for.
         */
        enum Storage {

            /** A {@code double[]} slab. */
            DOUBLES(0),
            /** A {@code float[]} slab. */
            FLOATS(1),
            /** A {@code short[]} slab, holding the sixteen-bit kinds. */
            SHORTS(2),
            /** A {@code byte[]} slab, holding the single-byte and sub-byte kinds. */
            BYTES(3),
            /** A direct {@link ByteBuffer}, addressed in place. */
            DIRECT(4);

            private final int code;

            Storage(int code) {
                this.code = code;
            }

            /** The code the native layer switches on. */
            int code() {
                return code;
            }

            /** Whether one element of this slab is as wide as one value of the kind. */
            boolean admits(ScalarKind kind) {
                return switch (this) {
                    case DOUBLES -> kind.valueBytes() == Double.BYTES;
                    case FLOATS -> kind.valueBytes() == Float.BYTES;
                    case SHORTS -> kind.valueBytes() == Short.BYTES;
                    case BYTES -> kind.valueBytes() == Byte.BYTES;
                    case DIRECT -> true;
                };
            }
        }

        private final Object slab;
        private final Storage storage;
        private final long byteOffset;
        private final long rows;
        private final long columns;
        private final long strideBytes;
        private final ScalarKind kind;

        private Matrix(
                Object slab,
                Storage storage,
                long byteOffset,
                long rows,
                long columns,
                long strideBytes,
                ScalarKind kind) {
            this.slab = slab;
            this.storage = storage;
            this.byteOffset = byteOffset;
            this.rows = rows;
            this.columns = columns;
            this.strideBytes = strideBytes;
            this.kind = kind;
        }

        /**
         * Views a packed F64 slab as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static Matrix of(double[] data, long columns) {
            return packed(data, columns, ScalarKind.F64);
        }

        /**
         * Views a packed F32 slab as rows of the given width.
         *
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static Matrix of(float[] data, long columns) {
            return packed(data, columns, ScalarKind.F32);
        }

        /**
         * Views a packed slab of sixteen-bit values as rows of the given width.
         *
         * @throws USearchException UNSUPPORTED_LAYOUT when the kind is not two bytes wide, and
         *     SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static Matrix of(short[] data, long columns, ScalarKind kind) {
            return packed(data, columns, kind);
        }

        /**
         * Views a packed slab of single-byte values as rows of the given width.
         *
         * @throws USearchException UNSUPPORTED_LAYOUT when the kind is not one byte wide, and
         *     SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static Matrix of(byte[] data, long columns, ScalarKind kind) {
            return packed(data, columns, kind);
        }

        /**
         * Views a slab whose pitch exceeds its row width - the padded layout a memory-mapped dataset
         * arrives in. The slab is a primitive array or a direct {@link ByteBuffer}.
         *
         * @throws USearchException UNSUPPORTED_LAYOUT for a heap buffer, a slab of some other type, or
         *     a stride too narrow for one row; SHAPE_MISMATCH when the rows overrun the slab.
         */
        public static Matrix strided(Object slab, long rows, long columns, long strideBytes, ScalarKind kind) {
            Storage storage = storageOf(slab, kind);
            long rowBytes = rowBytesOf(columns, kind);
            if (strideBytes < rowBytes) {
                throw new USearchException(
                        Status.UNSUPPORTED_LAYOUT,
                        "a stride of " + strideBytes + " bytes cannot hold a row of " + rowBytes);
            }
            long span = rows == 0
                    ? 0
                    : Extents.sum(
                            Extents.product(rows - 1, strideBytes, "the matrix span"), rowBytes, "the matrix span");
            if (span > capacityOf(slab)) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, rows + " rows of stride " + strideBytes + " overrun the slab");
            }
            return new Matrix(slab, storage, offsetOf(slab), rows, columns, strideBytes, kind);
        }

        /**
         * Allocates a packed slab of the natural Java array type for the kind.
         *
         * @throws USearchException SHAPE_MISMATCH on a zero width, UNKNOWN_KIND on an unnamed kind,
         *     and OVERFLOW when the slab would not fit in a Java array.
         */
        public static Matrix allocate(long rows, long columns, ScalarKind kind) {
            long rowBytes = rowBytesOf(columns, kind);
            long values = Extents.product(rows, rowBytes / kind.valueBytes(), "the matrix");
            int length = Extents.toArrayLength(values, "the matrix");
            Object slab =
                    switch (kind.valueBytes()) {
                        case 8 -> new double[length];
                        case 4 -> new float[length];
                        case 2 -> new short[length];
                        default -> new byte[length];
                    };
            return new Matrix(slab, storageOf(slab, kind), 0, rows, columns, rowBytes, kind);
        }

        private static Matrix packed(Object data, long columns, ScalarKind kind) {
            Storage storage = storageOf(data, kind);
            long rowBytes = rowBytesOf(columns, kind);
            long capacity = capacityOf(data);
            if (capacity % rowBytes != 0) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, capacity + " bytes do not divide into rows of " + rowBytes);
            }
            return new Matrix(data, storage, offsetOf(data), capacity / rowBytes, columns, rowBytes, kind);
        }

        private static long rowBytesOf(long columns, ScalarKind kind) {
            if (columns <= 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "a matrix row cannot be zero columns wide");
            }
            if (kind == ScalarKind.UNKNOWN) {
                throw new USearchException(Status.UNKNOWN_KIND, "the matrix names no scalar kind");
            }
            return kind.rowBytes(columns);
        }

        /** Names how the slab is addressed, refusing a slab whose element width the kind rejects. */
        private static Storage storageOf(Object slab, ScalarKind kind) {
            Storage storage =
                    switch (slab) {
                        case double[] ignored -> Storage.DOUBLES;
                        case float[] ignored -> Storage.FLOATS;
                        case short[] ignored -> Storage.SHORTS;
                        case byte[] ignored -> Storage.BYTES;
                        case ByteBuffer buffer when buffer.isDirect() -> Storage.DIRECT;
                        case ByteBuffer ignored ->
                            throw new USearchException(
                                    Status.UNSUPPORTED_LAYOUT,
                                    "a heap buffer has no stable address; allocate a direct one");
                        case null, default ->
                            throw new USearchException(
                                    Status.UNSUPPORTED_LAYOUT,
                                    "a matrix borrows a primitive array or a direct ByteBuffer");
                    };
            if (!storage.admits(kind)) {
                throw new USearchException(
                        Status.UNSUPPORTED_LAYOUT, "a " + kind + " value does not fill one slab element");
            }
            return storage;
        }

        /** The addressable bytes a slab holds. */
        private static long capacityOf(Object slab) {
            return switch (slab) {
                case double[] values -> (long) values.length * Double.BYTES;
                case float[] values -> (long) values.length * Float.BYTES;
                case short[] values -> (long) values.length * Short.BYTES;
                case byte[] values -> values.length;
                case ByteBuffer buffer -> buffer.remaining();
                case null, default ->
                    throw new USearchException(
                            Status.UNSUPPORTED_LAYOUT, "a matrix borrows a primitive array or a direct ByteBuffer");
            };
        }

        private static long offsetOf(Object slab) {
            return slab instanceof ByteBuffer buffer ? buffer.position() : 0;
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
        public ScalarKind kind() {
            return kind;
        }

        /** The slab as F64 values. */
        public double[] doubles() {
            return (double[]) slab;
        }

        /** The slab as F32 values. */
        public float[] floats() {
            return (float[]) slab;
        }

        /** The slab as sixteen-bit values. */
        public short[] shorts() {
            return (short[]) slab;
        }

        /** The slab as single-byte values. */
        public byte[] bytes() {
            return (byte[]) slab;
        }

        /** The slab the view borrows, handed across as the object the native layer addresses. */
        Object slab() {
            return slab;
        }

        /** Which accessor the native layer reaches for. */
        Storage storage() {
            return storage;
        }

        /** The byte offset the first row starts at; non-zero only for a positioned buffer. */
        long byteOffset() {
            return byteOffset;
        }

        /**
         * Checks the view against a frozen geometry before it crosses the boundary. A width of zero
         * skips the geometry check, which is what an index whose geometry is still unfrozen reports.
         */
        void validate(long dimensions) {
            if (dimensions != 0 && columns != dimensions) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, "the matrix rows are not as wide as the index geometry");
            }
        }

        /** Renders the view as the geometry it describes. */
        @Override
        public String toString() {
            return "Matrix(rows=" + rows + ", columns=" + columns + ", strideBytes=" + strideBytes + ", " + kind + ")";
        }
    }

    /**
     * Where a ranked run lands - three caller-owned slabs holding a queries-by-wanted matrix of keys,
     * the scores positionally paired with them, and one match count per query.
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

        private final byte[] keys;
        private final double[] scores;
        private final long[] counts;
        private final int wanted;

        private Matches(byte[] keys, double[] scores, long[] counts, int wanted) {
            this.keys = keys;
            this.scores = scores;
            this.counts = counts;
            this.wanted = wanted;
        }

        /**
         * Adopts three slabs the caller already owns, deriving the wanted count from their lengths.
         *
         * @throws USearchException MALFORMED_INPUT when a slab is null, and SHAPE_MISMATCH when the
         *     batch holds no query, the key and score slabs are not the same shape, or the slabs do
         *     not divide into one row per query.
         */
        public static Matches of(byte[] keys, double[] scores, long[] counts) {
            if (keys == null || scores == null || counts == null) {
                throw new USearchException(Status.MALFORMED_INPUT, "the outputs borrow no storage");
            }
            if (counts.length == 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "a run must hold at least one query");
            }
            if (keys.length % Key.BYTES != 0 || keys.length / Key.BYTES != scores.length) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the key and score slabs are not the same shape");
            }
            if (scores.length == 0 || scores.length % counts.length != 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the slabs do not divide into one row per query");
            }
            return new Matches(keys, scores, counts, scores.length / counts.length);
        }

        /**
         * Sizes three fresh slabs for a batch of queries, each reporting up to wanted matches.
         *
         * @throws USearchException SHAPE_MISMATCH when no query or no match is asked for, and
         *     OVERFLOW when the queries-by-wanted product would not fit in a Java array.
         */
        public static Matches packed(long queries, long wanted) {
            if (wanted <= 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the outputs reserve no room for a single match");
            }
            long slots = Extents.product(queries, wanted, "the match matrix");
            int keyBytes = Extents.toArrayLength(Extents.product(slots, Key.BYTES, "the key matrix"), "the key matrix");
            int scoreSlots = Extents.toArrayLength(slots, "the score matrix");
            int countSlots = Extents.toIndex(queries, "the query count");
            return of(new byte[keyBytes], new double[scoreSlots], new long[countSlots]);
        }

        /** The upper bound on matches per query, and the row width of both matrices. */
        public int wanted() {
            return wanted;
        }

        /** How many queries the slabs were sized for. */
        public int queries() {
            return counts.length;
        }

        /**
         * How many matches of a query are readable - the reported count clamped to the room reserved
         * for it.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch.
         */
        public int count(int query) {
            long reported = reportedCount(query);
            return (int) Math.min(reported, wanted);
        }

        /**
         * The count the engine reported, which may exceed {@link #wanted()} when the row was truncated.
         * Comparing the two is how a caller learns to raise the wanted count and search again.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch.
         */
        public long reportedCount(int query) {
            if (query < 0 || query >= counts.length) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the query is outside the batch");
            }
            return counts[query];
        }

        /**
         * Whether a query's matches were cut short by the room reserved for them.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch.
         */
        public boolean truncated(int query) {
            return reportedCount(query) > wanted;
        }

        /**
         * The key of one match, in nearest-first order.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank
         *     past the matches that query reported.
         */
        public Key key(int query, int rank) {
            return Key.ofBytes(keys, slotOf(query, rank) * Key.BYTES);
        }

        /**
         * The score of one match, in the metric of the index that produced it.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank
         *     past the matches that query reported.
         */
        public double score(int query, int rank) {
            return scores[slotOf(query, rank)];
        }

        private int slotOf(int query, int rank) {
            int readable = count(query);
            if (rank < 0 || rank >= readable) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, "the query reported " + readable + " readable matches");
            }
            return query * wanted + rank;
        }

        /** The flat key slab the native layer writes into. */
        byte[] keySlab() {
            return keys;
        }

        /** The score slab the native layer writes into. */
        double[] scoreSlab() {
            return scores;
        }

        /** The per-query count slab the native layer writes into. */
        long[] countSlab() {
            return counts;
        }

        /** Renders the batch shape, not its contents. */
        @Override
        public String toString() {
            return "Matches(queries=" + counts.length + ", wanted=" + wanted + ")";
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

    private static native String nativeVersion();

    private static native String nativeCapabilitiesCompiled();

    private static native String nativeCapabilitiesAvailable();

    private static native long nativeExecutorForRam(int[] memoryDomains, long threads);

    private static native long nativeExecutorForVram(int[] devices);

    private static native long nativeExecutorForNvme(byte[][] paths, long threads);

    private static native void nativeExecutorDrain(long handle);

    private static native void nativeExecutorFree(long handle);
}
