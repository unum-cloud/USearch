/**
 *  @file java/cloud/unum/usearch/Types.java
 *  @author Ash Vardanian
 *  @date September 8, 2026
 *  @brief The vocabulary every modality shares: identities, failures, scalar kinds, executors and deferred calls.
 */
package cloud.unum.usearch;

import java.io.Serial;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Function;
import java.util.function.Supplier;

import org.jspecify.annotations.Nullable;

/**
 * The vocabulary every modality shares: entry identities, failure reporting, scalar kinds, the
 * executor every fanned-out verb runs on, deferred calls, and the strided descriptors batches
 * travel in.
 *
 * <p>
 * The class itself is the library's introspection: its version, and the hardware it can and does
 * use. Everything else it holds is a nested type.
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
        /** Dimensionality or row counts disagree. */
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
     * the C enum reserves for kinds not yet implemented.
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
        /** A 32-bit signed integer, the coordinates of geospatial points. */
        I32(12, 4, 1),
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
        PARKED(1),
        /** Idle threads stay hot, trading power for the next call's latency. */
        SPINNING(2);

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

    /** A CPU family USearch ships tuned variants for, which a RAM or NVMe placement or executor may pin. */
    public enum CPUClass {

        /** The portable floor, with no SIMD assumptions. */
        SERIAL(1),
        /** Haswell-class x86: AVX2 and FMA. */
        INTEL_HASWELL(2),
        /** Sierra Forest-class x86: AVX-VNNI without AVX-512. */
        INTEL_SIERRA(3),
        /** Skylake-class x86: the AVX-512 foundation. */
        INTEL_SKYLAKE(4),
        /** Icelake-class x86: AVX-512 VNNI and VPOPCNT. */
        INTEL_ICELAKE(5),
        /** Genoa-class x86: native bf16 dot products over AVX-512. */
        AMD_GENOA(6),
        /** Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles. */
        INTEL_SAPPHIRE(7),
        /** Granite Rapids-class x86: the AMX-FP16 tiles. */
        INTEL_GRANITE(8),
        /** Diamond Rapids-class x86: AVX10.2's FP8 conversions. */
        INTEL_DIAMOND(9),
        /** Graviton2-class Arm: the bare NEON floor. */
        AWS_G2(10),
        /** Armv8.6 NEON with dot products and bf16. */
        APPLE_M2(11),
        /** Apple M4-class Arm: the SME outer-product tiles. */
        APPLE_M4(12),
        /** Graviton5-class Arm: Armv9.2. */
        AWS_G5(13),
        /** Vera-class Arm: FP8 dot products. */
        NVIDIA_VERA(14),
        /** Alibaba C930-class RISC-V: RVA23 vectors. */
        ALIBABA_C930(15),
        /** WebAssembly with fixed-width simd128. */
        WASM32_V128(16),
        /** WebAssembly with relaxed SIMD, threads and 64-bit addressing. */
        WASM64_EXTENDED(17),
        /** The one class of a monoversioned build. */
        MONOVERSIONED(18);

        private final int code;

        CPUClass(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Renders the class as the library spells it. Calls {@code usearch_cpu_class_name}. */
        @Override
        public String toString() {
            return Library.text(CPU_CLASS_NAME, code);
        }
    }

    /** A GPU family USearch ships tuned variants for, which a VRAM placement or executor may pin. */
    public enum GPUClass {

        /** Ampere-class NVIDIA, and every NVIDIA GPU without a class of its own. */
        NVIDIA_AMPERE(1),
        /** Ada-class NVIDIA: the fp8 conversions. */
        NVIDIA_ADA(2),
        /** Hopper-class NVIDIA. */
        NVIDIA_HOPPER(3),
        /** Blackwell-class NVIDIA. */
        NVIDIA_BLACKWELL(4),
        /** Consumer Blackwell NVIDIA. */
        NVIDIA_BLACKWELLRTX(5),
        /** Rubin-class NVIDIA. */
        NVIDIA_RUBIN(6),
        /** CDNA3 AMD: gfx942. */
        AMD_CDNA3(7),
        /** CDNA4 AMD: gfx950. */
        AMD_CDNA4(8),
        /** CDNA5 AMD: gfx1250. */
        AMD_CDNA5(9),
        /** Apple7-family GPUs, and every Apple GPU without a class of its own. */
        APPLE_METAL7(10),
        /** Apple9-family GPUs. */
        APPLE_METAL9(11),
        /** Apple10-family GPUs. */
        APPLE_METAL10(12);

        private final int code;

        GPUClass(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Renders the class as the library spells it. Calls {@code usearch_gpu_class_name}. */
        @Override
        public String toString() {
            return Library.text(GPU_CLASS_NAME, code);
        }
    }

    /**
     * Where an executor computes, and so where the arenas it makes keep their memory: host pages for
     * RAM and NVMe, device memory for VRAM.
     */
    public enum Residency {

        /** Host memory, walked by host threads. */
        RAM(1),
        /** Device memory, walked by device streams. */
        VRAM(2),
        /** NVMe drives, read through host submission rings into host memory. */
        NVME(3);

        private final int code;

        Residency(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /**
         * Names the residency a code carries.
         *
         * @throws USearchException UNKNOWN_KIND on a code this build cannot name.
         */
        static Residency of(int code) {
            for (Residency candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            throw new USearchException(Status.UNKNOWN_KIND, "the residency is unknown to this build");
        }

        /** Renders the residency as its C enumerator's stem. Calls {@code usearch_residency_name}. */
        @Override
        public String toString() {
            return Library.text(RESIDENCY_NAME, code);
        }
    }

    /** How a search reaches the indexed entries it scores. */
    public enum Traversal {

        /** Walks the graph, scoring the entries the expansion reaches - approximate, the default. */
        WALK(1),
        /** Scores every entry published when the query's scan starts - exact, on RAM and NVMe handles. */
        SCAN(2);

        private final int code;

        Traversal(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Renders the traversal as its C enumerator's stem. Calls {@code usearch_traversal_name}. */
        @Override
        public String toString() {
            return Library.text(TRAVERSAL_NAME, code);
        }
    }

    /**
     * The workers every fanned-out verb runs on, shared across indexes: an executor owns host threads
     * or device streams and their scratch, an index owns entries.
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
     * try (Executor pool = Executor.forRam(new int[] {0, 1}, null, 16, 0);
     *         DenseIndex index = DenseIndex.initRam(new DenseOptions().dimensions(256), new int[0])) {
     *     index.on(pool).insert(keys, vectors);
     *     pool.drain(Mood.PARKED);
     * }
     * }</pre>
     */
    public static final class Executor implements AutoCloseable {

        private final Handle handle;

        private Executor(MemorySegment spawned) {
            this.handle = new Handle(spawned, EXECUTOR_FREE);
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
         * @param cpuClass the class whose waiters and atomics the workers use, or {@code null} for the
         *     CPU's own.
         * @param threads Workers to spawn; zero takes 7/8 of the cores those domains carry, since full
         *     occupancy starves the ordered publication path.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         * @throws USearchException INVALID_ARGUMENT on a negative count, and OUT_OF_MEMORY when the
         *     scratch cannot be claimed.
         */
        public static Executor forRam(
                int[] memoryDomains, @Nullable CPUClass cpuClass, long threads, long deferredLimit) {
            return spawn(FOR_RAM, call -> {
                call.ordinals("ram", memoryDomains);
                call.pinned("ram", cpuClass);
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
         * @param cpuClass the class whose waiters and atomics the pool's threads use, or {@code null}
         *     for the CPU's own.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         */
        public static Executor onPool(
                MemorySegment pool, int[] memoryDomains, @Nullable CPUClass cpuClass, long deferredLimit) {
            return spawn(ON_POOL, call -> {
                call.ordinals("ram", memoryDomains);
                call.pinned("ram", cpuClass);
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
         * @param gpuClass a class the devices must run, refused with UNSUPPORTED otherwise, or
         *     {@code null} for the devices' own.
         * @param streams No streams, or one borrowed CUDA stream per device in {@code devices} order,
         *     each outliving the executor.
         * @param scratchLimit The ceiling on scratch bytes per device; zero lets scratch grow.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         * @throws USearchException SHAPE_MISMATCH when the streams are neither absent nor one per
         *     device, DEVICE when a named device does not answer, and OUT_OF_MEMORY when the scratch
         *     cannot be claimed.
         */
        public static Executor forVram(
                int[] devices,
                @Nullable GPUClass gpuClass,
                MemorySegment[] streams,
                long scratchLimit,
                long deferredLimit) {
            // The C side reads one stream per device from any non-null pointer, so only this side sees a short list.
            if (streams.length != 0 && streams.length != devices.length) {
                throw new USearchException(Status.SHAPE_MISMATCH, "pass no streams or one per device");
            }
            return spawn(FOR_VRAM, call -> {
                call.ordinals("vram", devices);
                call.pinned("vram", gpuClass);
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
         * @param cpuClass the class whose waiters and atomics the workers use, or {@code null} for the
         *     CPU's own.
         * @param threads Workers to spawn; zero derives from those devices' queue capabilities.
         * @param deferredLimit Deferred calls tracked at once; zero takes a default.
         * @throws USearchException INVALID_ARGUMENT on a negative count, and OUT_OF_MEMORY when the
         *     scratch cannot be claimed.
         */
        public static Executor forNvme(String[] paths, @Nullable CPUClass cpuClass, long threads, long deferredLimit) {
            return spawn(FOR_NVME, call -> {
                call.paths("nvme", paths);
                call.pinned("nvme", cpuClass);
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
     * Memory the library writes every output whose length scales with a call into - the matches of
     * {@link DenseVerbs#search}, the rows of {@link DenseVerbs#get}, the products of
     * {@link DenseVerbs#multiply}, the monolith of {@link DenseVerbs#saveBuffer} - reused across calls.
     *
     * <p>
     * An executor places it where it computes, decided once, and nothing is allocated until a call
     * writes into it. Every output verb requires one and answers a view over it; each call rewinds it,
     * so a steady loop allocates once, and the previous call's views raise {@link IllegalStateException}
     * rather than read stale memory. To keep one call's views alive past the next, pass the next call
     * another arena.
     *
     * <p>
     * An arena serves one call at a time and takes no lock: a second call naming it while the first
     * holds it fails with CONTENDED and leaves the first call's views readable. A blocking call holds
     * it until it returns, and a deferred one until it is joined, closed or collected. A VRAM arena
     * holds device memory, so the views over it refuse host reads with DEVICE, and it serves only
     * calls on the executor that made it, which it keeps reachable and which must be closed after it.
     * One the program drops without closing is freed once neither it nor any view over it is
     * reachable.
     *
     * <pre>{@code
     * try (Arena arena = new Arena(pool)) {
     *     for (Key key : keys) {
     *         DenseVectors vectors = index.on(pool).get(key, Scalar.F32, arena); // valid until the next call given `arena`
     *     }
     * }
     * }</pre>
     */
    public static final class Arena implements AutoCloseable {

        private final Handle handle;

        /** The scope the latest call's views read through, closed while a call holds the arena. */
        private volatile java.lang.foreign.Arena outputs = java.lang.foreign.Arena.ofShared();

        /** Host pages for calls on the calling thread. Calls {@code usearch_arena_init}. */
        public Arena() {
            this.handle = init(null);
        }

        /**
         * Memory where {@code executor} computes, serving the calls of its residency. A VRAM arena
         * grows, fills and frees on that executor's first stream, so it serves only calls on that
         * executor and must be closed before it. Calls {@code usearch_arena_init}.
         */
        public Arena(Executor executor) {
            this.handle = init(Objects.requireNonNull(executor));
        }

        private static Handle init(@Nullable Executor executor) {
            return Call.block(ARENA_INIT, null, executor, call -> {
                MemorySegment made = call.output("arena", Library.POINTER);
                return () -> {
                    MemorySegment arena = made.get(Library.POINTER, 0);
                    ArenaDescription description = described(naming -> naming.pointer("arena", arena));
                    // A GPU arena frees on its executor's stream, so its release keeps that executor alive.
                    Object owner = description.residency() == Residency.VRAM ? executor : null;
                    return new Handle(arena, ARENA_FREE, owner);
                };
            });
        }

        /**
         * Where the outputs live, as the executor that made the arena decided, so a caller reads them
         * as host memory or hands them to device code without remembering that executor.
         *
         * <p>
         * Calls {@code usearch_arena_describe}.
         *
         * @throws IllegalStateException on a closed arena.
         */
        public ArenaDescription describe() {
            return described(naming -> naming.lease("arena", handle));
        }

        private static ArenaDescription described(java.util.function.Consumer<Call> naming) {
            return Call.block(ARENA_DESCRIBE, null, null, call -> {
                naming.accept(call);
                MemorySegment description = call.output("description", ARENA_DESCRIPTION);
                return () -> ArenaDescription.lift(description);
            });
        }

        /**
         * Holds the arena for one call until {@link #release}, retiring the previous call's views.
         * Closing a scope is atomic, so the close is the claim.
         *
         * @throws USearchException CONTENDED while another call holds it, whose views stay readable.
         */
        void claim() {
            try {
                outputs.close();
            } catch (IllegalStateException held) {
                throw new USearchException(Status.CONTENDED, "the arena serves another call");
            }
        }

        /** Opens the scope the holder's views read through once it is final, letting the next call claim. */
        void release() {
            outputs = java.lang.foreign.Arena.ofShared();
        }

        /**
         * The {@code usearch_arena_t} calls name.
         *
         * @throws IllegalStateException on a closed arena.
         */
        MemorySegment address() {
            return handle.address();
        }

        /**
         * Refuses a host read of device memory.
         *
         * @throws USearchException DEVICE on a VRAM arena.
         */
        void host() {
            if (describe().residency() == Residency.VRAM) {
                throw new USearchException(Status.DEVICE, "the outputs are device memory");
            }
        }

        /**
         * Widens a pointer the released call wrote to the {@code length} bytes it addresses, valid
         * until the next call claims the arena. The view keeps this arena reachable, so the library's
         * memory outlives it; device memory is never widened, since no host read may reach it.
         *
         * @throws IllegalStateException once a later call rewound the arena, or it was closed.
         */
        @SuppressWarnings("restricted")
        MemorySegment reach(MemorySegment pointer, long length) {
            // A call joined after close() released a fresh scope over freed memory, so the handle refuses.
            handle.address();
            return length == 0 || describe().residency() == Residency.VRAM
                    ? MemorySegment.NULL
                    : pointer.reinterpret(length, outputs, ignored -> Reference.reachabilityFence(this));
        }

        /**
         * Frees the memory and invalidates every view over it, once every deferred call naming it is
         * final; joining such a call afterwards raises {@link IllegalStateException}. It is idempotent.
         *
         * <p>
         * Calls {@code usearch_arena_free}.
         */
        @Override
        public void close() {
            try {
                outputs.close();
            } catch (IllegalStateException held) {
                // Already closed, or held by a deferred call, whose release then meets the closed handle.
            }
            handle.close();
        }

        /** Renders the arena as the handle it wraps. */
        @Override
        public String toString() {
            return "Arena(" + handle + ")";
        }
    }

    /**
     * Where an arena's outputs live, as {@link Arena#describe} reports it.
     *
     * <p>
     * Mirrors {@code usearch_arena_description_t}.
     *
     * @param residency RAM for host pages, VRAM for one device's memory.
     * @param device the ordinal of the device holding the outputs, empty for host pages.
     */
    public record ArenaDescription(Residency residency, OptionalInt device) {

        /** Lifts a {@code usearch_arena_description_t}. */
        static ArenaDescription lift(MemorySegment description) {
            Residency residency =
                    Residency.of(description.get(Library.ENUM, Library.offset(ARENA_DESCRIPTION, "residency")));
            long device = description.get(Library.SIZE, Library.offset(ARENA_DESCRIPTION, "device"));
            return new ArenaDescription(
                    residency, residency == Residency.VRAM ? OptionalInt.of((int) device) : OptionalInt.empty());
        }
    }

    /**
     * The monolith a {@code saveBuffer} of any modality wrote into an arena, as
     * {@link DenseVerbs#saveBuffer} does. Valid until the arena's next call or its close.
     */
    public static final class Monolith {

        private final Arena arena;
        private final MemorySegment bytes;

        private Monolith(Arena arena, MemorySegment bytes) {
            this.arena = arena;
            this.bytes = bytes;
        }

        /** Lifts the {@code usearch_bytes_view_t} a call wrote into its arena. */
        static Monolith lift(Call call, MemorySegment view) {
            MemorySegment data = view.get(Library.POINTER, 0);
            return new Monolith(call.arena(), call.reach(data, view.get(Library.SIZE, Library.POINTER.byteSize())));
        }

        /**
         * The bytes in place, which {@link DenseIndex#loadRamBuffer} reads back with no copy.
         *
         * @throws USearchException DEVICE when the arena holds device memory.
         */
        public MemorySegment segment() {
            arena.host();
            return bytes;
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
     * A ranked run a call wrote into its arena - a queries-by-columns matrix of keys, the scores
     * positionally paired with them, and one match count per query, all read in place.
     *
     * <p>
     * {@link #key(int, int)} materializes a {@link Key} only for the slots a caller reads. Slots past
     * a query's count hold memory the engine never wrote, so every accessor here refuses them. Valid
     * until the arena's next call or its close, after which every accessor raises
     * {@link IllegalStateException}.
     */
    public static final class Matches {

        private final Arena arena;
        private final MemorySegment keys;
        private final MemorySegment scores;
        private final MemorySegment counts;
        private final int queries;
        private final int columns;

        private Matches(
                Arena arena, MemorySegment keys, MemorySegment scores, MemorySegment counts, int queries, int columns) {
            this.arena = arena;
            this.keys = keys;
            this.scores = scores;
            this.counts = counts;
            this.queries = queries;
            this.columns = columns;
        }

        /** Lifts the {@code usearch_matches_view_t} a call over {@code queries} wrote into its arena. */
        static Matches lift(Call call, MemorySegment view, long queries) {
            long columns = view.get(Library.SIZE, Library.offset(Library.MATCHES_VIEW, "columns"));
            long slots = Extents.product(queries, columns, "the match matrix");
            return new Matches(
                    call.arena(),
                    call.reach(
                            view.get(Library.POINTER, Library.offset(Library.MATCHES_VIEW, "keys")), slots * Key.BYTES),
                    call.reach(
                            view.get(Library.POINTER, Library.offset(Library.MATCHES_VIEW, "scores")),
                            slots * Double.BYTES),
                    call.reach(
                            view.get(Library.POINTER, Library.offset(Library.MATCHES_VIEW, "counts")),
                            queries * Long.BYTES),
                    Extents.toIndex(queries, "the query count"),
                    Extents.toIndex(columns, "the column count"));
        }

        /** Columns per query the call wrote, at most its wanted count, and the row width of both matrices. */
        public int columns() {
            return columns;
        }

        /** How many queries the call resolved. */
        public int queries() {
            return queries;
        }

        /**
         * How many matches of a query the engine wrote.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, and DEVICE when
         *     the arena holds device memory.
         */
        public int count(int query) {
            arena.host();
            if (query < 0 || query >= queries) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the query is outside the batch");
            }
            return (int) Math.min(counts.getAtIndex(ValueLayout.JAVA_LONG, query), columns);
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
            return (long) query * columns + rank;
        }

        /** Renders the batch shape, not its contents. */
        @Override
        public String toString() {
            return "Matches(queries=" + queries + ", columns=" + columns + ")";
        }
    }

    /**
     * A submitted call, which owns its outcome, argument structure and inputs, and borrows its handles
     * and arena, until the call is final.
     *
     * <p>
     * {@link #status()} polls without blocking and {@link #join()} blocks until the call is final,
     * then answers what it produced or raises what it reported; the first join frees the outcome and
     * every later one repeats its answer. Closing an unfinished one waits for the call before freeing
     * what it lent, and a dropped one keeps what it lent until the call is final, when the collector
     * frees it without ever waiting. It starts no thread: waiting off the caller's thread is the
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
            this.release = CLEANER.register(this, call);
        }

        /**
         * DEFERRED until the call is final, then its final status, read without blocking. Calls
         * {@code usearch_outcome_status}.
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
                return call.status();
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
                    call.settle();
                    result = finish.get();
                } catch (RuntimeException | Error raised) {
                    failure = raised;
                } finally {
                    settled = call.status();
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
            if (settled == null && !closed) {
                await(call.outcome);
            }
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

    /**
     * One verb taking an argument structure: the structure's layout, the downcall that takes it, and
     * the offset of every field, nested ones named by their dotted path, resolved once.
     */
    record Verb(StructLayout arguments, MethodHandle handle, Map<String, Long> offsets) {

        Verb(String name, StructLayout arguments) {
            this(arguments, Library.verb(name), Library.offsets(arguments));
        }

        /** The offset of a field, nested fields named by their dotted path. */
        long at(String path) {
            return offsets.get(path);
        }
    }

    /**
     * A native handle freed once - on close, or once neither its owner nor a call naming it reaches
     * it. Calls lease its lifetime, so closing it while another thread's call names it raises rather
     * than freeing memory under that call.
     */
    static final class Handle {

        /** The scope calls lease; closing it is what retires the handle. */
        final java.lang.foreign.Arena lifetime = java.lang.foreign.Arena.ofShared();

        /** The address, nulled when another handle absorbs this one. */
        private final MemorySegment cell = java.lang.foreign.Arena.ofAuto().allocate(Library.POINTER);

        private final Cleaner.Cleanable release;

        Handle(MemorySegment address, MethodHandle free) {
            this(address, free, null);
        }

        /** A handle whose release keeps {@code owner} reachable until it has freed the address. */
        Handle(MemorySegment address, MethodHandle free, @Nullable Object owner) {
            cell.set(Library.POINTER, 0, address);
            release = CLEANER.register(this, new Release(lifetime, cell, free, owner));
        }

        /** Frees a handle no call leases, and leaves one a call still leases to a later collection. */
        private record Release(
                java.lang.foreign.Arena lifetime,
                MemorySegment cell,
                MethodHandle free,
                @Nullable Object owner) implements Runnable {

            @Override
            public void run() {
                try {
                    if (lifetime.scope().isAlive()) {
                        lifetime.close();
                    }
                } catch (IllegalStateException leased) {
                    CLEANER.register(new Object(), this);
                    return;
                }
                Library.invoke(free, cell.get(Library.POINTER, 0));
                Reference.reachabilityFence(owner);
            }
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
     * when deferred. As the cleaner of a dropped {@link Deferred}, it frees nothing the library may
     * still write and never waits: an unfinished call is left to a later collection.
     */
    static final class Call implements Runnable {

        final java.lang.foreign.Arena memory;
        final MemorySegment arguments;
        final MemorySegment outcome;
        private final Verb verb;

        /** Everything the call names or borrows - handles, executor, arena, slabs - kept until final. */
        private final List<Object> lent = new ArrayList<>();

        /** Copies of fixed-size reports into the caller's objects, run once the call succeeded. */
        private final List<Runnable> reportCopies = new ArrayList<>();

        /** Runs once the call is final, joined or not, like handing back the donors an absorb left. */
        private final List<Runnable> settles = new ArrayList<>();

        /** The handle the downcall leases, so closing it from another thread raises instead of freeing it. */
        private @Nullable Handle leased;

        /** The arena the call claims as it reaches the library and releases once settled. */
        private @Nullable Arena arena;

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
                Supplier<Result> finish;
                try {
                    finish = call.open(index, executor, request);
                    call.invoke();
                    call.raise();
                    Library.check(call.outcome);
                    call.reportCopies.forEach(Runnable::run);
                } finally {
                    call.settle();
                }
                return finish.get();
            }
        }

        /** Submits one verb deferred, answering a {@link Deferred} that owns the call until final. */
        static <Result> Deferred<Result> submit(
                Verb verb, Handle index, Executor executor, Function<Call, Supplier<Result>> request) {
            Call call = new Call(verb, java.lang.foreign.Arena.ofShared());
            Supplier<Result> finish;
            try {
                call.enumeration("completion", 1);
                finish = call.open(index, executor, request);
                call.invoke();
            } catch (RuntimeException | Error refused) {
                call.release();
                throw refused;
            }
            Deferred<Result> deferred = new Deferred<>(call, () -> {
                call.reportCopies.forEach(Runnable::run);
                return finish.get();
            });
            return deferred;
        }

        private <Result> Supplier<Result> open(
                @Nullable Handle index, @Nullable Executor executor, Function<Call, Supplier<Result>> request) {
            if (index != null) {
                lease("index", index);
            }
            if (executor != null) {
                pointer("executor", executor.address());
                lend(executor);
            }
            return request.apply(this);
        }

        /**
         * Claims the arena, held until settled, and calls the verb, passing the structure in the
         * leased handle's lifetime.
         *
         * @throws USearchException CONTENDED while another call holds the arena.
         */
        @SuppressWarnings("restricted")
        private void invoke() {
            Handle handle = leased;
            MemorySegment passed =
                    handle == null ? arguments : arguments.reinterpret(arguments.byteSize(), handle.lifetime, null);
            Arena held = arena;
            if (held != null) {
                held.claim();
                settle(held::release);
            }
            Library.invoke(verb.handle(), passed);
        }

        /** The outcome's status, read without waiting. Calls {@code usearch_outcome_status}. */
        Status status() {
            try {
                return Status.of((int) OUTCOME_STATUS.invokeExact(outcome));
            } catch (Throwable failure) {
                throw Library.rethrow(failure);
            }
        }

        /**
         * What a dropped {@link Deferred} runs: frees a final call, and leaves an unfinished one to a
         * later collection.
         */
        @Override
        public void run() {
            if (status() == Status.DEFERRED) {
                CLEANER.register(new Object(), this);
                return;
            }
            release();
        }

        /** Settles and frees a call that is final or never reached the library. */
        private void release() {
            try {
                settle();
            } finally {
                memory.close();
            }
        }

        /** Runs what waits for the call to be final, then lets go of everything it named; idempotent. */
        private void settle() {
            try {
                settles.forEach(Runnable::run);
            } finally {
                settles.clear();
                LATCHED.remove(outcome.address());
                lent.clear();
            }
        }

        /** Latches the first exception an upcall raised for the call whose outcome is {@code state}. */
        static void latch(MemorySegment state, Throwable failure) {
            LATCHED.putIfAbsent(state.address(), failure);
        }

        /** Raises the first exception an upcall latched, which outranks the status it turned into. */
        void raise() {
            Throwable first = LATCHED.remove(outcome.address());
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

        /** Keeps what the call names or borrows reachable until it is final. */
        void lend(Object borrowed) {
            lent.add(borrowed);
        }

        /** Runs once the call is final, whether it is joined or not. */
        void settle(Runnable action) {
            settles.add(action);
        }

        /** Points {@code field} at a handle and leases it for the downcall, in place of the index. */
        void lease(String field, Handle handle) {
            pointer(field, handle.address());
            lend(handle);
            leased = handle;
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

        /** Pins the CPU class of the placement at {@code field}, leaving zero, which pins none, for {@code null}. */
        void pinned(String field, @Nullable CPUClass cpuClass) {
            if (cpuClass != null) {
                enumeration(field + ".cpu_class", cpuClass.code());
            }
        }

        /** Pins the GPU class of the placement at {@code field}, leaving zero, which pins none, for {@code null}. */
        void pinned(String field, @Nullable GPUClass gpuClass) {
            if (gpuClass != null) {
                enumeration(field + ".gpu_class", gpuClass.code());
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

        /** Fills a count and a pointer, as the {@code starts_count} and {@code starts} of a walk, leaving an empty run null. */
        void keyRun(String countField, String pointerField, Key[] keys) {
            if (keys.length == 0) {
                return;
            }
            MemorySegment staged = memory.allocate((long) Key.BYTES * keys.length, 8);
            for (int row = 0; row != keys.length; ++row) {
                keys[row].write(staged, (long) row * Key.BYTES);
            }
            pointer(pointerField, staged);
            size(countField, keys.length);
        }

        /** Fills a {@code usearch_rows_view_t} at {@code field} over a matrix the call reads. */
        void rows(String field, RowsView view) {
            pointer(field + ".data", input(view.segment().asSlice(0, view.spanBytes())));
            size(field + ".stride", view.strideBytes());
            size(field + ".columns", view.columns());
            enumeration(field + ".scalar", view.scalar().code());
        }

        /**
         * Points {@code field} at the arena the call claims as it reaches the library.
         *
         * @throws USearchException INVALID_ARGUMENT on a null arena, as C refuses one.
         * @throws IllegalStateException on a closed arena.
         */
        void arena(String field, Arena held) {
            if (held == null) {
                throw new USearchException(Status.INVALID_ARGUMENT, "the arena must not be null");
            }
            pointer(field, held.address());
            arena = held;
            lend(held);
        }

        /** The arena the call holds. */
        Arena arena() {
            return Objects.requireNonNull(arena);
        }

        /** Widens a pointer this call wrote into its arena to the {@code length} bytes it addresses. */
        MemorySegment reach(MemorySegment pointer, long length) {
            return arena().reach(pointer, length);
        }

        /** Points {@code field} at a fresh traversal-stats struct, copied out once the call succeeded. */
        void stats(String field, DenseIndex.@Nullable DenseTraversalStats stats) {
            if (stats != null) {
                MemorySegment reported = output(field, Library.TRAVERSAL_STATS);
                reportCopies.add(() -> stats.read(reported));
            }
        }
    }

    /** Releases every handle, arena and deferred call the program dropped without closing. */
    static final Cleaner CLEANER = Cleaner.create();

    /** The first exception an upcall raised, keyed by the outcome address of the call it ran under. */
    private static final Map<Long, Throwable> LATCHED = new ConcurrentHashMap<>();

    private static final MethodHandle VERSION =
            Library.downcall("usearch_version_name", FunctionDescriptor.of(Library.POINTER));

    private static final MethodHandle CAPABILITIES_COMPILED =
            Library.downcall("usearch_capabilities_compiled", FunctionDescriptor.of(Library.POINTER));

    private static final MethodHandle CAPABILITIES_AVAILABLE =
            Library.downcall("usearch_capabilities_available", FunctionDescriptor.of(Library.POINTER));

    private static final MethodHandle STATUS_NAME = Library.namer("usearch_status_name");
    private static final MethodHandle SCALAR_NAME = Library.namer("usearch_scalar_name");
    private static final MethodHandle MOOD_NAME = Library.namer("usearch_mood_name");
    private static final MethodHandle CPU_CLASS_NAME = Library.namer("usearch_cpu_class_name");
    private static final MethodHandle GPU_CLASS_NAME = Library.namer("usearch_gpu_class_name");
    private static final MethodHandle RESIDENCY_NAME = Library.namer("usearch_residency_name");
    private static final MethodHandle TRAVERSAL_NAME = Library.namer("usearch_traversal_name");

    private static final MethodHandle OUTCOME_WAIT =
            Library.downcall("usearch_outcome_wait", FunctionDescriptor.of(Library.ENUM, Library.POINTER));
    private static final MethodHandle OUTCOME_STATUS =
            Library.downcall("usearch_outcome_status", FunctionDescriptor.of(Library.ENUM, Library.POINTER));

    /** Blocks until an outcome is final. Calls {@code usearch_outcome_wait}. */
    private static void await(MemorySegment outcome) {
        try {
            int ignored = (int) OUTCOME_WAIT.invokeExact(outcome);
        } catch (Throwable failure) {
            throw Library.rethrow(failure);
        }
    }

    /** {@code usearch_executor_for_ram_t}. */
    private static final Verb FOR_RAM = new Verb(
            "usearch_executor_for_ram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.placement("ram"),
                    Library.SIZE.withName("threads"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_on_pool_t}. */
    private static final Verb ON_POOL = new Verb(
            "usearch_executor_on_pool",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.placement("ram"),
                    Library.POINTER.withName("pool"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_for_vram_t}. */
    private static final Verb FOR_VRAM = new Verb(
            "usearch_executor_for_vram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.placement("vram"),
                    Library.POINTER.withName("streams"),
                    Library.SIZE.withName("scratch_limit"),
                    Library.SIZE.withName("deferred_limit")));

    /** {@code usearch_executor_for_nvme_t}. */
    private static final Verb FOR_NVME = new Verb(
            "usearch_executor_for_nvme",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.placement("nvme"),
                    Library.SIZE.withName("threads"),
                    Library.SIZE.withName("deferred_limit"),
                    Library.ENUM.withName("nvme_class"),
                    Library.padding()));

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

    /** {@code usearch_arena_init_t}. */
    private static final Verb ARENA_INIT = new Verb(
            "usearch_arena_init",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("arena"),
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome")));

    private static final MethodHandle ARENA_FREE =
            Library.downcall("usearch_arena_free", FunctionDescriptor.ofVoid(Library.POINTER));

    /** {@code usearch_arena_description_t}. */
    private static final StructLayout ARENA_DESCRIPTION = MemoryLayout.structLayout(
            Library.ENUM.withName("residency"), Library.padding(), Library.SIZE.withName("device"));

    /** {@code usearch_arena_describe_t}. */
    private static final Verb ARENA_DESCRIBE = new Verb(
            "usearch_arena_describe",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("arena"),
                    Library.POINTER.withName("outcome"),
                    Library.POINTER.withName("description")));
}
