/**
 *  @file java/cloud/unum/usearch/Types.java
 *  @author Ash Vardanian
 *  @date September 8, 2026
 *  @brief The vocabulary every modality shares: identities, failures, scalar kinds, executors and deferred calls.
 */
package cloud.unum.usearch;

import java.io.Serial;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.lang.ref.Cleaner;
import java.lang.ref.Reference;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.UUID;
import java.util.function.Function;
import java.util.function.Predicate;
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
     * Keys need not be unique: several entries may share one, and {@link DenseIndex#count(Key)}
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
         * @throws USearchException MALFORMED when the high eight bytes hold anything, so no
         *     integer describes the key; {@link #isIntegral()} asks first.
         */
        public long toLong() {
            if (!isIntegral()) {
                throw new USearchException(Status.MALFORMED, "the key holds more than an integer identity");
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

        /** The call succeeded and holds no message. */
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
        /** A synchronization contract was violated - a held executor, or a busy arena. */
        CONTENDED(14),
        /**
         * A progress handler asked the operation to stop; it unwound and published nothing, so the
         * handle holds whatever it did before the call.
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

        /** Names the status a code holds; one this build cannot name reads as INVALID_ARGUMENT, as in C. */
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
     * One failed call: a status every caller can branch on, and the detail the status cannot hold.
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

        /** The status this failure holds. */
        public Status status() {
            return status;
        }
    }

    /**
     * A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
     * dimensions per byte, as their {@code xN} suffixes spell.
     *
     * <p>
     * The enumerators are ordered by shrinking dynamic range, and their codes hold the numbering gaps
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

        /** How many logical dimensions one stored value holds. */
        public int dimensionsPerValue() {
            return dimensionsPerValue;
        }

        /** Names the kind a code holds; a code this build cannot name becomes UNKNOWN. */
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

        /** Idle threads sleep; the next call pays a wake-up. The default. */
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

    /** A CPU family USearch ships tuned variants for, which a RAM or NVMe placement or executor may pin. */
    public enum CPUClass {

        /** The portable floor, with no SIMD assumptions. */
        SERIAL(1),
        /** The one class of a monoversioned build. */
        MONOVERSIONED(2),
        /** Haswell-class x86: AVX2 and FMA. */
        INTEL_HASWELL(3),
        /** Sierra Forest-class x86: AVX-VNNI without AVX-512. */
        INTEL_SIERRA(4),
        /** Skylake-class x86: the AVX-512 foundation. */
        INTEL_SKYLAKE(5),
        /** Icelake-class x86: AVX-512 VNNI and VPOPCNT. */
        INTEL_ICELAKE(6),
        /** Genoa-class x86: native bf16 dot products over AVX-512. */
        AMD_GENOA(7),
        /** Sapphire Rapids-class x86: AVX-512 FP16 and the AMX tiles. */
        INTEL_SAPPHIRE(8),
        /** Granite Rapids-class x86: the AMX-FP16 tiles. */
        INTEL_GRANITE(9),
        /** Diamond Rapids-class x86: AVX10.2's FP8 conversions. */
        INTEL_DIAMOND(10),
        /** Graviton2-class Arm: the bare NEON floor. */
        AWS_G2(12),
        /** Armv8.6 NEON with dot products and bf16. */
        APPLE_M2(13),
        /** Apple M4-class Arm: the SME outer-product tiles. */
        APPLE_M4(14),
        /** Graviton5-class Arm: Armv9.2. */
        AWS_G5(15),
        /** Vera-class Arm: FP8 dot products. */
        NVIDIA_VERA(16),
        /** Alibaba C930-class RISC-V: RVA23 vectors. */
        ALIBABA_C930(17),
        /** IBM POWER10-class Power: VSX vectors and MMA accumulators. */
        IBM_POWER10(18),
        /** WebAssembly with fixed-width simd128. */
        WASM32_V128(19),
        /** WebAssembly with relaxed SIMD, threads and 64-bit addressing. */
        WASM64_EXTENDED(20);

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
         * Names the residency a code holds.
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

    /** The GPU vendor whose runtime drives a device, as {@code usearch_gpu_vendor_t} numbers it. */
    public enum GpuVendor {

        /** No vendor, as host pages report. */
        UNKNOWN(0),
        /** NVIDIA, through the CUDA driver. */
        NVIDIA(1),
        /** AMD, through the HIP runtime. */
        AMD(2),
        /** Apple, through Metal. */
        APPLE(3);

        private final int code;

        GpuVendor(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the vendor a code holds, UNKNOWN for host pages or a code this build cannot name. */
        static GpuVendor of(int code) {
            for (GpuVendor candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            return UNKNOWN;
        }
    }

    /** Whether host code may read an arena's pages, as {@code usearch_host_access_t} numbers it. */
    public enum HostAccess {

        /** Device memory the host must not read. */
        REFUSED(0),
        /** Host, pinned, managed or Apple's shared memory. */
        READABLE(1);

        private final int code;

        HostAccess(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the access a code holds; a code this build cannot name is refused, never read. */
        static HostAccess of(int code) {
            return code == READABLE.code ? READABLE : REFUSED;
        }

        /** Renders the access as its C enumerator's stem. Calls {@code usearch_host_access_name}. */
        @Override
        public String toString() {
            return Library.text(HOST_ACCESS_NAME, code);
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
     * and a thread count for RAM, device ordinals and optional streams for VRAM, where a single host
     * thread drives one stream per device, and device paths and a thread count for NVMe. Naming no
     * hardware derives it, and the kind must match the residency of every handle it drives.
     *
     * <p>
     * A receiver names the executor its verbs run on, and the index alone runs them on the caller's
     * thread. One call holds an executor at a time, and a second finding it held finishes
     * CONTENDED, as nothing queues; handles stay freely concurrent. A deferred call on a
     * caller-exclusive pool runs on the pool's thread 0, so polling alone finishes it, while a single
     * thread or a VRAM executor runs it at submission.
     *
     * <pre>{@code
     * try (Executor pool = Executor.forRam(new int[] {0, 1}, null, 16);
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
         * @param cpuClassCeiling the newest class dispatch may pick, intersected with what the
         *     machine runs, so a class the machine lacks narrows rather than refuses; {@code null}
         *     means no ceiling.
         * @param threads Workers to spawn; zero takes 7/8 of the cores those domains hold, since full
         *     occupancy starves the ordered publication path.
         * @throws USearchException OUT_OF_MEMORY when the scratch cannot be claimed.
         */
        public static Executor forRam(int[] memoryDomains, @Nullable CPUClass cpuClassCeiling, long threads) {
            return spawn(FOR_RAM, call -> {
                call.ram(call.arguments, FOR_RAM_RAM, memoryDomains, cpuClassCeiling);
                call.size(FOR_RAM_THREADS, threads);
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
         * @param gpuClassCeiling the newest class dispatch may pick, capped below the best class
         *     every named device loads; a ceiling no named device loads is refused as UNSUPPORTED;
         *     {@code null} means no ceiling.
         * @param streams No streams, which uses the one non-blocking stream per device the executor
         *     opens and destroys with itself; one CUDA stream per device in {@code devices} order is
         *     declared for the caller's own streams, which answer UNSUPPORTED, since the executor
         *     drives only its own.
         * @param scratchLimit The ceiling on scratch bytes per device; zero lets scratch grow.
         * @throws USearchException DEVICE when a named device does not answer, and OUT_OF_MEMORY when
         *     the scratch cannot be claimed.
         */
        public static Executor forVram(
                int[] devices, @Nullable GPUClass gpuClassCeiling, MemorySegment[] streams, long scratchLimit) {
            return spawn(FOR_VRAM, call -> {
                call.vram(call.arguments, FOR_VRAM_VRAM, devices, gpuClassCeiling);
                if (streams.length != 0 && streams.length != devices.length)
                    throw new USearchException(Status.SHAPE_MISMATCH, "pass no streams or one per device");
                if (streams.length != 0) {
                    call.pointer(FOR_VRAM_STREAMS, call.addresses(streams));
                }
                call.size(FOR_VRAM_SCRATCH_LIMIT, scratchLimit);
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
         * @param cpuClassCeiling the newest class dispatch may pick, intersected with what the
         *     machine runs, so a class the machine lacks narrows rather than refuses; {@code null}
         *     means no ceiling.
         * @param threads Workers to spawn; zero derives from those devices' queue capabilities.
         * @throws USearchException OUT_OF_MEMORY when the scratch cannot be claimed.
         */
        public static Executor forNvme(String[] paths, @Nullable CPUClass cpuClassCeiling, long threads) {
            return spawn(FOR_NVME, call -> {
                call.nvme(call.arguments, FOR_NVME_NVME, paths, cpuClassCeiling);
                call.size(FOR_NVME_THREADS, threads);
            });
        }

        /**
         * Waits until no call holds the executor, then sets what its threads do until the next call;
         * the last drain wins. Every call reports its own failure, so this reports only what befell the
         * executor itself.
         *
         * <p>
         * Calls {@code usearch_executor_drain}.
         *
         * @throws USearchException DEVICE on a lost device.
         * @throws IllegalStateException on a closed executor.
         */
        public void drain(Mood mood) {
            Call.block(DRAIN, null, this, call -> {
                call.enumeration(DRAIN_MOOD, mood.code());
                return () -> null;
            });
        }

        /**
         * Drains every thread or stream and releases them. It is idempotent, and a freed executor must
         * not be used again. One the program drops without closing is released once it is unreachable.
         *
         * @throws USearchException CONTENDED while an arena made with this executor lives; close it first.
         *
         * <p>
         * Calls {@code usearch_executor_free}.
         */
        @Override
        public void close() {
            handle.close();
        }

        /** Runs one executor factory and adopts what it spawned. */
        private static Executor spawn(Verb verb, java.util.function.Consumer<Call> fill) {
            return Call.block(verb, null, null, call -> {
                MemorySegment spawned = call.output(verb.executor(), Library.POINTER);
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
     * {@link DenseIndex#search}, the rows of {@link DenseIndex#get}, the products of
     * {@link DenseIndex#multiply}, the monolith of {@link DenseIndex#saveBuffer} - reused across calls.
     *
     * <p>
     * An executor places it where it computes, decided once, and nothing is allocated until a call
     * writes into it. Every output verb requires one and answers a view over it; each call rewinds it,
     * so a steady loop allocates once. A view ends at the arena's next call, and reading it afterwards
     * reads whatever that call left; to keep one call's views past the next, pass the next call
     * another arena. Where the outputs live is what {@link #describe()} reports, and a view of device
     * memory is the caller's to hand to device code rather than read from the host.
     *
     * <p>
     * An arena serves one call at a time and takes no lock: a second call naming it while the first
     * holds it fails with CONTENDED. A VRAM arena serves only calls on the executor that made it,
     * which it keeps reachable and which must be closed after it. One the program drops without
     * closing is freed once neither it nor any view over it is reachable.
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

        /** The scope every view reads through, closed with the arena, keeping it reachable while a view is. */
        private final java.lang.foreign.Arena views = viewing();

        /** Host pages for calls on the calling thread. Calls {@code usearch_arena_init}. */
        public Arena() {
            this.handle = init(null);
        }

        /**
         * Memory where {@code executor} computes, serving the calls of its residency. A VRAM arena
         * grows, fills and frees on that executor's first stream, so it serves only calls on that
         * executor and must be closed before it. Every result is a host view, so memory the host may
         * not read is refused here, before any call runs, while Apple's shared memory passes. Calls
         * {@code usearch_arena_init}.
         *
         * @throws USearchException DEVICE when the executor's memory is memory the host may not read.
         */
        public Arena(Executor executor) {
            this.handle = init(Objects.requireNonNull(executor));
            try {
                describe().host();
            } catch (USearchException refusal) {
                close();
                throw refusal;
            }
        }

        private static Handle init(@Nullable Executor executor) {
            return Call.block(ARENA_INIT, null, executor, call -> {
                MemorySegment made = call.output(ARENA_INIT_ARENA, Library.POINTER);
                // A GPU arena frees on its executor's stream, so its release keeps that executor alive.
                return () -> new Handle(made.get(Library.POINTER, 0), ARENA_FREE, executor);
            });
        }

        @SuppressWarnings("restricted")
        private java.lang.foreign.Arena viewing() {
            java.lang.foreign.Arena scope = java.lang.foreign.Arena.ofShared();
            MemorySegment.NULL.reinterpret(0, scope, ignored -> Reference.reachabilityFence(this));
            return scope;
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
            return Call.block(ARENA_DESCRIBE, null, null, call -> {
                call.reference(ARENA_DESCRIBE_ARENA, handle);
                MemorySegment description = call.output(ARENA_DESCRIBE_DESCRIPTION, ARENA_DESCRIPTION);
                return () -> ArenaDescription.lift(description);
            });
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
         * Widens a pointer a call wrote to the {@code length} bytes it addresses, readable until the
         * arena closes, and keeping the arena reachable meanwhile; the constructor already refused
         * memory the host may not read.
         *
         * @throws IllegalStateException once the arena was closed.
         */
        @SuppressWarnings("restricted")
        MemorySegment reach(MemorySegment pointer, long length) {
            return length == 0 ? MemorySegment.NULL : pointer.reinterpret(length, views, null);
        }

        /** The engine's generation of the arena, which every call that takes it advances once. */
        long generation() {
            return Library.generation(ARENA_GENERATION, handle.address());
        }

        /**
         * Runs {@code copy} under an open export of the arena, so no call takes it meanwhile.
         *
         * @throws USearchException CONTENDED when a later call took the arena since {@code generation}
         *     wrote the result, or a call holds it now.
         */
        <T> T read(long generation, java.util.function.Supplier<T> copy) {
            MemorySegment address = handle.address();
            if (Library.release(ARENA_EXPORT_OPEN, address) != 0) {
                throw new USearchException(Status.CONTENDED, "a call holds the arena");
            }
            try {
                if (Library.generation(ARENA_GENERATION, address) != generation) {
                    throw new USearchException(
                            Status.CONTENDED, "a later call took the arena since this result was written");
                }
                return copy.get();
            } finally {
                Library.release(ARENA_EXPORT_CLOSE, address);
            }
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
            if (views.scope().isAlive()) {
                views.close();
            }
            handle.close();
        }

        /** Renders the arena as the handle it wraps. */
        @Override
        public String toString() {
            return "Arena(" + handle + ")";
        }
    }

    /** The arena a result reads, and the generation the call that wrote it brought the arena to. */
    record Stamp(Arena arena, long generation) {

        /** Runs {@code copy}, which reads the result, under an open export; see {@link Arena#read}. */
        <T> T read(java.util.function.Supplier<T> copy) {
            return arena.read(generation, copy);
        }
    }

    /**
     * Where an arena's outputs live, as {@link Arena#describe} reports it.
     *
     * <p>
     * Mirrors {@code usearch_arena_description_t}.
     *
     * @param residency RAM for host pages, VRAM for one device's memory.
     * @param device the device's position in the GPU listing, empty for host pages.
     * @param vendor the vendor whose memory it is, UNKNOWN for host pages.
     * @param vendorOrdinal the vendor's own device number, which Arrow and DLPack name, empty for host pages.
     * @param hostAccess whether host code may read the pages; every host view of the arena checks it.
     */
    public record ArenaDescription(
            Residency residency,
            OptionalInt device,
            GpuVendor vendor,
            OptionalInt vendorOrdinal,
            HostAccess hostAccess) {

        /**
         * Lets host code read the outputs, or refuses.
         *
         * @throws USearchException DEVICE when the outputs are memory the host may not read.
         */
        void host() {
            if (hostAccess != HostAccess.READABLE) {
                throw new USearchException(Status.DEVICE, "the outputs are memory the host may not read");
            }
        }

        /** Lifts a {@code usearch_arena_description_t}. */
        static ArenaDescription lift(MemorySegment description) {
            return new ArenaDescription(
                    Residency.of((int) DESCRIPTION_RESIDENCY.get(description, 0L)),
                    ordinal((int) DESCRIPTION_DEVICE.get(description, 0L)),
                    GpuVendor.of((int) DESCRIPTION_VENDOR.get(description, 0L)),
                    ordinal((int) DESCRIPTION_VENDOR_ORDINAL.get(description, 0L)),
                    HostAccess.of((int) DESCRIPTION_HOST_ACCESS.get(description, 0L)));
        }

        /** Reads a device number the C side spells -1 for host pages. */
        private static OptionalInt ordinal(int number) {
            return number < 0 ? OptionalInt.empty() : OptionalInt.of(number);
        }
    }

    /**
     * The monolith a {@code saveBuffer} of any modality wrote into an arena, as
     * {@link DenseIndex#saveBuffer} does. Valid until the arena's next call or its close.
     */
    public static final class Monolith {

        private final MemorySegment bytes;
        private final Stamp stamp;

        private Monolith(MemorySegment bytes, Stamp stamp) {
            this.bytes = bytes;
            this.stamp = stamp;
        }

        /** Lifts the {@code usearch_bytes_view_t} a call wrote into its arena. */
        static Monolith lift(Call call, MemorySegment view) {
            return new Monolith(
                    call.reach(
                            (MemorySegment) Library.BYTES_DATA.get(view, 0L),
                            (long) Library.BYTES_LENGTH.get(view, 0L)),
                    call.stamp());
        }

        /**
         * The bytes in place, which {@link DenseIndex#loadRamBuffer} reads back with no copy; valid until
         * the arena's next call.
         *
         * @throws USearchException CONTENDED once a later call took the arena.
         */
        public MemorySegment segment() {
            return stamp.read(() -> bytes);
        }
    }

    /**
     * A slab viewed as rows of equal width: a base, a row count, a logical column count, and a byte
     * pitch between consecutive row starts. It is the shape every batch argument takes, from
     * {@link DenseIndex#insert} to {@link DenseIndex#attend}.
     *
     * <p>
     * The slab is a {@link MemorySegment}: a heap one, as the array factories wrap, is copied into
     * native memory for the duration of one call and never retained, and a native one - a mapped
     * file, a direct buffer through {@link MemorySegment#ofBuffer}, an arena allocation - is
     * addressed in place with no copy. A view only points at memory it never owns, so a native slab must stay
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
         * @throws USearchException SHAPE_MISMATCH when the slab does not divide into whole rows.
         */
        public static RowsView of(MemorySegment data, long columns, Scalar scalar) {
            long rowBytes = scalar.rowBytes(columns);
            if (rowBytes == 0 || data.byteSize() % rowBytes != 0) {
                throw new USearchException(
                        Status.SHAPE_MISMATCH, data.byteSize() + " bytes do not divide into rows of " + rowBytes);
            }
            return new RowsView(data, data.byteSize() / rowBytes, columns, rowBytes, scalar);
        }

        /**
         * Views a slab whose pitch exceeds its row width - the padded layout a memory-mapped dataset
         * arrives in.
         *
         * @throws USearchException SHAPE_MISMATCH when the rows overrun the slab.
         */
        public static RowsView strided(MemorySegment slab, long rows, long columns, long strideBytes, Scalar scalar) {
            long rowBytes = scalar.rowBytes(columns);
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

        /** The slab the view points into. */
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
     * until the arena's next call; once the arena is closed, every accessor raises
     * {@link IllegalStateException}.
     */
    public static final class Matches {

        private final MemorySegment keys;
        private final MemorySegment scores;
        private final MemorySegment counts;
        private final int queries;
        private final int columns;
        private final Stamp stamp;

        private Matches(
                MemorySegment keys, MemorySegment scores, MemorySegment counts, int queries, int columns, Stamp stamp) {
            this.keys = keys;
            this.scores = scores;
            this.counts = counts;
            this.queries = queries;
            this.columns = columns;
            this.stamp = stamp;
        }

        /** Lifts the {@code usearch_matches_view_t} a call over {@code queries} wrote into its arena. */
        static Matches lift(Call call, MemorySegment view, long queries) {
            long columns = (long) Library.MATCHES_COLUMNS.get(view, 0L);
            long slots = Extents.product(queries, columns, "the match matrix");
            return new Matches(
                    call.reach((MemorySegment) Library.MATCHES_KEYS.get(view, 0L), slots * Key.BYTES),
                    call.reach((MemorySegment) Library.MATCHES_SCORES.get(view, 0L), slots * Double.BYTES),
                    call.reach((MemorySegment) Library.MATCHES_COUNTS.get(view, 0L), queries * Long.BYTES),
                    Extents.toIndex(queries, "the query count"),
                    Extents.toIndex(columns, "the column count"),
                    call.stamp());
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
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch.
         */
        public int count(int query) {
            if (query < 0 || query >= queries) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the query is outside the batch");
            }
            return stamp.read(() -> (int) Math.min(counts.getAtIndex(ValueLayout.JAVA_LONG, query), columns));
        }

        /**
         * The key of one match, in nearest-first order.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank
         *     past the matches that query reported.
         */
        public Key key(int query, int rank) {
            return stamp.read(() -> Key.read(keys, slotOf(query, rank) * Key.BYTES));
        }

        /**
         * The score of one match, in the metric of the index that produced it.
         *
         * @throws USearchException SHAPE_MISMATCH when the query is outside the batch, or the rank
         *     past the matches that query reported.
         */
        public double score(int query, int rank) {
            return stamp.read(() -> scores.getAtIndex(ValueLayout.JAVA_DOUBLE, slotOf(query, rank)));
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
     * A submitted call, which owns its outcome, argument structure and inputs, and uses its handles
     * and arena, until the call is final.
     *
     * <p>
     * {@link #status()} polls without blocking and {@link #join()} blocks until the call is final,
     * then answers what it produced or raises what it reported; the first join frees the outcome and
     * every later one repeats its answer. Closing an unfinished one waits for the call before freeing
     * what it passed, and a dropped one keeps what it passed until the call is final, when the collector
     * frees it without ever waiting. It starts no thread: waiting off the caller's thread is the
     * caller's own, as {@code withContext(Dispatchers.IO) { call.join() }} in a Kotlin coroutine or a
     * virtual thread in Java.
     *
     * <p>
     * One thread owns a deferred call at a time: it takes no lock, so polling, joining and closing it
     * from two threads at once is a race. Handing it to another thread is safe across any
     * happens-before edge, such as a queue or a future.
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
        private @Nullable Status settled;

        private boolean closed;

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
            return settled != null ? settled : call.status();
        }

        /**
         * Blocks until the call is final, then answers what it produced; a later join answers or
         * raises the same without waiting again.
         *
         * @throws USearchException the failure the call reported.
         * @throws IllegalStateException once closed.
         */
        public Result join() {
            if (closed) {
                throw new IllegalStateException("usearch: the deferred call is closed");
            }
            if (settled == null) {
                await(call.outcome);
                try {
                    Library.check(call.outcome);
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

        /** Waits for the call if it is not final yet, then frees what it passed. It is idempotent. */
        @Override
        public void close() {
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
     * the handles of the fields every call fills, each {@code null} where the structure has none.
     */
    record Verb(
            StructLayout arguments,
            MethodHandle handle,
            @Nullable VarHandle index,
            @Nullable VarHandle executor,
            VarHandle outcome,
            @Nullable VarHandle completion) {

        Verb(String name, StructLayout arguments) {
            this(
                    arguments,
                    Library.verb(name),
                    optional(arguments, "index"),
                    optional(arguments, "executor"),
                    Library.field(arguments, "outcome"),
                    optional(arguments, "completion"));
        }

        /** The handle of one possibly nested field. */
        VarHandle field(String... path) {
            return Library.field(arguments, path);
        }

        /** The offset of one embedded structure, which its own layout's handles address from. */
        long at(String name) {
            return Library.offset(arguments, name);
        }

        private static @Nullable VarHandle optional(StructLayout layout, String name) {
            for (MemoryLayout member : layout.memberLayouts()) {
                if (member.name().filter(name::equals).isPresent()) {
                    return Library.field(layout, name);
                }
            }
            return null;
        }
    }

    /**
     * A native handle freed once - on close, or once neither its owner nor a call naming it reaches
     * it. Calls reference its lifetime, so closing it while another thread's call names it raises rather
     * than freeing memory under that call.
     */
    static final class Handle {

        private final Cleaner.Cleanable cleanable;

        private final Release release;

        Handle(MemorySegment address, MethodHandle free) {
            this(address, free, null);
        }

        /** A handle whose release keeps {@code owner} reachable until it has freed the address. */
        Handle(MemorySegment address, MethodHandle free, @Nullable Object owner) {
            release = new Release(address, free, owner);
            cleanable = CLEANER.register(this, release);
        }

        /**
         * Frees a handle no call references, and leaves one a call still references to a later collection.
         * It owns the state the cleaner reads, so an explicit close can retire it first.
         */
        private static final class Release implements Runnable {

            /** The scope calls reference; closing it is what retires the handle. */
            volatile java.lang.foreign.Arena lifetime = java.lang.foreign.Arena.ofShared();

            private final MemorySegment address;
            private final MethodHandle free;
            private final @Nullable Object owner;
            private volatile boolean freed;

            Release(MemorySegment address, MethodHandle free, @Nullable Object owner) {
                this.address = address;
                this.free = free;
                this.owner = owner;
            }

            /** Frees now, or throws and leaves the handle open when the library refuses. */
            void close() {
                if (freed) {
                    return;
                }
                if (lifetime.scope().isAlive()) {
                    lifetime.close();
                }
                int status = Library.release(free, address);
                if (status != 0) {
                    lifetime = java.lang.foreign.Arena.ofShared();
                    throw new USearchException(Status.of(status), "an arena made with the handle is alive");
                }
                freed = true;
                Reference.reachabilityFence(owner);
            }

            @Override
            public void run() {
                if (freed) {
                    return;
                }
                try {
                    if (lifetime.scope().isAlive()) {
                        lifetime.close();
                    }
                } catch (IllegalStateException referenced) {
                    CLEANER.register(new Object(), this);
                    return;
                }
                freed = true;
                Library.release(free, address);
                Reference.reachabilityFence(owner);
            }
        }

        /** The scope calls reference. */
        java.lang.foreign.Arena lifetime() {
            return release.lifetime;
        }

        /**
         * The address calls name.
         *
         * @throws IllegalStateException on a closed handle.
         */
        MemorySegment address() {
            if (!release.lifetime.scope().isAlive()) {
                throw new IllegalStateException("usearch: the handle is closed");
            }
            return release.address;
        }

        /**
         * Frees the handle; idempotent. Concurrent calls to it are undefined, as for the C free.
         *
         * @throws IllegalStateException while another thread's call references the handle.
         * @throws USearchException CONTENDED when the library refuses the free, which leaves the handle open.
         */
        void close() {
            release.close();
            cleanable.clean();
        }

        @Override
        public String toString() {
            return release.lifetime.scope().isAlive() ? "at=" + Long.toHexString(release.address.address()) : "closed";
        }
    }

    /**
     * One call: its argument structure, its outcome, and everything it passes the library,
     * in memory that lives until the call is final - confined to the caller when blocking, shared
     * when deferred. As the cleaner of a dropped {@link Deferred}, it frees nothing the library may
     * still write and never waits: an unfinished call is left to a later collection.
     */
    static final class Call implements Runnable {

        final java.lang.foreign.Arena memory;
        final MemorySegment arguments;
        final MemorySegment outcome;
        private final Verb verb;

        /** Everything the call names or views - handles, executor, arena, slabs - kept until final. */
        private final List<Object> passed = new ArrayList<>();

        /** Copies of fixed-size reports into the caller's objects, run once the call succeeded. */
        private final List<Runnable> reportCopies = new ArrayList<>();

        /** The handle the downcall references, so closing it from another thread raises instead of freeing it. */
        @Nullable
        Handle referenced;

        /** The executor named in the structure, or {@code null} for the calling thread. */
        @Nullable
        Executor executor;

        private @Nullable Arena arena;

        /** The generation the call brings its arena to, which its results read current at. */
        private long generation;

        private Call(Verb verb, java.lang.foreign.Arena memory) {
            this.verb = verb;
            this.memory = memory;
            this.arguments = memory.allocate(verb.arguments());
            this.outcome = memory.allocate(Library.OUTCOME);
            verb.outcome().set(arguments, 0L, outcome);
        }

        /**
         * Runs one verb blocking: fills its structure, calls, and answers what {@code request}'s
         * finisher reads once the call is final.
         *
         * @param index the handle the call references, or {@code null} for a verb that names none.
         * @param executor the executor named in the structure, or {@code null} for the calling thread.
         */
        static <Result> Result block(
                Verb verb,
                @Nullable Handle index,
                @Nullable Executor executor,
                Function<Call, Supplier<Result>> request) {
            try (java.lang.foreign.Arena memory = java.lang.foreign.Arena.ofConfined()) {
                Call call = new Call(verb, memory);
                Supplier<Result> finish = call.open(index, executor, request);
                call.invoke();
                Library.check(call.outcome);
                call.reportCopies.forEach(Runnable::run);
                return finish.get();
            }
        }

        /** Submits one verb deferred, answering a {@link Deferred} that owns the call until final. */
        static <Result> Deferred<Result> submit(
                Verb verb, @Nullable Handle index, Executor executor, Function<Call, Supplier<Result>> request) {
            Call call = new Call(verb, java.lang.foreign.Arena.ofShared());
            Supplier<Result> finish;
            try {
                call.enumeration(Objects.requireNonNull(verb.completion()), 1);
                finish = call.open(index, executor, request);
                call.invoke();
            } catch (RuntimeException | Error refused) {
                call.release();
                throw refused;
            }
            return new Deferred<>(call, () -> {
                call.reportCopies.forEach(Runnable::run);
                return finish.get();
            });
        }

        private <Result> Supplier<Result> open(
                @Nullable Handle index, @Nullable Executor executor, Function<Call, Supplier<Result>> request) {
            VarHandle indexField = verb.index();
            if (index != null && indexField != null) {
                reference(indexField, index);
            }
            VarHandle executorField = verb.executor();
            if (executor != null && executorField != null) {
                pointer(executorField, executor.address());
                pass(executor);
                this.executor = executor;
            }
            return request.apply(this);
        }

        /** Calls the verb, passing the structure in the referenced handle's lifetime. */
        @SuppressWarnings("restricted")
        private void invoke() {
            Handle handle = referenced;
            MemorySegment passed =
                    handle == null ? arguments : arguments.reinterpret(arguments.byteSize(), handle.lifetime(), null);
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

        /** Lets go of everything the call named and frees its memory, once it is final or never reached the library. */
        private void release() {
            passed.clear();
            memory.close();
        }

        /** Keeps what the call names or views reachable until it is final. */
        void pass(Object viewed) {
            passed.add(viewed);
        }

        /** Points {@code field} at a handle and references it for the downcall. */
        void reference(VarHandle field, Handle handle) {
            pointer(field, handle.address());
            pass(handle);
            referenced = handle;
        }

        void size(VarHandle field, long value) {
            field.set(arguments, 0L, count(value));
        }

        /** Refuses a negative count, which no unsigned Java type stops and C would read as a huge size. */
        static long count(long value) {
            if (value < 0) throw new USearchException(Status.SHAPE_MISMATCH, "a count is negative");
            return value;
        }

        /** Writes a seed's 64 bits as they stand, since a seed is no count and its top bit no sign. */
        void seed(VarHandle field, long value) {
            field.set(arguments, 0L, value);
        }

        void enumeration(VarHandle field, int value) {
            field.set(arguments, 0L, value);
        }

        void real(VarHandle field, float value) {
            field.set(arguments, 0L, value);
        }

        void pointer(VarHandle field, MemorySegment value) {
            field.set(arguments, 0L, value);
        }

        void key(long at, Key key) {
            key.write(arguments, at);
        }

        /** Allocates a zeroed output of {@code layout} in the call's memory and points {@code field} at it. */
        MemorySegment output(VarHandle field, MemoryLayout layout) {
            MemorySegment allocated = memory.allocate(layout);
            pointer(field, allocated);
            return allocated;
        }

        /** Passes native bytes in place, or copies heap ones into the call's memory. */
        MemorySegment input(MemorySegment bytes) {
            if (bytes.isNative()) {
                pass(bytes);
                return bytes;
            }
            return memory.allocate(bytes.byteSize(), 8).copyFrom(bytes);
        }

        /** Copies addresses into the call's memory as the array C reads. */
        MemorySegment addresses(MemorySegment[] addresses) {
            MemorySegment array = memory.allocate(Library.POINTER, Math.max(addresses.length, 1));
            for (int order = 0; order != addresses.length; ++order) {
                array.setAtIndex(Library.POINTER, order, addresses[order]);
            }
            return array;
        }

        /** Fills the {@code usearch_bytes_view_t} embedded at {@code at}. */
        void bytes(long at, MemorySegment bytes) {
            Library.BYTES_DATA.set(arguments, at, input(bytes));
            Library.BYTES_LENGTH.set(arguments, at, bytes.byteSize());
        }

        /** Fills the {@code usearch_bytes_view_t} embedded at {@code at} with a path, never terminated. */
        void path(long at, String path) {
            bytes(at, MemorySegment.ofArray(path.getBytes(StandardCharsets.UTF_8)));
        }

        /** Fills a RAM placement at {@code at} of {@code struct}, leaving an empty run as the null C derives from. */
        void ram(MemorySegment struct, long at, int[] domains, @Nullable CPUClass cpuClassCeiling) {
            if (domains.length != 0) {
                Library.RAM_DATA.set(struct, at, memory.allocateFrom(ValueLayout.JAVA_INT, domains));
                Library.RAM_COUNT.set(struct, at, (long) domains.length);
            }
            if (cpuClassCeiling != null) {
                Library.RAM_CPU_CLASS_CEILING.set(struct, at, cpuClassCeiling.code());
            }
        }

        /** Fills a VRAM placement at {@code at} of {@code struct}, leaving an empty run as the null C derives from. */
        void vram(MemorySegment struct, long at, int[] devices, @Nullable GPUClass gpuClassCeiling) {
            if (devices.length != 0) {
                Library.VRAM_DATA.set(struct, at, memory.allocateFrom(ValueLayout.JAVA_INT, devices));
                Library.VRAM_COUNT.set(struct, at, (long) devices.length);
            }
            if (gpuClassCeiling != null) {
                Library.VRAM_GPU_CLASS_CEILING.set(struct, at, gpuClassCeiling.code());
            }
        }

        /** Fills an NVMe placement at {@code at} of {@code struct}, leaving an empty run as the null C derives from. */
        void nvme(MemorySegment struct, long at, String[] paths, @Nullable CPUClass cpuClassCeiling) {
            if (paths.length != 0) {
                MemorySegment views = memory.allocate(Library.BYTES_VIEW, paths.length);
                for (int order = 0; order != paths.length; ++order) {
                    byte[] encoded = paths[order].getBytes(StandardCharsets.UTF_8);
                    long view = order * Library.BYTES_VIEW.byteSize();
                    Library.BYTES_DATA.set(views, view, memory.allocateFrom(ValueLayout.JAVA_BYTE, encoded));
                    Library.BYTES_LENGTH.set(views, view, (long) encoded.length);
                }
                Library.NVME_PATHS.set(struct, at, views);
                Library.NVME_COUNT.set(struct, at, (long) paths.length);
            }
            if (cpuClassCeiling != null) {
                Library.NVME_CPU_CLASS_CEILING.set(struct, at, cpuClassCeiling.code());
            }
        }

        /** Copies keys into the call's memory as the packed run C reads. */
        MemorySegment staged(Key[] keys) {
            MemorySegment staged = memory.allocate((long) Key.BYTES * Math.max(keys.length, 1), 8);
            for (int row = 0; row != keys.length; ++row) {
                keys[row].write(staged, (long) row * Key.BYTES);
            }
            return staged;
        }

        /** Fills the {@code usearch_keys_view_t} embedded at {@code at} with one key per row. */
        void keys(long at, Key[] keys) {
            Library.KEYS_DATA.set(arguments, at, staged(keys));
            Library.KEYS_STRIDE.set(arguments, at, (long) Key.BYTES);
        }

        /** Fills the {@code usearch_rows_view_t} embedded at {@code at} over a matrix the call reads. */
        void rows(long at, RowsView view) {
            Library.ROWS_DATA.set(arguments, at, input(view.segment().asSlice(0, view.spanBytes())));
            Library.ROWS_STRIDE.set(arguments, at, view.strideBytes());
            Library.ROWS_COLUMNS.set(arguments, at, view.columns());
            Library.ROWS_SCALAR.set(arguments, at, view.scalar().code());
        }

        /**
         * Points {@code field} at the arena the call writes its outputs into.
         *
         * @throws IllegalStateException on a closed arena.
         */
        void arena(VarHandle field, Arena held) {
            pointer(field, held.address());
            arena = held;
            generation = held.generation() + 1;
            pass(held);
        }

        /** The arena this call writes into, and the generation it brings it to. */
        Stamp stamp() {
            return new Stamp(Objects.requireNonNull(arena), generation);
        }

        /** Widens a pointer this call wrote into its arena to the {@code length} bytes it addresses. */
        MemorySegment reach(MemorySegment pointer, long length) {
            return Objects.requireNonNull(arena).reach(pointer, length);
        }

        /** Points {@code field} at a fresh traversal-stats struct, copied out once the call succeeded. */
        void stats(VarHandle field, DenseIndex.@Nullable DenseTraversalStats stats) {
            if (stats != null) {
                MemorySegment reported = output(field, Library.TRAVERSAL_STATS);
                reportCopies.add(() -> stats.read(reported));
            }
        }

        /** Fills the {@code usearch_predicate_t} embedded at {@code at} with an upcall reaching {@code predicate}. */
        @SuppressWarnings("restricted")
        void predicate(long at, @Nullable Predicate<Key> predicate) {
            if (predicate != null) {
                MemorySegment stub = Linker.nativeLinker().upcallStub(ADMISSION.bindTo(predicate), ADMITTING, memory);
                Library.PREDICATE_CALLBACK.set(arguments, at, stub);
            }
        }

        /** Fills the {@code usearch_progress_t} embedded at {@code at} with an upcall reaching {@code progress}. */
        @SuppressWarnings("restricted")
        void progress(long at, @Nullable Progress progress) {
            if (progress != null) {
                MemorySegment stub =
                        Linker.nativeLinker().upcallStub(CONTINUATION.bindTo(progress), CONTINUING, memory);
                Library.PROGRESS_CALLBACK.set(arguments, at, stub);
            }
        }
    }

    /**
     * Hears how far a long call has advanced, in whatever units it consumes, and decides whether it
     * goes on. Never called concurrently, and never after the call it was passed to is final.
     *
     * <p>
     * It must not throw; an exception escaping it terminates the process.
     */
    @FunctionalInterface
    public interface Progress {

        /**
         * Hears that {@code completed} of {@code total} units are done.
         *
         * @return true to go on, false to cancel, which the call answers with CANCELLED.
         */
        boolean proceed(long completed, long total);
    }

    /** The kind of work a refine or reshard plan is doing, a reshard's moves first, then a round's phases in order. */
    public enum Phase {

        /** No plan, or one that has not started. */
        NONE,
        /** A reshard copying source entries into their target with their links shifted. */
        COPY,
        /** A reshard building graphs over entries that arrived with no links. */
        BUILD,
        /** A reshard cutting a source into one part per target. */
        PARTITION,
        /** A reshard moving each part into its target. */
        SCATTER,
        /** A round's health check, opening or closing it. */
        HEALTH,
        /** A round linking each island with the root's component. */
        HEAL,
        /** A round drawing the hubs its gather streams entries past. */
        HUBS,
        /** A round gathering every entry into a neighborhood. */
        GATHER,
        /** A round re-stitching each neighborhood. */
        RESTITCH,
        /** A plan filing what its moves or rounds produced where it lands. */
        WRITE,
        /** The plan ended its round, and nothing is pending. */
        DONE;

        /** Names the phase a code holds. */
        static Phase of(int code) {
            return values()[code];
        }

        /** Renders the phase as the library spells it. Calls {@code usearch_phase_name}. */
        @Override
        public String toString() {
            return Library.text(PHASE_NAME, ordinal());
        }
    }

    /**
     * Where a plan stands once a call naming it returned.
     *
     * @param phase the kind of work in hand.
     * @param ordinal this phase's position among the ones the plan runs.
     * @param completed units of this phase done, entries or islands.
     * @param total units of this phase in all.
     */
    public record PhaseProgress(Phase phase, long ordinal, long completed, long total) {

        /** Lifts a {@code usearch_phase_progress_t}. */
        static PhaseProgress lift(MemorySegment progress) {
            return new PhaseProgress(
                    Phase.of((int) Library.PHASE_PHASE.get(progress, 0L)),
                    (long) Library.PHASE_ORDINAL.get(progress, 0L),
                    (long) Library.PHASE_COMPLETED.get(progress, 0L),
                    (long) Library.PHASE_TOTAL.get(progress, 0L));
        }
    }

    /** Admits each of {@code count} keys {@code predicate} accepts, as an upcall from a walk. */
    @SuppressWarnings("restricted")
    private static void admission(
            Predicate<Key> predicate, MemorySegment keys, MemorySegment admissions, long count, MemorySegment state) {
        MemorySegment named = keys.reinterpret(count * Key.BYTES);
        MemorySegment admitted = admissions.reinterpret(count * Library.ENUM.byteSize());
        for (long entry = 0; entry != count; ++entry) {
            admitted.setAtIndex(Library.ENUM, entry, predicate.test(Key.read(named, entry * Key.BYTES)) ? 1 : 0);
        }
    }

    /** Answers whether a long call goes on, as an upcall from its engine. */
    private static int continuation(Progress progress, long completed, long total, MemorySegment state) {
        return progress.proceed(completed, total) ? 0 : 1;
    }

    /** Releases every handle, arena and deferred call the program dropped without closing. */
    static final Cleaner CLEANER = Cleaner.create();

    /** {@code usearch_predicate_callback_t}. */
    private static final FunctionDescriptor ADMITTING =
            FunctionDescriptor.ofVoid(Library.POINTER, Library.POINTER, Library.SIZE, Library.POINTER);

    /** {@code usearch_progress_callback_t}. */
    private static final FunctionDescriptor CONTINUING =
            FunctionDescriptor.of(Library.ENUM, Library.SIZE, Library.SIZE, Library.POINTER);

    private static final MethodHandle ADMISSION;
    private static final MethodHandle CONTINUATION;

    static {
        MethodHandles.Lookup lookup = MethodHandles.lookup();
        try {
            ADMISSION = lookup.findStatic(
                    Types.class, "admission", ADMITTING.toMethodType().insertParameterTypes(0, Predicate.class));
            CONTINUATION = lookup.findStatic(
                    Types.class, "continuation", CONTINUING.toMethodType().insertParameterTypes(0, Progress.class));
        } catch (ReflectiveOperationException missing) {
            throw new ExceptionInInitializerError(missing);
        }
    }

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
    private static final MethodHandle HOST_ACCESS_NAME = Library.namer("usearch_host_access_name");
    private static final MethodHandle TRAVERSAL_NAME = Library.namer("usearch_traversal_name");
    private static final MethodHandle PHASE_NAME = Library.namer("usearch_phase_name");

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
                    Library.SIZE.withName("scratch_limit")));

    private static final long FOR_RAM_RAM = FOR_RAM.at("ram");
    private static final VarHandle FOR_RAM_THREADS = FOR_RAM.field("threads");

    /** {@code usearch_executor_for_vram_t}. */
    private static final Verb FOR_VRAM = new Verb(
            "usearch_executor_for_vram",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.placement("vram"),
                    Library.POINTER.withName("streams"),
                    Library.SIZE.withName("scratch_limit")));

    private static final long FOR_VRAM_VRAM = FOR_VRAM.at("vram");
    private static final VarHandle FOR_VRAM_STREAMS = FOR_VRAM.field("streams");
    private static final VarHandle FOR_VRAM_SCRATCH_LIMIT = FOR_VRAM.field("scratch_limit");

    /** {@code usearch_executor_for_nvme_t}. */
    private static final Verb FOR_NVME = new Verb(
            "usearch_executor_for_nvme",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.placement("nvme"),
                    Library.placement("ram"),
                    Library.SIZE.withName("threads"),
                    Library.ENUM.withName("nvme_class"),
                    Library.padding(),
                    Library.SIZE.withName("scratch_limit")));

    private static final long FOR_NVME_NVME = FOR_NVME.at("nvme");
    private static final VarHandle FOR_NVME_THREADS = FOR_NVME.field("threads");

    /** {@code usearch_executor_drain_t}. */
    private static final Verb DRAIN = new Verb(
            "usearch_executor_drain",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome"),
                    Library.ENUM.withName("mood"),
                    Library.padding()));

    private static final VarHandle DRAIN_MOOD = DRAIN.field("mood");

    private static final MethodHandle EXECUTOR_FREE =
            Library.downcall("usearch_executor_free", FunctionDescriptor.of(Library.ENUM, Library.POINTER));

    /** {@code usearch_arena_init_t}. */
    private static final Verb ARENA_INIT = new Verb(
            "usearch_arena_init",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("arena"),
                    Library.POINTER.withName("executor"),
                    Library.POINTER.withName("outcome")));

    private static final VarHandle ARENA_INIT_ARENA = ARENA_INIT.field("arena");

    private static final MethodHandle ARENA_EXPORT_OPEN =
            Library.downcall("usearch_arena_export_open", FunctionDescriptor.of(Library.ENUM, Library.POINTER));

    private static final MethodHandle ARENA_EXPORT_CLOSE =
            Library.downcall("usearch_arena_export_close", FunctionDescriptor.ofVoid(Library.POINTER));

    private static final MethodHandle ARENA_GENERATION =
            Library.downcall("usearch_arena_generation", FunctionDescriptor.of(ValueLayout.JAVA_LONG, Library.POINTER));

    /** {@code usearch_arena_free}, which refuses while an Arrow export of the outputs is open. */
    private static final MethodHandle ARENA_FREE =
            Library.downcall("usearch_arena_free", FunctionDescriptor.of(Library.ENUM, Library.POINTER));

    /** {@code usearch_arena_description_t}. */
    private static final StructLayout ARENA_DESCRIPTION = MemoryLayout.structLayout(
            Library.ENUM.withName("residency"),
            ValueLayout.JAVA_INT.withName("device"),
            Library.ENUM.withName("vendor"),
            ValueLayout.JAVA_INT.withName("vendor_ordinal"),
            Library.ENUM.withName("host_access"));

    private static final VarHandle DESCRIPTION_RESIDENCY = Library.field(ARENA_DESCRIPTION, "residency");
    private static final VarHandle DESCRIPTION_DEVICE = Library.field(ARENA_DESCRIPTION, "device");
    private static final VarHandle DESCRIPTION_VENDOR = Library.field(ARENA_DESCRIPTION, "vendor");
    private static final VarHandle DESCRIPTION_VENDOR_ORDINAL = Library.field(ARENA_DESCRIPTION, "vendor_ordinal");
    private static final VarHandle DESCRIPTION_HOST_ACCESS = Library.field(ARENA_DESCRIPTION, "host_access");

    /** {@code usearch_arena_describe_t}. */
    private static final Verb ARENA_DESCRIBE = new Verb(
            "usearch_arena_describe",
            MemoryLayout.structLayout(
                    Library.POINTER.withName("arena"),
                    Library.POINTER.withName("outcome"),
                    Library.POINTER.withName("description")));

    private static final VarHandle ARENA_DESCRIBE_ARENA = ARENA_DESCRIBE.field("arena");
    private static final VarHandle ARENA_DESCRIBE_DESCRIPTION = ARENA_DESCRIBE.field("description");
}
