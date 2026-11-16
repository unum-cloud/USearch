/**
 *  @file java/cloud/unum/usearch/Library.java
 *  @author Ash Vardanian
 *  @date September 8, 2026
 *  @brief Loads the engine's shared object once per class loader and links the C interface through it.
 */
package cloud.unum.usearch;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.VarHandle;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.ProviderNotFoundException;
import java.nio.file.StandardCopyOption;

/**
 * Loads the engine's shared object once per class loader, and links the C interface through it.
 *
 * <p>
 * The system library path is tried first, which is what a locally built tree and the test task both
 * offer; the platform-specific copy packaged inside the JAR is the fallback, and is extracted to a
 * temporary directory before it can be loaded, since no loader reads a shared object out of an
 * archive.
 *
 * @see <a href="https://github.com/adamheinrich/native-utils">Adam Heinrich's native-utils</a>,
 *     published under the MIT license, from which the extraction path derives.
 */
final class Library {

    static {
        load();
    }

    private Library() {}

    /** The engine's symbols, reachable because the loader above put them in this loader's set. */
    private static final SymbolLookup SYMBOLS = SymbolLookup.loaderLookup();

    private static final Linker LINKER = Linker.nativeLinker();

    /**
     * The handle for one C verb.
     *
     * <p>
     * A verb the engine does not export answers with a handle of the right shape that raises when
     * it is called. Failing at the call rather than at class initialization is what keeps the rest
     * of the binding usable against an engine that implements only part of the interface - which
     * every engine does while a modality is still being implemented.
     *
     * @param name The exported symbol.
     * @param descriptor Its argument and return layouts.
     * @param options Linker options, such as the captured-state or critical annotations.
     * @return A handle whose {@code invokeExact} matches {@code descriptor} exactly.
     */
    @SuppressWarnings("restricted")
    static MethodHandle downcall(String name, FunctionDescriptor descriptor, Linker.Option... options) {
        return SYMBOLS.find(name)
                .map(symbol -> LINKER.downcallHandle(symbol, descriptor, options))
                .orElseGet(() -> absent(name, descriptor.toMethodType()));
    }

    /** A handle of @p shape that raises rather than calling, for a verb the engine never defined. */
    private static MethodHandle absent(String name, MethodType shape) {
        MethodHandle raises = MethodHandles.throwException(shape.returnType(), UnsatisfiedLinkError.class);
        raises = MethodHandles.insertArguments(
                raises, 0, new UnsatisfiedLinkError("usearch: the engine exports no " + name));
        return MethodHandles.dropArguments(raises, 0, shape.parameterList());
    }

    /** @c size_t and every count the ABI spells with it. */
    static final ValueLayout.OfLong SIZE = ValueLayout.JAVA_LONG;
    /** An enumerator, which C sizes as an @c int. */
    static final ValueLayout.OfInt ENUM = ValueLayout.JAVA_INT;
    /** Any pointer the ABI hands over or takes back. */
    static final AddressLayout POINTER = ValueLayout.ADDRESS;

    /** The four bytes C inserts after an enumerator that a pointer or a @c size_t follows. */
    static MemoryLayout padding() {
        return MemoryLayout.paddingLayout(4);
    }

    /** @c usearch_outcome_t - the status every verb reports through, and its static message. */
    static final StructLayout OUTCOME = MemoryLayout.structLayout(
                    ENUM.withName("status"), padding(), POINTER.withName("message"))
            .withName("usearch_outcome_t");

    /** @c usearch_key_t - sixteen opaque bytes, embedded by value. */
    static final StructLayout KEY = MemoryLayout.structLayout(
                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE).withName("bytes"))
            .withName("usearch_key_t");

    /** @c usearch_bytes_view_t - a borrowed byte run, which a path is. */
    static final StructLayout BYTES_VIEW = MemoryLayout.structLayout(POINTER.withName("data"), SIZE.withName("length"))
            .withName("usearch_bytes_view_t");

    /** @c usearch_keys_view_t - one key per row of the call's count. */
    static final StructLayout KEYS_VIEW = MemoryLayout.structLayout(POINTER.withName("data"), SIZE.withName("stride"))
            .withName("usearch_keys_view_t");

    /** @c usearch_rows_view_t - a strided run of rows, read from the caller or written into an arena. */
    static final StructLayout ROWS_VIEW = MemoryLayout.structLayout(
                    POINTER.withName("data"),
                    SIZE.withName("stride"),
                    SIZE.withName("columns"),
                    ENUM.withName("scalar"),
                    padding())
            .withName("usearch_rows_view_t");

    /** @c usearch_matches_view_t - a ranked run a call wrote into its arena, @c columns packed columns per query. */
    static final StructLayout MATCHES_VIEW = MemoryLayout.structLayout(
                    POINTER.withName("keys"),
                    POINTER.withName("scores"),
                    POINTER.withName("counts"),
                    SIZE.withName("columns"))
            .withName("usearch_matches_view_t");

    /** @c usearch_matches_tape_view_t - a ragged run of matches a call wrote into its arena. */
    static final StructLayout MATCHES_TAPE_VIEW = MemoryLayout.structLayout(
                    POINTER.withName("keys"), POINTER.withName("scores"), POINTER.withName("offsets"))
            .withName("usearch_matches_tape_view_t");

    /** @c usearch_ram_options_t - the memory domains shards go to, their capacity, the CPU class pinned,
     *  and the bytes a view's key tables may take on each domain. */
    static final StructLayout RAM_OPTIONS = MemoryLayout.structLayout(
                    POINTER.withName("data"),
                    SIZE.withName("count"),
                    SIZE.withName("shard_capacity"),
                    ENUM.withName("cpu_class"),
                    padding(),
                    SIZE.withName("memory_limit"))
            .withName("usearch_ram_options_t");

    /** @c usearch_vram_options_t - the devices shards go to, their capacity, and the GPU class pinned. */
    static final StructLayout VRAM_OPTIONS = MemoryLayout.structLayout(
                    POINTER.withName("data"),
                    SIZE.withName("count"),
                    SIZE.withName("shard_capacity"),
                    ENUM.withName("gpu_class"),
                    padding())
            .withName("usearch_vram_options_t");

    /** @c usearch_nvme_options_t - the paths shards serve from, and the CPU class pinned. */
    static final StructLayout NVME_OPTIONS = MemoryLayout.structLayout(
                    POINTER.withName("data"), SIZE.withName("count"), ENUM.withName("cpu_class"), padding())
            .withName("usearch_nvme_options_t");

    /** The placement layout a residency's verbs take, named @p residency : ram, vram or nvme. */
    static StructLayout placement(String residency) {
        StructLayout layout =
                switch (residency) {
                    case "ram" -> RAM_OPTIONS;
                    case "vram" -> VRAM_OPTIONS;
                    default -> NVME_OPTIONS;
                };
        return layout.withName(residency);
    }

    /** @c usearch_keys_tape_view_t - a ragged run of keys a call wrote into its arena. */
    static final StructLayout KEYS_TAPE_VIEW = MemoryLayout.structLayout(
                    POINTER.withName("keys"), POINTER.withName("offsets"))
            .withName("usearch_keys_tape_view_t");

    /** @c usearch_traversal_stats_t - a pure output, fully overwritten by its call. */
    static final StructLayout TRAVERSAL_STATS = MemoryLayout.structLayout(
                    SIZE.withName("visited_members"),
                    SIZE.withName("computed_distances"),
                    SIZE.withName("restarts"),
                    SIZE.withName("missing_hints"))
            .withName("usearch_traversal_stats_t");

    /** @c usearch_predicate_t, the handler a walk asks which candidate keys it may report. */
    static final StructLayout PREDICATE = MemoryLayout.structLayout(
                    POINTER.withName("callback"), POINTER.withName("state"), ENUM.withName("cost"), padding())
            .withName("predicate");

    /** @c usearch_progress_t, the handler a long call reports its advance to. */
    static final StructLayout PROGRESS = MemoryLayout.structLayout(
                    POINTER.withName("callback"), POINTER.withName("state"), SIZE.withName("steps"))
            .withName("progress");

    /** @c usearch_phase_progress_t, where a plan stands. */
    static final StructLayout PHASE_PROGRESS = MemoryLayout.structLayout(
                    ENUM.withName("phase"),
                    padding(),
                    SIZE.withName("ordinal"),
                    SIZE.withName("completed"),
                    SIZE.withName("total"))
            .withName("usearch_phase_progress_t");

    /**
     * The fields that open every verb on a handle - the handle, the executor, the outcome, the
     * completion - then @p rest.
     */
    static StructLayout opening(MemoryLayout... rest) {
        MemoryLayout[] fields = new MemoryLayout[4 + rest.length];
        fields[0] = POINTER.withName("index");
        fields[1] = POINTER.withName("executor");
        fields[2] = POINTER.withName("outcome");
        fields[3] = ENUM.withName("completion");
        System.arraycopy(rest, 0, fields, 4, rest.length);
        return MemoryLayout.structLayout(fields);
    }

    /** The offset of one possibly nested field, named by the path of member names leading to it. */
    static long offset(MemoryLayout layout, String... path) {
        return layout.byteOffset(elements(path));
    }

    /**
     * The handle of one possibly nested field, named by the path of member names leading to it, whose
     * coordinates are a segment and the offset of the structure inside it.
     */
    static VarHandle field(MemoryLayout layout, String... path) {
        return layout.varHandle(elements(path));
    }

    private static MemoryLayout.PathElement[] elements(String... path) {
        MemoryLayout.PathElement[] elements = new MemoryLayout.PathElement[path.length];
        for (int depth = 0; depth != path.length; ++depth) {
            elements[depth] = MemoryLayout.PathElement.groupElement(path[depth]);
        }
        return elements;
    }

    static final VarHandle BYTES_DATA = field(BYTES_VIEW, "data");
    static final VarHandle BYTES_LENGTH = field(BYTES_VIEW, "length");
    static final VarHandle KEYS_DATA = field(KEYS_VIEW, "data");
    static final VarHandle KEYS_STRIDE = field(KEYS_VIEW, "stride");
    static final VarHandle ROWS_DATA = field(ROWS_VIEW, "data");
    static final VarHandle ROWS_STRIDE = field(ROWS_VIEW, "stride");
    static final VarHandle ROWS_COLUMNS = field(ROWS_VIEW, "columns");
    static final VarHandle ROWS_SCALAR = field(ROWS_VIEW, "scalar");
    static final VarHandle MATCHES_KEYS = field(MATCHES_VIEW, "keys");
    static final VarHandle MATCHES_SCORES = field(MATCHES_VIEW, "scores");
    static final VarHandle MATCHES_COUNTS = field(MATCHES_VIEW, "counts");
    static final VarHandle MATCHES_COLUMNS = field(MATCHES_VIEW, "columns");
    static final VarHandle TAPE_KEYS = field(MATCHES_TAPE_VIEW, "keys");
    static final VarHandle TAPE_SCORES = field(MATCHES_TAPE_VIEW, "scores");
    static final VarHandle TAPE_OFFSETS = field(MATCHES_TAPE_VIEW, "offsets");
    static final VarHandle KEYS_TAPE_KEYS = field(KEYS_TAPE_VIEW, "keys");
    static final VarHandle KEYS_TAPE_OFFSETS = field(KEYS_TAPE_VIEW, "offsets");
    static final VarHandle RAM_DATA = field(RAM_OPTIONS, "data");
    static final VarHandle RAM_COUNT = field(RAM_OPTIONS, "count");
    static final VarHandle RAM_SHARD_CAPACITY = field(RAM_OPTIONS, "shard_capacity");
    static final VarHandle RAM_CPU_CLASS = field(RAM_OPTIONS, "cpu_class");
    static final VarHandle VRAM_DATA = field(VRAM_OPTIONS, "data");
    static final VarHandle VRAM_COUNT = field(VRAM_OPTIONS, "count");
    static final VarHandle VRAM_SHARD_CAPACITY = field(VRAM_OPTIONS, "shard_capacity");
    static final VarHandle VRAM_GPU_CLASS = field(VRAM_OPTIONS, "gpu_class");
    static final VarHandle NVME_PATHS = field(NVME_OPTIONS, "data");
    static final VarHandle NVME_COUNT = field(NVME_OPTIONS, "count");
    static final VarHandle NVME_CPU_CLASS = field(NVME_OPTIONS, "cpu_class");
    static final VarHandle PREDICATE_CALLBACK = field(PREDICATE, "callback");
    static final VarHandle PROGRESS_CALLBACK = field(PROGRESS, "callback");
    static final VarHandle PHASE_PHASE = field(PHASE_PROGRESS, "phase");
    static final VarHandle PHASE_ORDINAL = field(PHASE_PROGRESS, "ordinal");
    static final VarHandle PHASE_COMPLETED = field(PHASE_PROGRESS, "completed");
    static final VarHandle PHASE_TOTAL = field(PHASE_PROGRESS, "total");

    /** A verb taking one argument structure, which is every verb but the name lookups and frees. */
    static MethodHandle verb(String name) {
        return downcall(name, FunctionDescriptor.ofVoid(POINTER));
    }

    /** One of the C name tables, which spell an enumerator as a static string. */
    static MethodHandle namer(String name) {
        return downcall(name, FunctionDescriptor.of(POINTER, ENUM));
    }

    /** Calls a verb or a free, either of which takes one pointer and answers nothing. */
    static void invoke(MethodHandle verb, MemorySegment argument) {
        try {
            verb.invokeExact(argument);
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /** Reads the static string a name query answers. */
    static String text(MethodHandle query) {
        try {
            return string((MemorySegment) query.invokeExact());
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /** Reads the static string a name table spells for @p code. */
    static String text(MethodHandle namer, int code) {
        try {
            return string((MemorySegment) namer.invokeExact(code));
        } catch (Throwable failure) {
            throw rethrow(failure);
        }
    }

    /** Reads a static, null-terminated C string, answering an empty one for null. */
    @SuppressWarnings("restricted")
    static String string(MemorySegment text) {
        return text.equals(MemorySegment.NULL)
                ? ""
                : text.reinterpret(Long.MAX_VALUE).getString(0);
    }

    /**
     * Raises whatever a final @p outcome carries, and answers nothing on success.
     *
     * @throws Types.USearchException when the verb reported a failure.
     */
    static void check(MemorySegment outcome) {
        int status = (int) OUTCOME_STATUS.get(outcome, 0L);
        if (status == 0) {
            return;
        }
        throw new Types.USearchException(
                Types.Status.of(status), string((MemorySegment) OUTCOME_MESSAGE.get(outcome, 0L)));
    }

    private static final VarHandle OUTCOME_STATUS = field(OUTCOME, "status");
    private static final VarHandle OUTCOME_MESSAGE = field(OUTCOME, "message");

    /**
     * Carries a downcall's failure out unchanged when it is already one of ours, and wraps anything
     * else - a linkage fault, a layout mismatch - as the internal error it is.
     */
    static RuntimeException rethrow(Throwable failure) {
        if (failure instanceof RuntimeException raised) {
            return raised;
        }
        if (failure instanceof Error raised) {
            throw raised;
        }
        return new IllegalStateException("usearch: the C interface could not be reached", failure);
    }

    /**
     * Loads from the library path, falling back to the copy packaged in the JAR.
     *
     * @throws UnsatisfiedLinkError when neither source yields a loadable library.
     */
    @SuppressWarnings("restricted")
    private static void load() {
        try {
            System.loadLibrary("usearch_shared");
        } catch (UnsatisfiedLinkError missing) {
            try {
                loadFromJar();
            } catch (IOException absent) {
                throw new UnsatisfiedLinkError(
                        "usearch: the native library loaded from neither the library path nor the JAR: "
                                + absent.getMessage());
            }
        }
    }

    /**
     * Tries the platform-specific resource, then the flat one a single-platform JAR carries. Only a
     * missing resource advances to the next candidate; one that extracts and then fails to link
     * leaves here as the {@link UnsatisfiedLinkError} it was.
     *
     * @throws IOException when no candidate is packaged, or none can be written to disk.
     * @throws UnsatisfiedLinkError when a candidate extracts but the platform refuses to link it.
     */
    private static void loadFromJar() throws IOException {
        String name = libraryName();
        String[] candidates = {"/usearch-native/" + platform() + "/" + name, "/usearch-native/" + name};
        IOException absent = null;
        for (String candidate : candidates) {
            File extracted;
            try {
                extracted = extract(candidate);
            } catch (IOException missing) {
                absent = missing;
                continue;
            }
            loadExtracted(extracted);
            return;
        }
        throw new IOException("tried " + String.join(", ", candidates), absent);
    }

    /**
     * Copies one packaged shared object into a temporary directory of its own.
     *
     * @throws IOException when the resource is absent from the JAR or cannot be written to disk.
     */
    private static File extract(String resource) throws IOException {
        Path directory = Files.createTempDirectory("usearch");
        directory.toFile().deleteOnExit();
        File extracted = new File(directory.toFile(), resource.substring(resource.lastIndexOf('/') + 1));
        try (InputStream packaged = Library.class.getResourceAsStream(resource)) {
            if (packaged == null) {
                throw new FileNotFoundException(resource + " is not packaged in the JAR");
            }
            Files.copy(packaged, extracted.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException failure) {
            extracted.delete();
            throw failure;
        }
        return extracted;
    }

    /**
     * Loads an extracted copy and unlinks it. The copy goes immediately on a POSIX filesystem,
     * where an open shared object survives the removal of the file it came from, and at exit
     * everywhere else.
     *
     * @throws UnsatisfiedLinkError when the platform refuses to link the extracted copy.
     */
    @SuppressWarnings("restricted")
    private static void loadExtracted(File extracted) {
        try {
            System.load(extracted.getAbsolutePath());
        } finally {
            if (posixCompliant()) {
                extracted.delete();
            } else {
                extracted.deleteOnExit();
            }
        }
    }

    /** Whether an open shared object survives the unlinking of the file it was loaded from. */
    private static boolean posixCompliant() {
        try {
            return FileSystems.getDefault().supportedFileAttributeViews().contains("posix");
        } catch (FileSystemNotFoundException | ProviderNotFoundException | SecurityException unreadable) {
            return false;
        }
    }

    /** The file name the platform's loader expects. */
    private static String libraryName() {
        String operatingSystem = System.getProperty("os.name", "").toLowerCase();
        if (operatingSystem.contains("mac") || operatingSystem.contains("darwin")) {
            return "libusearch_shared.dylib";
        }
        if (operatingSystem.contains("windows")) {
            return "libusearch_shared.dll";
        }
        return "libusearch_shared.so";
    }

    /**
     * The directory one platform's libraries sit under inside a multi-platform JAR. The
     * operating-system probe matches substrings because {@code os.name} carries a version -
     * {@code Mac OS X}, {@code Windows 11} - while the architecture below does not.
     */
    private static String platform() {
        String operatingSystem = System.getProperty("os.name", "").toLowerCase();
        if (operatingSystem.contains("mac") || operatingSystem.contains("darwin")) {
            return "darwin-" + architecture();
        }
        if (operatingSystem.contains("windows")) {
            return "windows-" + architecture();
        }
        return (android() ? "android-" : "linux-") + architecture();
    }

    /**
     * The architecture name the packaging writes. A 32-bit Arm runtime reports {@code arm} on Linux
     * and {@code armv7l} or {@code armv8l} on Android, and all three name the one {@code arm32}
     * build; a spelling nobody enumerated passes through unchanged.
     */
    private static String architecture() {
        String reported = System.getProperty("os.arch", "").toLowerCase();
        return switch (reported) {
            case "amd64", "x86_64" -> "amd64";
            case "aarch64", "arm64" -> "arm64";
            case "x86", "i386" -> "x86";
            case "arm", "armv7l", "armv8l" -> "arm32";
            default -> reported;
        };
    }

    /**
     * Whether the runtime is Android, whose libraries the JAR names under their own prefix. That
     * prefix is reachable on API 28 and below only, since from API 29 the platform refuses to link a
     * shared object out of writable application storage; a modern Android app needs an AAR carrying
     * the library under {@code lib/<abi>/} instead.
     */
    private static boolean android() {
        return System.getProperty("java.vendor", "").toLowerCase().contains("android")
                || System.getProperty("java.vm.name", "").toLowerCase().contains("dalvik");
    }
}
