package cloud.unum.usearch;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
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
     * Ensures the shared object is loaded before a caller reaches the C interface. The body is empty
     * on purpose: invoking it initializes this class, and that initialization runs the loader above.
     */
    static void require() {}

    /**
     * The handle for one C verb.
     *
     * <p>
     * A verb the engine does not export answers with a handle of the right shape that raises when
     * it is called. Failing at the call rather than at class initialization is what keeps the rest
     * of the binding usable against an engine that implements only part of the interface - which
     * every engine does while a modality is still landing.
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
    static final java.lang.foreign.AddressLayout POINTER = ValueLayout.ADDRESS;

    /** @c usearch_error_t - the status every verb reports through, and its borrowed message. */
    static final StructLayout ERROR = MemoryLayout.structLayout(
                    ENUM.withName("status"), MemoryLayout.paddingLayout(4), POINTER.withName("message"))
            .withName("usearch_error_t");

    /** @c usearch_keys_view_t - one key per row the batch it accompanies carries. */
    static final StructLayout KEYS_VIEW = MemoryLayout.structLayout(POINTER.withName("data"), SIZE.withName("count"))
            .withName("usearch_keys_view_t");

    /** @c usearch_matrix_view_t - a strided run of rows the engine reads. */
    static final StructLayout MATRIX_VIEW = MemoryLayout.structLayout(
                    POINTER.withName("data"),
                    SIZE.withName("rows"),
                    SIZE.withName("columns"),
                    SIZE.withName("stride_bytes"),
                    ENUM.withName("scalar_kind"),
                    MemoryLayout.paddingLayout(4))
            .withName("usearch_matrix_view_t");

    /** @c usearch_matrix_span_t - its writable twin, whose capacity bounds what a verb fills. */
    static final StructLayout MATRIX_SPAN = MemoryLayout.structLayout(
                    POINTER.withName("data"),
                    SIZE.withName("capacity"),
                    SIZE.withName("columns"),
                    SIZE.withName("stride_bytes"),
                    ENUM.withName("scalar_kind"),
                    MemoryLayout.paddingLayout(4))
            .withName("usearch_matrix_span_t");

    /** @c usearch_matches_span_t - the three caller-owned buffers a ranked verb fills. */
    static final StructLayout MATCHES_SPAN = MemoryLayout.structLayout(
                    POINTER.withName("keys"),
                    SIZE.withName("keys_stride_bytes"),
                    POINTER.withName("scores"),
                    SIZE.withName("scores_stride_bytes"),
                    POINTER.withName("counts"),
                    SIZE.withName("wanted"))
            .withName("usearch_matches_span_t");

    /** @c usearch_dense_traversal_stats_t - a pure output, fully overwritten by its call. */
    static final StructLayout TRAVERSAL_STATS = MemoryLayout.structLayout(
                    SIZE.withName("visited_members"), SIZE.withName("computed_distances"), SIZE.withName("restarts"))
            .withName("usearch_dense_traversal_stats_t");

    /** @c usearch_ram_options_t, @c usearch_vram_options_t and @c usearch_nvme_options_t all pair
     *      a borrowed run with its count, so one layout serves the three. */
    static final StructLayout PLACEMENT = MemoryLayout.structLayout(POINTER.withName("data"), SIZE.withName("count"))
            .withName("usearch_placement_options_t");

    /** @c usearch_bytes_view_t - a borrowed byte run, which a path is. */
    static final StructLayout BYTES_VIEW = MemoryLayout.structLayout(POINTER.withName("data"), SIZE.withName("length"))
            .withName("usearch_bytes_view_t");

    /** @c usearch_key_t - sixteen opaque bytes, passed and returned by value. */
    static final StructLayout KEY = MemoryLayout.structLayout(
                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE).withName("bytes"))
            .withName("usearch_key_t");

    /** @c usearch_dense_options_t - the geometry, the metric, and the residency arm one of three. */
    static final StructLayout DENSE_OPTIONS = MemoryLayout.structLayout(
                    ENUM.withName("metric"),
                    MemoryLayout.paddingLayout(4),
                    POINTER.withName("metric_callback"),
                    POINTER.withName("metric_callback_state"),
                    ENUM.withName("quantization"),
                    MemoryLayout.paddingLayout(4),
                    SIZE.withName("dimensions"),
                    SIZE.withName("connectivity"),
                    SIZE.withName("values_per_key"),
                    ENUM.withName("residency"),
                    MemoryLayout.paddingLayout(4),
                    POINTER.withName("placement_data"),
                    SIZE.withName("placement_count"))
            .withName("usearch_dense_options_t");

    /** @c usearch_dense_metadata_t - the four fields a checkpoint froze. */
    static final StructLayout DENSE_METADATA = MemoryLayout.structLayout(
                    ENUM.withName("metric"),
                    ENUM.withName("quantization"),
                    SIZE.withName("dimensions"),
                    SIZE.withName("connectivity"))
            .withName("usearch_dense_metadata_t");

    /**
     * Raises whatever @p error carries, and answers nothing when it carries success.
     *
     * @throws Types.USearchException when the verb reported a failure.
     */
    @SuppressWarnings("restricted")
    static void raise(MemorySegment error) {
        int status = error.get(ENUM, ERROR.byteOffset(MemoryLayout.PathElement.groupElement("status")));
        if (status == 0) {
            return;
        }
        MemorySegment message = error.get(POINTER, ERROR.byteOffset(MemoryLayout.PathElement.groupElement("message")));
        throw Types.USearchException.of(
                status,
                message.equals(MemorySegment.NULL)
                        ? "no detail"
                        : message.reinterpret(Long.MAX_VALUE).getString(0));
    }

    /** A zeroed @c usearch_error_t in @p arena, which every verb writes its verdict into. */
    static MemorySegment error(Arena arena) {
        return arena.allocate(ERROR);
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
