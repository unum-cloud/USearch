package cloud.unum.usearch;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.ProviderNotFoundException;
import java.nio.file.StandardCopyOption;

/**
 * Loads the JNI shared object, once per class loader.
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

    /**
     * Ensures the shared object is loaded before a caller reaches a native method. The body is empty
     * on purpose: invoking it initializes this class, and that initialization runs the loader above.
     */
    static void require() {}

    /**
     * Loads from the library path, falling back to the copy packaged in the JAR.
     *
     * @throws UnsatisfiedLinkError when neither source yields a loadable library.
     */
    private static void load() {
        try {
            System.loadLibrary("usearch_jni");
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
            return "libusearch_jni.dylib";
        }
        if (operatingSystem.contains("windows")) {
            return "usearch_jni.dll";
        }
        return "libusearch_jni.so";
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
