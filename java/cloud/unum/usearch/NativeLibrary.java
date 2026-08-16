package cloud.unum.usearch;

import java.io.IOException;

/**
 * Loads the JNI shared object, once per class loader.
 *
 * <p>
 * The system library path is tried first, which is what a locally built tree and the test task
 * both offer, and the platform-specific copy packaged inside the JAR is the fallback.
 */
final class NativeLibrary {

    private static final Object LOADED = load();

    private NativeLibrary() {}

    /** Forces the loader to have run before the caller touches a native method. */
    static void require() {
        assert LOADED != null;
    }

    private static Object load() {
        try {
            System.loadLibrary("usearch_jni");
        } catch (UnsatisfiedLinkError missing) {
            try {
                loadFromJar();
            } catch (IOException failure) {
                throw new UnsatisfiedLinkError(
                        "usearch: the native library loaded from neither the library path nor the JAR: "
                                + failure.getMessage());
            }
        }
        return NativeLibrary.class;
    }

    private static void loadFromJar() throws IOException {
        String name = libraryName();
        String[] candidates = {"/usearch-native/" + platform() + "/" + name, "/usearch-native/" + name};
        IOException last = null;
        for (String candidate : candidates) {
            try {
                NativeUtils.loadLibraryFromJar(candidate);
                return;
            } catch (IOException absent) {
                last = absent;
            }
        }
        throw new IOException("tried " + String.join(", ", candidates), last);
    }

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

    private static String platform() {
        String operatingSystem = System.getProperty("os.name", "").toLowerCase();
        String architecture = System.getProperty("os.arch", "").toLowerCase();
        String normalized;
        if (architecture.equals("amd64") || architecture.equals("x86_64")) {
            normalized = "amd64";
        } else if (architecture.equals("aarch64") || architecture.equals("arm64")) {
            normalized = "arm64";
        } else if (architecture.equals("x86") || architecture.equals("i386")) {
            normalized = "x86";
        } else if (architecture.contains("armv7")) {
            normalized = "arm32";
        } else {
            normalized = architecture;
        }
        boolean android = System.getProperty("java.vendor", "").toLowerCase().contains("android")
                || System.getProperty("java.vm.name", "").toLowerCase().contains("dalvik");
        if (operatingSystem.contains("mac") || operatingSystem.contains("darwin")) {
            return "darwin-" + normalized;
        }
        if (operatingSystem.contains("windows")) {
            return "windows-" + normalized;
        }
        return (android ? "android-" : "linux-") + normalized;
    }
}
