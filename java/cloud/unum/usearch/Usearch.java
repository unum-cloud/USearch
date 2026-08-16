package cloud.unum.usearch;

/** Introspection of the linked library: its version, and the hardware it can and does use. */
public final class Usearch {

    static {
        NativeLibrary.require();
    }

    private Usearch() {}

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

    private static native String nativeVersion();

    private static native String nativeCapabilitiesCompiled();

    private static native String nativeCapabilitiesAvailable();
}
