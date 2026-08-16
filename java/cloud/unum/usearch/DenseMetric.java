package cloud.unum.usearch;

/**
 * A distance kind supported out of the box, with SIMD kernels dispatched by the machine's
 * capabilities. All of them derive from a single inner-product kernel with pre-computed magnitudes;
 * Hamming and Jaccard are bit-level, over the packed sub-byte kinds.
 *
 * <p>
 * The C layer's custom-callback escape hatch stays unexposed, as in the Go and Rust bindings: a
 * caller-supplied metric forfeits the SIMD kernel chains and the magnitude cache both.
 */
public enum DenseMetric {

    /** A metric the library could not name. */
    UNKNOWN(0, "unknown"),
    /** The inner product. */
    INNER(1, "inner"),
    /** The cosine distance. */
    ANGULAR(2, "angular"),
    /** The squared euclidean distance. */
    EUCLIDEAN(3, "euclidean"),
    /** The count of differing bits. */
    HAMMING(4, "hamming"),
    /** The bit-level Jaccard distance. */
    JACCARD(5, "jaccard");

    private final int code;
    private final String stem;

    DenseMetric(int code, String stem) {
        this.code = code;
        this.stem = stem;
    }

    /** The C enumerator's numeric value. */
    public int code() {
        return code;
    }

    /** Names the metric a code carries; a code this build cannot name becomes UNKNOWN. */
    public static DenseMetric of(int code) {
        for (DenseMetric candidate : values()) {
            if (candidate.code == code) {
                return candidate;
            }
        }
        return UNKNOWN;
    }

    /** Renders the metric as its C enumerator's stem. */
    @Override
    public String toString() {
        return stem;
    }
}
