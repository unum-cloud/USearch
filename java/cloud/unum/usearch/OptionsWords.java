package cloud.unum.usearch;

/**
 * The flat layout a {@link DenseOptions} travels in, one word per field.
 *
 * <p>
 * The C options struct ends in a residency-tagged union, and JNI has no way to describe a union to
 * Java. The three arms are therefore projected onto the same three trailing words, with
 * {@link #RESIDENCY} deciding what they mean - the tag doing on this side exactly what it does on
 * the other. The native layer is the only place that reads or writes the struct itself.
 */
final class OptionsWords {

    private OptionsWords() {}

    /** The metric kind's code. */
    static final int METRIC = 0;
    /** The stored quantization's code. */
    static final int QUANTIZATION = 1;
    /** The frozen dimensionality; zero defers the geometry to a later load. */
    static final int DIMENSIONS = 2;
    /** The dimension padding's ordinal. */
    static final int DIMENSION_PADDING = 3;
    /** The connections-per-node bound; zero derives. */
    static final int CONNECTIVITY = 4;
    /** The residency's code, selecting what the three words below mean. */
    static final int RESIDENCY = 5;
    /** RAM threads, VRAM device ordinal, or NVMe sector bytes. */
    static final int RESIDENCY_FIRST = 6;
    /** RAM huge-page demand, VRAM addressing, or NVMe queue depth. */
    static final int RESIDENCY_SECOND = 7;
    /** NVMe I/O mode; the other arms leave it zero. */
    static final int RESIDENCY_THIRD = 8;
    /** How many words one options structure occupies. */
    static final int COUNT = 9;

    /** The three traversal counters a call reports, in the order the native layer writes them. */
    static final int STATS_COUNT = 3;
}
