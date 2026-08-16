package cloud.unum.usearch;

/**
 * The construction and loading options of a dense index.
 *
 * <p>
 * Every setter returns this instance, so an index is configured in one expression:
 *
 * <pre>{@code
 * try (DenseIndex index = new DenseOptions()
 *         .dimensions(768)
 *         .metric(DenseMetric.ANGULAR)
 *         .quantization(ScalarKind.I8)
 *         .open()) {
 *     index.reserve(1_000_000);
 * }
 * }</pre>
 */
public final class DenseOptions {

    /** Whether stored rows round up to the SIMD step quantum. */
    public enum DimensionPadding {
        /** Rows keep their natural width. */
        NATURAL,
        /**
         * Rows are cache-line aligned and kernels run tail-free, at slightly larger arenas. A
         * property of the stored layout, so it applies whichever residency traverses it.
         */
        SIMD_QUANTUM
    }

    private DenseMetric metric = DenseMetric.ANGULAR;
    private ScalarKind quantization = ScalarKind.BF16;
    private long dimensions = 0;
    private DimensionPadding dimensionPadding = DimensionPadding.NATURAL;
    private long connectivity = 0;
    private Residency residency = Residency.ram();

    /** An angular BF16 index in RAM, with every optional field derived. */
    public DenseOptions() {}

    /** The distance kind used between vectors. */
    public DenseOptions metric(DenseMetric metric) {
        this.metric = metric;
        return this;
    }

    /** The scalar type vectors are stored and quantized into. */
    public DenseOptions quantization(ScalarKind quantization) {
        this.quantization = quantization;
        return this;
    }

    /**
     * The width of the indexed vectors; zero defers the geometry to a later
     * {@link DenseIndex#load(String)}, which reads it off the checkpoint.
     */
    public DenseOptions dimensions(long dimensions) {
        if (dimensions < 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the dimensionality is negative");
        }
        this.dimensions = dimensions;
        return this;
    }

    /** Whether stored rows round up to the SIMD step quantum. */
    public DenseOptions dimensionPadding(DimensionPadding dimensionPadding) {
        this.dimensionPadding = dimensionPadding;
        return this;
    }

    /** The connections-per-node bound of the graph; zero derives. */
    public DenseOptions connectivity(long connectivity) {
        if (connectivity < 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the connectivity is negative");
        }
        this.connectivity = connectivity;
        return this;
    }

    /** Where the payload lives, and what loading a checkpoint means. */
    public DenseOptions residency(Residency residency) {
        this.residency = residency;
        return this;
    }

    /** The distance kind used between vectors. */
    public DenseMetric metric() {
        return metric;
    }

    /** The scalar type vectors are stored and quantized into. */
    public ScalarKind quantization() {
        return quantization;
    }

    /** The width of the indexed vectors; zero means the geometry is still unfrozen. */
    public long dimensions() {
        return dimensions;
    }

    /** Whether stored rows round up to the SIMD step quantum. */
    public DimensionPadding dimensionPadding() {
        return dimensionPadding;
    }

    /** The connections-per-node bound of the graph; zero means it derives. */
    public long connectivity() {
        return connectivity;
    }

    /** Where the payload lives, and what loading a checkpoint means. */
    public Residency residency() {
        return residency;
    }

    /** Initializes an index for the residency these options select. */
    public DenseIndex open() {
        return new DenseIndex(this);
    }

    /** Lowers the options into the flat layout the native layer reads. */
    long[] toWords() {
        long[] words = new long[OptionsWords.COUNT];
        words[OptionsWords.METRIC] = metric.code();
        words[OptionsWords.QUANTIZATION] = quantization.code();
        words[OptionsWords.DIMENSIONS] = dimensions;
        words[OptionsWords.DIMENSION_PADDING] = dimensionPadding == DimensionPadding.SIMD_QUANTUM ? 1 : 0;
        words[OptionsWords.CONNECTIVITY] = connectivity;
        residency.writeInto(words);
        return words;
    }

    /** Lifts the options a checkpoint's headers describe. */
    static DenseOptions fromWords(long[] words) {
        DenseOptions options = new DenseOptions();
        options.metric = DenseMetric.of(Extents.toIndex(words[OptionsWords.METRIC], "the metric code"));
        options.quantization = ScalarKind.of(Extents.toIndex(words[OptionsWords.QUANTIZATION], "the scalar code"));
        options.dimensions = words[OptionsWords.DIMENSIONS];
        options.dimensionPadding =
                words[OptionsWords.DIMENSION_PADDING] != 0 ? DimensionPadding.SIMD_QUANTUM : DimensionPadding.NATURAL;
        options.connectivity = words[OptionsWords.CONNECTIVITY];
        options.residency = Residency.readFrom(words);
        if (options.metric == DenseMetric.UNKNOWN) {
            throw new IndexException(Status.UNKNOWN_KIND, "the checkpoint names an unknown metric");
        }
        if (options.quantization == ScalarKind.UNKNOWN) {
            throw new IndexException(Status.UNKNOWN_KIND, "the checkpoint names an unknown scalar kind");
        }
        return options;
    }

    /**
     * Lowers the options and lifts them straight back - the marshalling round trip through the C
     * struct and its residency union, which nothing else on this side can reach.
     */
    DenseOptions reload() {
        long[] words = new long[OptionsWords.COUNT];
        DenseIndex.nativeReloadOptions(toWords(), words);
        return fromWords(words);
    }

    /** Renders the options as the fields a reader would set. */
    @Override
    public String toString() {
        return "DenseOptions(dimensions=" + dimensions
                + ", metric=" + metric
                + ", quantization=" + quantization
                + ", padding=" + dimensionPadding
                + ", connectivity=" + connectivity
                + ", residency=" + residency + ")";
    }
}
