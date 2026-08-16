package cloud.unum.usearch;

/**
 * A stored or exchanged scalar type, named by width and encoding. Sub-byte kinds pack several
 * dimensions per byte, as their {@code xN} suffixes spell.
 *
 * <p>
 * The enumerators are ordered by shrinking dynamic range, and their codes carry the numbering gaps
 * the C enum reserves for kinds that have not landed yet.
 */
public enum ScalarKind {

    /** A type the library could not name. */
    UNKNOWN(0, "unknown", 0, 1),
    /** 64-bit double precision. */
    F64(1, "f64", 8, 1),
    /** 32-bit single precision. */
    F32(2, "f32", 4, 1),
    /** The brain floating-point format. */
    BF16(3, "bf16", 2, 1),
    /** IEEE half precision. */
    F16(4, "f16", 2, 1),
    /** 8-bit floating point with five exponent bits. */
    E5M2(5, "e5m2", 1, 1),
    /** 8-bit floating point with four exponent bits. */
    E4M3(6, "e4m3", 1, 1),
    /** 6-bit floating point with three exponent bits, one per byte. */
    E3M2(7, "e3m2", 1, 1),
    /** 6-bit floating point with two exponent bits, one per byte. */
    E2M3(8, "e2m3", 1, 1),
    /** An 8-bit signed integer. */
    I8(10, "i8", 1, 1),
    /** Two 4-bit signed integers per byte. */
    I4X2(11, "i4x2", 1, 2),
    /** An 8-bit unsigned integer. */
    U8(20, "u8", 1, 1),
    /** Two 4-bit unsigned integers per byte. */
    U4X2(21, "u4x2", 1, 2),
    /** Eight single-bit dimensions per byte. */
    U1X8(22, "u1x8", 1, 8);

    private final int code;
    private final String stem;
    private final int valueBytes;
    private final int dimensionsPerValue;

    ScalarKind(int code, String stem, int valueBytes, int dimensionsPerValue) {
        this.code = code;
        this.stem = stem;
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
    public static ScalarKind of(int code) {
        for (ScalarKind candidate : values()) {
            if (candidate.code == code) {
                return candidate;
            }
        }
        return UNKNOWN;
    }

    /**
     * The bytes one row of the given width occupies.
     *
     * @throws IndexException when the width does not fill whole values of this kind.
     */
    public long rowBytes(long dimensions) {
        if (dimensions % dimensionsPerValue != 0) {
            throw new IndexException(
                    Status.SHAPE_MISMATCH, dimensions + " dimensions do not fill whole " + stem + " values");
        }
        return Extents.product(dimensions / dimensionsPerValue, valueBytes, "the row width");
    }

    /** Renders the scalar kind as its C enumerator's stem. */
    @Override
    public String toString() {
        return stem;
    }
}
