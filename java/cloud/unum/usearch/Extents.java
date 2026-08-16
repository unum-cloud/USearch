package cloud.unum.usearch;

/**
 * Overflow-checked extent arithmetic, standing between every caller-supplied count and the
 * allocation or narrowing it drives.
 *
 * <p>
 * The C interface bounds neither the wanted count nor the row count, so a product like
 * {@code rows × wanted × Key.BYTES} silently wraps - and a Java {@code int} wraps four billion
 * times sooner than a {@code size_t}. Every product feeding an array length, a stride, or a
 * boundary-crossing size passes through here, so a wrap becomes an OVERFLOW failure instead of a
 * short buffer the native side then writes past.
 */
final class Extents {

    private Extents() {}

    /** Multiplies two non-negative extents, reporting OVERFLOW instead of wrapping. */
    static long product(long left, long right, String what) {
        nonNegative(left, what);
        nonNegative(right, what);
        if (left != 0 && right > Long.MAX_VALUE / left) {
            throw new IndexException(Status.OVERFLOW, what + " overflows sixty-four bits");
        }
        return left * right;
    }

    /** Multiplies three non-negative extents, reporting OVERFLOW instead of wrapping. */
    static long product(long first, long second, long third, String what) {
        return product(product(first, second, what), third, what);
    }

    /** Adds two non-negative extents, reporting OVERFLOW instead of wrapping. */
    static long sum(long left, long right, String what) {
        nonNegative(left, what);
        nonNegative(right, what);
        if (left > Long.MAX_VALUE - right) {
            throw new IndexException(Status.OVERFLOW, what + " overflows sixty-four bits");
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
            throw new IndexException(Status.OVERFLOW, what + " does not fit in a Java array");
        }
        return (int) extent;
    }

    /** Narrows an extent to an index, reporting OVERFLOW when it does not fit. */
    static int toIndex(long extent, String what) {
        nonNegative(extent, what);
        if (extent > Integer.MAX_VALUE) {
            throw new IndexException(Status.OVERFLOW, what + " does not fit in a Java index");
        }
        return (int) extent;
    }

    private static void nonNegative(long extent, String what) {
        if (extent < 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, what + " is negative");
        }
    }
}
