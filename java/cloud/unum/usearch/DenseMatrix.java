package cloud.unum.usearch;

import java.nio.ByteBuffer;

/**
 * A caller-owned slab viewed as rows of equal width: a base, a row count, a logical width, and a
 * byte pitch between consecutive row starts. It is the shape every batch argument and every batch
 * output takes, so one type serves {@link DenseIndex#add} and {@link DenseIndex#get} alike.
 *
 * <p>
 * The slab is either a primitive array or a direct {@link ByteBuffer}. An array is pinned for the
 * duration of one call and never retained; a direct buffer is addressed in place, which is what
 * lets a memory-mapped dataset reach the index with no reshape and no copy. A heap buffer has no
 * stable address and is refused rather than silently copied.
 *
 * <p>
 * A matrix borrows and never owns, so the slab must outlive every call it crosses.
 */
public final class DenseMatrix {

    private final Object storage;
    private final long byteOffset;
    private final long rows;
    private final long dimensions;
    private final long stride;
    private final ScalarKind kind;

    private DenseMatrix(Object storage, long byteOffset, long rows, long dimensions, long stride, ScalarKind kind) {
        this.storage = storage;
        this.byteOffset = byteOffset;
        this.rows = rows;
        this.dimensions = dimensions;
        this.stride = stride;
        this.kind = kind;
    }

    /** Views a packed F64 slab as rows of the given width. */
    public static DenseMatrix of(double[] data, long dimensions) {
        return packed(data, dimensions, ScalarKind.F64);
    }

    /** Views a packed F32 slab as rows of the given width. */
    public static DenseMatrix of(float[] data, long dimensions) {
        return packed(data, dimensions, ScalarKind.F32);
    }

    /** Views a packed slab of sixteen-bit values as rows of the given width. */
    public static DenseMatrix of(short[] data, long dimensions, ScalarKind kind) {
        return packed(data, dimensions, kind);
    }

    /** Views a packed slab of single-byte values as rows of the given width. */
    public static DenseMatrix of(byte[] data, long dimensions, ScalarKind kind) {
        return packed(data, dimensions, kind);
    }

    /**
     * Views a slab whose pitch exceeds its row width - the padded layout a memory-mapped dataset
     * arrives in. The storage is a primitive array or a direct {@link ByteBuffer}.
     */
    public static DenseMatrix strided(Object storage, long rows, long dimensions, long stride, ScalarKind kind) {
        long rowBytes = rowBytesOf(dimensions, kind);
        if (stride < rowBytes) {
            throw new IndexException(
                    Status.UNSUPPORTED_LAYOUT, "a stride of " + stride + " bytes cannot hold a row of " + rowBytes);
        }
        long byteOffset = offsetOf(storage);
        long span = rows == 0 ? 0 : Extents.sum(Extents.product(rows - 1, stride, "the matrix span"), rowBytes,
                "the matrix span");
        if (span > capacityOf(storage, kind)) {
            throw new IndexException(
                    Status.SHAPE_MISMATCH, rows + " rows of stride " + stride + " overrun the slab");
        }
        return new DenseMatrix(storage, byteOffset, rows, dimensions, stride, kind);
    }

    /** Allocates a packed slab of the natural Java array type for the kind. */
    public static DenseMatrix allocate(long rows, long dimensions, ScalarKind kind) {
        long rowBytes = rowBytesOf(dimensions, kind);
        long values = Extents.product(rows, rowBytes / kind.valueBytes(), "the matrix");
        int length = Extents.toArrayLength(values, "the matrix");
        Object storage;
        switch (kind.valueBytes()) {
            case 8:
                storage = new double[length];
                break;
            case 4:
                storage = new float[length];
                break;
            case 2:
                storage = new short[length];
                break;
            default:
                storage = new byte[length];
                break;
        }
        return new DenseMatrix(storage, 0, rows, dimensions, rowBytes, kind);
    }

    private static DenseMatrix packed(Object data, long dimensions, ScalarKind kind) {
        long rowBytes = rowBytesOf(dimensions, kind);
        long capacity = capacityOf(data, kind);
        if (capacity % rowBytes != 0) {
            throw new IndexException(
                    Status.SHAPE_MISMATCH, capacity + " bytes do not divide into rows of " + rowBytes);
        }
        return new DenseMatrix(data, offsetOf(data), capacity / rowBytes, dimensions, rowBytes, kind);
    }

    private static long rowBytesOf(long dimensions, ScalarKind kind) {
        if (dimensions <= 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, "a matrix row cannot be zero dimensions wide");
        }
        if (kind == ScalarKind.UNKNOWN) {
            throw new IndexException(Status.UNKNOWN_KIND, "the matrix names no scalar kind");
        }
        return kind.rowBytes(dimensions);
    }

    /** The addressable bytes a slab holds, refusing a slab whose element width the kind rejects. */
    private static long capacityOf(Object storage, ScalarKind kind) {
        if (storage instanceof double[]) {
            return elementBytes(kind, 8, ((double[]) storage).length);
        }
        if (storage instanceof float[]) {
            return elementBytes(kind, 4, ((float[]) storage).length);
        }
        if (storage instanceof short[]) {
            return elementBytes(kind, 2, ((short[]) storage).length);
        }
        if (storage instanceof byte[]) {
            return elementBytes(kind, 1, ((byte[]) storage).length);
        }
        if (storage instanceof ByteBuffer) {
            ByteBuffer buffer = (ByteBuffer) storage;
            if (!buffer.isDirect()) {
                throw new IndexException(
                        Status.UNSUPPORTED_LAYOUT, "a heap buffer has no stable address; allocate a direct one");
            }
            return buffer.remaining();
        }
        throw new IndexException(
                Status.UNSUPPORTED_LAYOUT, "a matrix borrows a primitive array or a direct ByteBuffer");
    }

    private static long elementBytes(ScalarKind kind, int width, int length) {
        if (kind.valueBytes() != width) {
            throw new IndexException(
                    Status.UNSUPPORTED_LAYOUT, "a " + kind + " value is not " + width + " bytes wide");
        }
        return (long) length * width;
    }

    private static long offsetOf(Object storage) {
        return storage instanceof ByteBuffer ? ((ByteBuffer) storage).position() : 0;
    }

    /** How many rows the view spans. */
    public long rows() {
        return rows;
    }

    /** The logical width of one row. */
    public long dimensions() {
        return dimensions;
    }

    /** The byte pitch between consecutive row starts. */
    public long stride() {
        return stride;
    }

    /** The scalar type the rows hold. */
    public ScalarKind kind() {
        return kind;
    }

    /** The slab the view borrows - a primitive array or a direct ByteBuffer. */
    public Object storage() {
        return storage;
    }

    /** The byte offset the first row starts at; non-zero only for a positioned buffer. */
    long byteOffset() {
        return byteOffset;
    }

    /** The slab as F64 values. */
    public double[] doubles() {
        return (double[]) storage;
    }

    /** The slab as F32 values. */
    public float[] floats() {
        return (float[]) storage;
    }

    /** The slab as sixteen-bit values. */
    public short[] shorts() {
        return (short[]) storage;
    }

    /** The slab as single-byte values. */
    public byte[] bytes() {
        return (byte[]) storage;
    }

    /**
     * Checks the view against a frozen geometry before it crosses the boundary. A width of zero
     * skips the geometry check, which is what an index whose geometry is still unfrozen reports.
     */
    void validate(long dimensions) {
        if (dimensions != 0 && this.dimensions != dimensions) {
            throw new IndexException(
                    Status.SHAPE_MISMATCH, "the matrix rows are not as wide as the index geometry");
        }
    }

    /** Renders the view as the geometry it describes. */
    @Override
    public String toString() {
        return "DenseMatrix(rows=" + rows + ", dimensions=" + dimensions + ", stride=" + stride + ", " + kind + ")";
    }
}
