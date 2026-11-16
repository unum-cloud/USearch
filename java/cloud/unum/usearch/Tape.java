/**
 *  @file java/cloud/unum/usearch/Tape.java
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The ragged batch sequence and shape indexes take and answer, as Apache Arrow lays one out.
 */
package cloud.unum.usearch;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;

import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;

/**
 * A ragged batch: the elements of every entry end to end, and one more offset than entries around
 * them, entry {@code i} spanning elements {@code offset(i)} up to {@code offset(i + 1)}. An element
 * is a byte of a sequence, or a point of a shape, three scalars {@code x}, {@code y} and {@code z}
 * interleaved, so offsets count bytes or points, never scalars.
 *
 * <p>
 * An input built from Java arrays is copied into native memory for one call and never retained,
 * while one over a native segment is addressed in place. A tape {@link SequenceIndex#get} or
 * {@link ShapesIndex#get} answers lives in their arena, valid until its next call or its close.
 *
 * <p>
 * Mirrors {@code usearch_tape_view_t}, always with 64-bit offsets.
 */
public final class Tape {

    private final MemorySegment data;
    private final MemorySegment offsets;
    private final long count;
    private final Scalar scalar;
    private final int width;

    private Tape(MemorySegment data, MemorySegment offsets, long count, Scalar scalar, int width) {
        this.data = data;
        this.offsets = offsets;
        this.count = count;
        this.scalar = scalar;
        this.width = width;
    }

    /** Byte sequences, one entry apiece, for a {@link SequenceIndex}. */
    public static Tape ofBytes(byte[]... sequences) {
        long[] offsets = new long[sequences.length + 1];
        for (int entry = 0; entry != sequences.length; ++entry) {
            offsets[entry + 1] = Types.Extents.sum(offsets[entry], sequences[entry].length, "the tape");
        }
        byte[] joined = new byte[Types.Extents.toArrayLength(offsets[sequences.length], "the tape")];
        for (int entry = 0; entry != sequences.length; ++entry) {
            System.arraycopy(sequences[entry], 0, joined, (int) offsets[entry], sequences[entry].length);
        }
        return new Tape(MemorySegment.ofArray(joined), MemorySegment.ofArray(offsets), sequences.length, Scalar.U8, 1);
    }

    /** Texts as their UTF-8 bytes, one entry apiece, for a {@link SequenceIndex}. */
    public static Tape ofStrings(String... texts) {
        byte[][] encoded = new byte[texts.length][];
        for (int entry = 0; entry != texts.length; ++entry) {
            encoded[entry] = texts[entry].getBytes(StandardCharsets.UTF_8);
        }
        return ofBytes(encoded);
    }

    /**
     * Point clouds of F32 coordinates, {@code x}, {@code y} and {@code z} interleaved, one entry
     * apiece, for a {@link ShapesIndex}.
     *
     * @throws USearchException SHAPE_MISMATCH when a cloud does not hold three coordinates per point.
     */
    public static Tape ofPoints(float[]... shapes) {
        long[] offsets = new long[shapes.length + 1];
        for (int entry = 0; entry != shapes.length; ++entry) {
            if (shapes[entry].length % 3 != 0) {
                throw new USearchException(Status.SHAPE_MISMATCH, "a point holds three coordinates");
            }
            offsets[entry + 1] = offsets[entry] + shapes[entry].length / 3;
        }
        float[] joined = new float[Types.Extents.toArrayLength(offsets[shapes.length] * 3, "the tape")];
        for (int entry = 0; entry != shapes.length; ++entry) {
            System.arraycopy(shapes[entry], 0, joined, (int) offsets[entry] * 3, shapes[entry].length);
        }
        return new Tape(MemorySegment.ofArray(joined), MemorySegment.ofArray(offsets), shapes.length, Scalar.F32, 3);
    }

    /**
     * Elements already end to end, like an Arrow {@code LargeBinary} column or a mapped file, with
     * {@code width} scalars per element: one for sequences, three for shape points.
     *
     * @throws USearchException SHAPE_MISMATCH when the offsets are empty, decrease, or reach past
     *     {@code data}.
     */
    public static Tape of(MemorySegment data, long[] offsets, Scalar scalar, int width) {
        if (offsets.length == 0 || width <= 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "a tape holds one more offset than entries");
        }
        for (int entry = 1; entry != offsets.length; ++entry) {
            if (offsets[entry] < offsets[entry - 1]) {
                throw new USearchException(Status.SHAPE_MISMATCH, "the offsets decrease");
            }
        }
        long elementBytes = Types.Extents.product(scalar.valueBytes(), width, "the element");
        if (Types.Extents.product(offsets[offsets.length - 1], elementBytes, "the tape") > data.byteSize()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the offsets reach past the data");
        }
        return new Tape(data, MemorySegment.ofArray(offsets), offsets.length - 1, scalar, width);
    }

    /** Lifts the {@code usearch_tape_view_t} of {@code written} entries a call wrote into its arena. */
    static Tape lift(Call call, MemorySegment view, long written, int width) {
        Scalar scalar = Scalar.of((int) SCALAR.get(view, 0L));
        if (written == 0) {
            return new Tape(MemorySegment.NULL, MemorySegment.ofArray(new long[1]), 0, scalar, width);
        }
        MemorySegment offsets = call.reach((MemorySegment) OFFSETS.get(view, 0L), (written + 1) * Long.BYTES);
        long elements = offsets.getAtIndex(ValueLayout.JAVA_LONG, written);
        long bytes = Types.Extents.product(elements, (long) scalar.valueBytes() * width, "the tape");
        return new Tape(call.reach((MemorySegment) DATA.get(view, 0L), bytes), offsets, written, scalar, width);
    }

    /** Fills the {@code usearch_tape_view_t} embedded at {@code at} of the call's structure. */
    void fill(Call call, long at) {
        long bytes = offset(count) * scalar.valueBytes() * width;
        DATA.set(call.arguments, at, call.input(data.asSlice(0, bytes)));
        LAYOUT.set(call.arguments, at, 1);
        OFFSETS.set(call.arguments, at, call.input(offsets.asSlice(0, (count + 1) * Long.BYTES)));
        SCALAR.set(call.arguments, at, scalar.code());
    }

    /** How many entries the tape holds. */
    public long count() {
        return count;
    }

    /** The scalar type every element's values are in. */
    public Scalar scalar() {
        return scalar;
    }

    /** Scalars per element: one for a sequence's bytes, three for a shape's points. */
    public int width() {
        return width;
    }

    /**
     * Where an entry starts, in elements; {@code offset(count())} closes the last one.
     *
     * @throws USearchException SHAPE_MISMATCH past the last offset.
     */
    public long offset(long entry) {
        if (entry < 0 || entry > count) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the tape holds " + count + " entries");
        }
        return offsets.getAtIndex(ValueLayout.JAVA_LONG, entry);
    }

    /**
     * One entry's elements in place.
     *
     * @throws USearchException SHAPE_MISMATCH when the entry is past the tape.
     */
    public MemorySegment segment(long entry) {
        if (entry < 0 || entry >= count) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the tape holds " + count + " entries");
        }
        long elementBytes = (long) scalar.valueBytes() * width;
        long begin = offset(entry) * elementBytes;
        return data.asSlice(begin, offset(entry + 1) * elementBytes - begin);
    }

    /** One entry's bytes, copied out. */
    public byte[] bytes(long entry) {
        return segment(entry).toArray(ValueLayout.JAVA_BYTE);
    }

    /** One entry's bytes, decoded as UTF-8. */
    public String string(long entry) {
        return new String(bytes(entry), StandardCharsets.UTF_8);
    }

    /**
     * One entry's F32 coordinates, {@code x}, {@code y} and {@code z} interleaved, copied out.
     *
     * @throws USearchException INVALID_ARGUMENT on a tape of another scalar, read through
     *     {@link #segment(long)} instead.
     */
    public float[] points(long entry) {
        if (scalar != Scalar.F32) {
            throw new USearchException(Status.INVALID_ARGUMENT, "the tape holds " + scalar + " values");
        }
        return segment(entry).toArray(ValueLayout.JAVA_FLOAT);
    }

    /** Renders the tape's shape, not its contents. */
    @Override
    public String toString() {
        return "Tape(count=" + count + ", " + scalar + ", width=" + width + ")";
    }

    private static final VarHandle DATA = Library.field(BasicCalls.TAPE_VIEW, "data");
    private static final VarHandle LAYOUT = Library.field(BasicCalls.TAPE_VIEW, "layout");
    private static final VarHandle OFFSETS = Library.field(BasicCalls.TAPE_VIEW, "offsets");
    private static final VarHandle SCALAR = Library.field(BasicCalls.TAPE_VIEW, "scalar");
}
