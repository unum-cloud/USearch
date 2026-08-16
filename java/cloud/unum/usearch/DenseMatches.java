package cloud.unum.usearch;

/**
 * Where a batched search writes its results - three slabs sized together, holding a
 * queries-by-wanted matrix of keys, the distances positionally paired with them, and one match
 * count per query.
 *
 * <p>
 * The keys are one flat byte slab rather than an array of {@link Key} objects, so a batch of a
 * million queries costs no allocations beyond the three slabs, and {@link #key(int, int)}
 * materializes a key only for the slots a caller actually reads. Slots past a query's count hold
 * memory the engine never wrote, so every accessor here refuses them.
 *
 * <p>
 * The wanted count is the "k" in kANN and the row width of both matrices, so it is fixed at
 * construction and cannot be restated afterwards.
 */
public final class DenseMatches {

    private final byte[] keys;
    private final double[] distances;
    private final long[] counts;
    private final int queries;
    private final int wanted;

    /** Sizes the three slabs for a batch of queries, each reporting up to wanted matches. */
    public DenseMatches(long queries, long wanted) {
        if (wanted <= 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the outputs reserve no room for a single match");
        }
        this.queries = Extents.toIndex(queries, "the query count");
        this.wanted = Extents.toIndex(wanted, "the wanted count");
        long slots = Extents.product(queries, wanted, "the match matrix");
        this.keys = new byte[Extents.toArrayLength(
                Extents.product(slots, Key.BYTES, "the key matrix"), "the key matrix")];
        this.distances = new double[Extents.toArrayLength(slots, "the distance matrix")];
        this.counts = new long[this.queries];
    }

    /** The upper bound on matches per query, and the row width of both matrices. */
    public int wanted() {
        return wanted;
    }

    /** How many queries the slabs were sized for. */
    public int queries() {
        return queries;
    }

    /**
     * How many matches of a query are readable - the reported count clamped to the room reserved
     * for it.
     */
    public int count(int query) {
        long reported = reportedCount(query);
        return (int) Math.min(reported, wanted);
    }

    /**
     * The count the engine reported, which may exceed {@link #wanted()} when the row was truncated.
     * Comparing the two is how a caller learns to raise the wanted count and search again.
     */
    public long reportedCount(int query) {
        if (query < 0 || query >= queries) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the query is outside the batch");
        }
        return counts[query];
    }

    /** Whether a query's matches were cut short by the room reserved for them. */
    public boolean truncated(int query) {
        return reportedCount(query) > wanted;
    }

    /** The key of one match, in nearest-first order. */
    public Key key(int query, int rank) {
        return Key.ofBytes(keys, slotOf(query, rank) * Key.BYTES);
    }

    /** The distance of one match, in the metric of the index that produced it. */
    public double distance(int query, int rank) {
        return distances[slotOf(query, rank)];
    }

    private int slotOf(int query, int rank) {
        int readable = count(query);
        if (rank < 0 || rank >= readable) {
            throw new IndexException(
                    Status.SHAPE_MISMATCH, "the query reported " + readable + " readable matches");
        }
        return query * wanted + rank;
    }

    /** The flat key slab the native layer writes into. */
    byte[] keySlab() {
        return keys;
    }

    /** The distance slab the native layer writes into. */
    double[] distanceSlab() {
        return distances;
    }

    /** The per-query count slab the native layer writes into. */
    long[] countSlab() {
        return counts;
    }

    /** Renders the batch shape, not its contents. */
    @Override
    public String toString() {
        return "DenseMatches(queries=" + queries + ", wanted=" + wanted + ")";
    }
}
