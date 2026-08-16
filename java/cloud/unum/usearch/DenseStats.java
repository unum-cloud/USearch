package cloud.unum.usearch;

/**
 * The traversal telemetry one call reports. Every call overwrites it rather than accumulating, so a
 * sweep over many batches sums them itself with {@link #accumulate(DenseStats)}.
 */
public final class DenseStats {

    private long visitedMembers;
    private long computedDistances;
    private long restarts;

    /** A zeroed set of counters, ready to be passed to a call. */
    public DenseStats() {}

    /** Graph members the call visited. */
    public long visitedMembers() {
        return visitedMembers;
    }

    /** Distances the call computed, diversity re-selections included. */
    public long computedDistances() {
        return computedDistances;
    }

    /**
     * Starvation re-seedings - a graph-connectivity health signal rather than a cost counter.
     */
    public long restarts() {
        return restarts;
    }

    /** Folds another call's telemetry into this one. */
    public void accumulate(DenseStats other) {
        visitedMembers += other.visitedMembers;
        computedDistances += other.computedDistances;
        restarts += other.restarts;
    }

    /** Adopts the three counters the native layer wrote. */
    void adopt(long[] words) {
        visitedMembers = words[0];
        computedDistances = words[1];
        restarts = words[2];
    }

    /** Renders the telemetry as the three counters it carries. */
    @Override
    public String toString() {
        return "DenseStats(visitedMembers=" + visitedMembers
                + ", computedDistances=" + computedDistances
                + ", restarts=" + restarts + ")";
    }
}
