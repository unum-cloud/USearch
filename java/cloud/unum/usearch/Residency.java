package cloud.unum.usearch;

/**
 * Where an index's payload lives and which engine traverses it. The kind selects which fields are
 * read, and what loading a checkpoint means for the handle.
 *
 * <p>
 * Instances are immutable and come from the three factories, so an arm's fields can never be set
 * on a residency that does not read them.
 */
public final class Residency {

    /** Which engine traverses the payload, and where the payload lives. */
    public enum Kind {
        /** A volatile-memory index traversed by CPU threads - the concurrent baseline. */
        RAM(0),
        /** An accelerator-memory index traversed by GPU warps on a chosen device. */
        VRAM(1),
        /**
         * A persisted index traversed through pluggable I/O engines, sized past RAM. External-memory
         * graphs are never modified in place, so the handle is read-only and every mutating verb
         * fails on it with READ_ONLY.
         */
        NVME(2);

        private final int code;

        Kind(int code) {
            this.code = code;
        }

        /** The C enumerator's numeric value. */
        public int code() {
            return code;
        }

        /** Names the kind a code carries. */
        public static Kind of(int code) {
            for (Kind candidate : values()) {
                if (candidate.code == code) {
                    return candidate;
                }
            }
            throw new IndexException(Status.UNKNOWN_KIND, "the residency names neither RAM, VRAM, nor NVMe");
        }
    }

    /** Whether the RAM arenas demand a provisioned huge-page pool. */
    public enum HugePages {
        /** Ordinary pages, whatever the kernel provides. */
        BASE,
        /** Fails loudly when the pool is missing, rather than degrading silently. */
        REQUIRED
    }

    /** Whether a VRAM index may spill past the device's memory. */
    public enum Addressing {
        /** Everything stays in device memory. */
        DEVICE_ONLY,
        /**
         * Cold shards spill into host memory through the pager, trading traversal latency for
         * datasets past the device.
         */
        UNIFIED
    }

    /** How an NVMe index reaches the block device. */
    public enum IoMode {
        /** The O_DIRECT-class page-cache bypass - the default and the point. */
        DIRECT,
        /** Ordinary buffered reads through the page cache. */
        PAGE_CACHED
    }

    /** Picks the first device with free memory. */
    public static final int FIRST_FREE_DEVICE = -1;

    private final Kind kind;
    private final long first;
    private final long second;
    private final long third;

    private Residency(Kind kind, long first, long second, long third) {
        this.kind = kind;
        this.first = first;
        this.second = second;
        this.third = third;
    }

    /**
     * A volatile-memory index. Zero threads takes 7/8 of the logical cores, since full occupancy
     * starves the ordered publication path.
     */
    public static Residency ram(long threads, HugePages hugePages) {
        if (threads < 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the thread count is negative");
        }
        return new Residency(Kind.RAM, threads, hugePages == HugePages.REQUIRED ? 1 : 0, 0);
    }

    /** A volatile-memory index with a derived thread count and ordinary pages. */
    public static Residency ram() {
        return ram(0, HugePages.BASE);
    }

    /** An accelerator-memory index on a CUDA device ordinal. */
    public static Residency vram(int device, Addressing addressing) {
        return new Residency(Kind.VRAM, device, addressing == Addressing.UNIFIED ? 1 : 0, 0);
    }

    /**
     * A persisted index. Zero sector bytes probes the device - 512 fits one 256-dimensional I8 row -
     * and a zero queue depth derives from the device's queue capabilities.
     */
    public static Residency nvme(int sectorBytes, int queueDepth, IoMode ioMode) {
        if (sectorBytes < 0 || queueDepth < 0) {
            throw new IndexException(Status.SHAPE_MISMATCH, "the NVMe geometry is negative");
        }
        return new Residency(Kind.NVME, sectorBytes, queueDepth, ioMode == IoMode.DIRECT ? 0 : 1);
    }

    /** Which engine traverses the payload. */
    public Kind kind() {
        return kind;
    }

    /** The RAM worker count; zero derives. */
    public long threads() {
        return expect(Kind.RAM) ? first : 0;
    }

    /** The RAM backing demand. */
    public HugePages hugePages() {
        return expect(Kind.RAM) && isSet(second) ? HugePages.REQUIRED : HugePages.BASE;
    }

    /** The VRAM accelerator ordinal. */
    public int device() {
        return expect(Kind.VRAM) ? (int) first : 0;
    }

    /** The VRAM spill policy. */
    public Addressing addressing() {
        return expect(Kind.VRAM) && isSet(second) ? Addressing.UNIFIED : Addressing.DEVICE_ONLY;
    }

    /** The NVMe I/O granularity in bytes; zero probes the device. */
    public int sectorBytes() {
        return expect(Kind.NVME) ? (int) first : 0;
    }

    /** The NVMe submission ring depth; zero derives from the device. */
    public int queueDepth() {
        return expect(Kind.NVME) ? (int) second : 0;
    }

    /** How the NVMe handle reaches the block device. */
    public IoMode ioMode() {
        return expect(Kind.NVME) && isSet(third) ? IoMode.PAGE_CACHED : IoMode.DIRECT;
    }

    private boolean expect(Kind wanted) {
        return kind == wanted;
    }

    private static boolean isSet(long word) {
        return word != 0;
    }

    /** Stamps the tag and its arm into the flat options layout. */
    void writeInto(long[] words) {
        words[OptionsWords.RESIDENCY] = kind.code();
        words[OptionsWords.RESIDENCY_FIRST] = first;
        words[OptionsWords.RESIDENCY_SECOND] = second;
        words[OptionsWords.RESIDENCY_THIRD] = third;
    }

    /** Reconstructs a residency from the tag and the arm it selects. */
    static Residency readFrom(long[] words) {
        Kind kind = Kind.of(Extents.toIndex(words[OptionsWords.RESIDENCY], "the residency code"));
        return new Residency(
                kind,
                words[OptionsWords.RESIDENCY_FIRST],
                words[OptionsWords.RESIDENCY_SECOND],
                words[OptionsWords.RESIDENCY_THIRD]);
    }

    /** Renders the residency as its kind and the arm it selects. */
    @Override
    public String toString() {
        switch (kind) {
            case RAM:
                return "ram(threads=" + threads() + ", " + hugePages() + ")";
            case VRAM:
                return "vram(device=" + device() + ", " + addressing() + ")";
            default:
                return "nvme(sectorBytes=" + sectorBytes() + ", queueDepth=" + queueDepth() + ", " + ioMode() + ")";
        }
    }

    /** Two residencies are equal when their kind and every word of their arm are. */
    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Residency)) {
            return false;
        }
        Residency peer = (Residency) other;
        return kind == peer.kind && first == peer.first && second == peer.second && third == peer.third;
    }

    /** Hashes the same words {@link #equals(Object)} compares, so the two agree. */
    @Override
    public int hashCode() {
        return java.util.Objects.hash(kind, first, second, third);
    }
}
