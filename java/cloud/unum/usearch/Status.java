package cloud.unum.usearch;

/**
 * What went wrong, in a vocabulary shared by the engine and by this binding. Validation performed
 * here before a call crosses reports the status the engine would have reported for the same fault.
 */
public enum Status {

    /** The call completed and carries no message. */
    SUCCESS(0, "success"),
    /** An allocation or reservation could not be satisfied. */
    OUT_OF_MEMORY(1, "out_of_memory"),
    /** Dimensionality, row counts, or output capacity disagree. */
    SHAPE_MISMATCH(2, "shape_mismatch"),
    /** A stride or packing the engine cannot traverse. */
    UNSUPPORTED_LAYOUT(3, "unsupported_layout"),
    /** A scalar, metric, or residency tag this build does not know. */
    UNKNOWN_KIND(4, "unknown_kind"),
    /** A mutating verb reached a handle that opens its checkpoint in place. */
    READ_ONLY(5, "read_only"),
    /** A file could not be opened, read, written, or parsed. */
    IO_FAILURE(6, "io"),
    /** A device could not be selected, or its memory could not be claimed. */
    DEVICE_FAILURE(7, "device"),
    /** A caller-supplied representation could not be decoded. */
    MALFORMED_INPUT(8, "malformed_input"),
    /** The library reported neither success nor a failure. */
    INTERNAL(9, "internal"),
    /** A hard design ceiling was reached, like the 32-bit identity space. */
    OVERFLOW(10, "overflow"),
    /** The verb needs a prior successful reservation that never happened. */
    UNINITIALIZED(11, "uninitialized"),
    /** The request contradicts an already-frozen configuration. */
    CONFIG_MISMATCH(12, "config_mismatch"),
    /**
     * A synchronization contract was violated by a concurrent operation; discard any produced
     * output and retry in a quieter phase.
     */
    CONTENDED(13, "contended"),
    /**
     * A status a newer library reported and this build cannot name. An unknown status is never
     * success.
     */
    UNRECOGNIZED(-1, "unrecognized");

    private final int code;
    private final String stem;

    Status(int code, String stem) {
        this.code = code;
        this.stem = stem;
    }

    /** The C enumerator's numeric value. */
    public int code() {
        return code;
    }

    /** Names the status a code carries; a code this build cannot name becomes UNRECOGNIZED. */
    public static Status of(int code) {
        for (Status candidate : values()) {
            if (candidate.code == code && candidate != UNRECOGNIZED) {
                return candidate;
            }
        }
        return UNRECOGNIZED;
    }

    /** Renders the status as its C enumerator's stem. */
    @Override
    public String toString() {
        return stem;
    }
}
