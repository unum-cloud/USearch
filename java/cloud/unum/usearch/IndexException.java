package cloud.unum.usearch;

/**
 * One failed call: a status every caller can branch on, and the detail the status cannot carry.
 *
 * <p>
 * It is unchecked because nearly every verb can report one, and a checked exception on all of them
 * would turn each call site into a funnel rather than a branch.
 */
public final class IndexException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    private final Status status;

    /** Builds the failure a status and its detail describe. */
    public IndexException(Status status, String message) {
        super("usearch: " + status + ": " + message);
        this.status = status;
    }

    /**
     * Builds the failure a status code and its message describe - the shape the native layer
     * raises, which cannot name a Java enumerator itself.
     */
    static IndexException of(int code, String message) {
        return new IndexException(Status.of(code), message);
    }

    /** The status this failure carries. */
    public Status status() {
        return status;
    }
}
