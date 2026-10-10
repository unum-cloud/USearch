package cloud.unum.usearch;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.Function;
import java.util.function.Supplier;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.DenseIndex.DenseAttendConfig;
import cloud.unum.usearch.DenseIndex.DenseCursor;
import cloud.unum.usearch.DenseIndex.DenseIndexDescription;
import cloud.unum.usearch.DenseIndex.DenseInsertConfig;
import cloud.unum.usearch.DenseIndex.DenseMaxsimConfig;
import cloud.unum.usearch.DenseIndex.DenseMultiplyConfig;
import cloud.unum.usearch.DenseIndex.DenseReachability;
import cloud.unum.usearch.DenseIndex.DenseRefineConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchStream;
import cloud.unum.usearch.DenseIndex.DenseSoftmaxState;
import cloud.unum.usearch.DenseIndex.Products;
import cloud.unum.usearch.DenseIndex.ValueLookup;
import cloud.unum.usearch.DenseIndex.Vectors;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * The blocking verbs of a dense index, written once for the two receivers that block: the
 * {@link DenseIndex} itself, which runs every call on the caller's thread, and the
 * {@link DenseIndex.On} that {@link DenseIndex#on} makes, which runs it on an executor. The
 * {@link DenseIndex.DeferredOn} that {@link DenseIndex#deferred} makes offers the same verbs, each
 * answering a {@link Types.Deferred}.
 *
 * <p>
 * Every verb that takes knobs beyond its inputs has a bare form and one taking its
 * {@code Dense*Config}, named after the C structure it fills.
 *
 * <p>
 * Binding-only; every verb names the C symbol it calls.
 */
public abstract sealed class DenseVerbs permits DenseIndex, DenseIndex.On {

    DenseVerbs() {}

    /** The index every call names. */
    abstract DenseIndex index();

    /** The executor every call runs on, or {@code null} for the calling thread. */
    abstract @Nullable Executor executor();

    private <Result> Result block(Request<Result> request) {
        return Call.block(request.verb(), index().handle, executor(), request.fill());
    }

    /**
     * Claims capacity for a number of incoming vectors ahead of insertion. The capacity is the
     * desired total, including the current size. Calls {@code usearch_dense_reserve}.
     *
     * @throws USearchException OUT_OF_MEMORY when the capacity cannot be claimed, READ_ONLY on a
     *     view, and OVERFLOW past the identity space's ceiling.
     */
    public void reserve(long capacity) {
        block(reserving(capacity));
    }

    /**
     * Erases all entries, keeping the reserved capacity and the frozen geometry. No other call may
     * overlap it. Calls {@code usearch_dense_clear}.
     *
     * @throws USearchException READ_ONLY on a view.
     */
    public void clear() {
        block(clearing());
    }

    /** Reports the frozen metadata and the current counters in one call. Calls {@code usearch_dense_describe}. */
    public DenseIndexDescription describe() {
        return block(describing());
    }

    /** How many entries are stored under a key; zero means the index holds none. Calls {@code usearch_dense_count}. */
    public long count(Key key) {
        return block(counting(key));
    }

    /** Whether the index holds at least one entry under a key. Binding-only sugar over {@link #count}. */
    public boolean contains(Key key) {
        return count(key) != 0;
    }

    /**
     * Retrieves every vector stored under a key into a fresh arena the result owns.
     *
     * @throws USearchException the statuses {@link #get(Key, Scalar, Arena)} names.
     */
    public Vectors get(Key key, Scalar scalar) {
        return block(getting(key, scalar, null));
    }

    /**
     * Retrieves every vector stored under a key, one row per entry sharing it, in insertion order,
     * packed at the natural width of the kind asked for.
     *
     * <p>
     * Only the call knows how many entries share the key, so the rows land in an arena: the one
     * passed, which the result stays valid until that arena is used again, or a fresh one the result
     * owns. Calls {@code usearch_dense_get}.
     *
     * @return the rows read in place, none when the key holds nothing.
     * @throws USearchException UNKNOWN_KIND on a scalar the engine cannot convert into, and
     *     CONTENDED when the arena is serving another call.
     */
    public Vectors get(Key key, Scalar scalar, @Nullable Arena arena) {
        return block(getting(key, scalar, arena));
    }

    /**
     * Inserts a matrix of vectors under their keys, with every knob derived.
     *
     * @throws USearchException the statuses {@link #insert(Key[], RowsView, DenseInsertConfig)} names.
     */
    public void insert(Key[] keys, RowsView vectors) {
        block(inserting(keys, vectors, new DenseInsertConfig()));
    }

    /**
     * Inserts a matrix of vectors under their keys - the native batch operation, where a single
     * vector is a one-row matrix. Overlaps safely with concurrent searches on the same handle.
     *
     * <p>
     * The input scalar type is independent of the stored quantization; the library converts on
     * ingest. A batch is all-or-nothing: a returning call published every row, a failing one
     * published none. An accelerator fault severe enough to poison the device context is the sole
     * exception, reporting DEVICE and latching the index unusable rather than partially applied.
     * Calls {@code usearch_dense_insert}.
     *
     * @throws USearchException SHAPE_MISMATCH when the key count and the row count disagree or the
     *     rows are not as wide as the frozen geometry, UNINITIALIZED while the geometry is unfrozen,
     *     UNKNOWN_KIND on a scalar the engine cannot convert, READ_ONLY on a view, and
     *     OUT_OF_MEMORY when the batch cannot be accommodated.
     */
    public void insert(Key[] keys, RowsView vectors, DenseInsertConfig config) {
        block(inserting(keys, vectors, config));
    }

    /**
     * Re-selects the neighbors of every stored entry, with every knob derived.
     *
     * @throws USearchException the statuses {@link #refine(DenseRefineConfig)} names.
     */
    public long refine() {
        return block(refining(new DenseRefineConfig()));
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, improving neighbors
     * chosen when the graph around them was still incomplete.
     *
     * <p>
     * Searches may run throughout on a RAM handle, since re-selection publishes one node at a time
     * and never removes one; insertions may not. Re-selecting a settled node changes nothing, so a
     * caller loops until the changed-node count falls below whatever rate it considers settled.
     * Calls {@code usearch_dense_refine}.
     *
     * @return the nodes whose neighbors the pass re-selected; zero means the graph has settled.
     * @throws USearchException READ_ONLY on a view.
     */
    public long refine(DenseRefineConfig config) {
        return block(refining(config));
    }

    /**
     * Audits how much of the graph a search reaches: forward reachability from as many landmarks as
     * one search draws, swept level by level. Read-only, so concurrent insertions merely leave fresh
     * entries unaudited. Calls {@code usearch_dense_health}.
     */
    public DenseReachability health() {
        return block(auditing());
    }

    /**
     * Resolves a matrix of queries into up to {@code wanted} nearest matches apiece, in outputs the
     * call allocates.
     *
     * @throws USearchException the statuses {@link #search(RowsView, DenseSearchConfig)} names.
     */
    public Matches search(RowsView queries, long wanted) {
        return block(searching(queries, new DenseSearchConfig(wanted)));
    }

    /**
     * Resolves a matrix of queries into the outputs the config names - the allocation-free batch
     * path when it lends {@link Matches}, parallelized internally. Never writes to the graph, so any
     * number of these may overlap with concurrent insertions. Calls {@code usearch_dense_search}.
     *
     * @return the matches, landed where the config says.
     * @throws USearchException SHAPE_MISMATCH when the rows are not as wide as the frozen geometry or
     *     the outputs hold fewer rows than there are queries, and UNKNOWN_KIND on a scalar the
     *     engine cannot convert.
     */
    public Matches search(RowsView queries, DenseSearchConfig config) {
        return block(searching(queries, config));
    }

    /**
     * Late-interaction scoring keeping up to {@code wanted} keys per query, in outputs the call
     * allocates.
     *
     * @throws USearchException the statuses {@link #maxsim(RowsView, long, DenseMaxsimConfig)} names.
     */
    public Matches maxsim(RowsView queries, long rowsPerQuery, long wanted) {
        return block(scoring(queries, rowsPerQuery, new DenseMaxsimConfig(wanted)));
    }

    /**
     * Late-interaction scoring of multi-row queries against multi-row entries: every query row takes
     * its best match within a key, and those maxima sum to that key's score. A document is one key
     * holding its token rows, so the wanted count is the "k" over keys rather than over rows.
     *
     * <p>
     * Every query carries the same number of rows, so a padded token batch applies as is, and the
     * query count is the row count divided by that. Approximate in the same way a search is: a key
     * can score below its exact value but never above it. Runs on the inner-product kinds alone.
     * Calls {@code usearch_dense_maxsim}.
     *
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind, and
     *     SHAPE_MISMATCH when the token rows do not divide into whole queries, the rows are not as
     *     wide as the frozen geometry, or the outputs hold fewer rows than there are queries.
     */
    public Matches maxsim(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config) {
        return block(scoring(queries, rowsPerQuery, config));
    }

    /**
     * Approximate sparse product keeping every pair, into a fresh arena the result owns.
     *
     * @throws USearchException the statuses {@link #multiply(RowsView, DenseMultiplyConfig)} names.
     */
    public Products multiply(RowsView left) {
        return block(multiplying(left, new DenseMultiplyConfig()));
    }

    /**
     * Approximate sparse product of a matrix against the indexed rows - the linear-algebra sibling
     * of {@link #search}, reporting every pair the traversal scored rather than only the nearest.
     * Pairs arrive in traversal order, and the pairs the walk never reaches are the low and negative
     * products. Runs on the inner-product kinds alone.
     *
     * <p>
     * Only the call knows how many pairs it scores, so they land in an arena: the one the config
     * passes, which the result stays valid until that arena is used again, or a fresh one the result
     * owns. Calls {@code usearch_dense_multiply}.
     *
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind,
     *     SHAPE_MISMATCH when the rows are not as wide as the frozen geometry, and CONTENDED when the
     *     arena is serving another call.
     */
    public Products multiply(RowsView left, DenseMultiplyConfig config) {
        return block(multiplying(left, config));
    }

    /**
     * Softmax attention landing in {@code outputs}, with every other knob derived.
     *
     * @throws USearchException the statuses {@link #attend(RowsView, ValueLookup, DenseAttendConfig)}
     *     names.
     */
    public RowsView attend(RowsView queries, ValueLookup valueLookup, RowsView outputs) {
        return block(attending(queries, valueLookup, new DenseAttendConfig(outputs)));
    }

    /**
     * Softmax attention of every query row over every indexed key its walk scores, folded as Flash
     * Attention folds key tiles, with no top-k cut, landing in the config's outputs already divided
     * by the partition. Runs on the inner-product kinds alone, since a softmax needs logits.
     *
     * <p>
     * The index holds one layer's keys for one key-value head and no values, so the lookup resolves
     * every scored key to its value row. Calls {@code usearch_dense_attend}.
     *
     * @return the outputs the config names.
     * @throws USearchException SHAPE_MISMATCH when a scored key resolves to no row, the head count
     *     does not divide the rows, or the outputs are too few or of another width than the keys;
     *     CONFIG_MISMATCH on a graph frozen under a distance kind; and whatever the lookup raised,
     *     which outranks the status it turned into.
     */
    public RowsView attend(RowsView queries, ValueLookup valueLookup, DenseAttendConfig config) {
        return block(attending(queries, valueLookup, config));
    }

    /**
     * Absorbs every donor into this index - cheaper than re-inserting their vectors one by one,
     * linking across donors via landmark stitching. The donors are consumed whether or not the merge
     * succeeds, and closed from the call on. Calls {@code usearch_dense_merge}.
     *
     * @throws USearchException UNSUPPORTED where no residency absorbs donors yet, and INVALID_ARGUMENT
     *     when the index is among its own donors.
     */
    public void merge(DenseIndex... donors) {
        block(merging(donors));
    }

    /**
     * Writes the index under a path, in the layout its extension picks - one {@code .safegraphs}
     * monolith, or a {@code .safegraphs.json} manifest beside its shards. Demands a mutation-free
     * window; concurrent readers keep searching throughout. Calls {@code usearch_dense_save}.
     *
     * @throws USearchException INVALID_ARGUMENT on an empty path, and IO when the destination cannot
     *     be written - which leaves a file without its header, for the caller to remove.
     */
    public void save(String path) {
        block(saving(path));
    }

    /**
     * Writes the index as one monolith into a fresh arena the result owns.
     *
     * @throws USearchException the statuses {@link #saveBuffer(Arena)} names.
     */
    public Monolith saveBuffer() {
        return block(serializing(null));
    }

    /**
     * Writes the index as one monolith - every shard, rows in placement order - the same bytes
     * {@link #save} writes to a {@code .safegraphs} path, into the arena passed, which the result
     * stays valid until that arena is used again, or into a fresh one the result owns. Calls
     * {@code usearch_dense_save_buffer}.
     *
     * @throws USearchException CONTENDED when the arena is serving another call.
     */
    public Monolith saveBuffer(@Nullable Arena arena) {
        return block(serializing(arena));
    }

    /**
     * Opens a row scan over the current contents, pulled on this receiver's executor. Each shard's
     * publication watermark is read as the walk enters it, so a row once passed is never revisited.
     * Calls {@code usearch_dense_cursor_init}.
     */
    public DenseCursor cursor() {
        DenseIndex index = index();
        return Call.block(DenseIndex.CURSOR_INIT, index.handle, null, call -> {
            MemorySegment opened = call.output("cursor", Library.POINTER);
            return () -> new DenseCursor(opened.get(Library.POINTER, 0), index, executor());
        });
    }

    /**
     * Opens a resumable search for one query row, with the candidate-pool breadth derived.
     *
     * @throws USearchException the statuses {@link #searchStream(RowsView, long)} names.
     */
    public DenseSearchStream searchStream(RowsView query) {
        return searchStream(query, 0);
    }

    /**
     * Opens a resumable search for one query row, pulled on this receiver's executor: best-first
     * traversal is naturally pausable, so callers that post-filter aggressively pull until satisfied
     * instead of guessing the wanted count upfront. Calls {@code usearch_dense_search_stream_init}.
     *
     * @throws USearchException SHAPE_MISMATCH when the query is not exactly one row, and UNSUPPORTED
     *     where no residency implements the stream yet.
     */
    public DenseSearchStream searchStream(RowsView query, long expansionFactor) {
        if (query.rows() != 1) {
            throw new USearchException(Status.SHAPE_MISMATCH, "a stream walks exactly one query row");
        }
        DenseIndex index = index();
        return Call.block(DenseIndex.SEARCH_STREAM_INIT, index.handle, null, call -> {
            MemorySegment opened = call.output("stream", Library.POINTER);
            call.rows("query", query);
            call.size("expansion_factor", nonNegative(expansionFactor));
            return () -> new DenseSearchStream(opened.get(Library.POINTER, 0), index, executor());
        });
    }

    /** One verb's argument structure and how a call fills it, whichever completion runs it. */
    record Request<Result>(Verb verb, Function<Call, Supplier<Result>> fill) {}

    /** What a verb answering nothing answers once final. */
    private static final Supplier<Void> NOTHING = () -> null;

    static Request<Void> reserving(long capacity) {
        return new Request<>(DenseIndex.RESERVE, call -> {
            call.size("capacity", capacity);
            return NOTHING;
        });
    }

    static Request<Void> clearing() {
        return new Request<>(DenseIndex.CLEAR, call -> NOTHING);
    }

    static Request<DenseIndexDescription> describing() {
        return new Request<>(DenseIndex.DESCRIBE, call -> {
            MemorySegment description = call.output("description", DenseIndex.DESCRIPTION);
            return () -> DenseIndexDescription.lift(description);
        });
    }

    static Request<Long> counting(Key key) {
        return new Request<>(DenseIndex.COUNT, call -> {
            call.key("key", key);
            MemorySegment entries = call.output("entries", Library.SIZE);
            return () -> entries.get(Library.SIZE, 0);
        });
    }

    static Request<Vectors> getting(Key key, Scalar scalar, @Nullable Arena borrowed) {
        return new Request<>(DenseIndex.GET, call -> {
            Arena arena = borrowed != null ? borrowed : new Arena();
            call.key("key", key);
            call.enumeration("scalar", scalar.code());
            MemorySegment vectors = call.output("vectors", Library.POINTER);
            MemorySegment written = call.output("written", Library.SIZE);
            MemorySegment stride = call.output("stride", Library.SIZE);
            MemorySegment columns = call.output("columns", Library.SIZE);
            java.lang.foreign.Arena scope = call.arena("arena", arena);
            return () -> {
                long rows = written.get(Library.SIZE, 0);
                long pitch = stride.get(Library.SIZE, 0);
                return new Vectors(
                        borrowed != null ? null : arena,
                        arena.reach(scope, vectors.get(Library.POINTER, 0), rows * pitch),
                        rows,
                        columns.get(Library.SIZE, 0),
                        pitch,
                        scalar);
            };
        });
    }

    static Request<Void> inserting(Key[] keys, RowsView vectors, DenseInsertConfig config) {
        // The call has one count, so only this side sees the keys and the rows disagree.
        if (keys.length != vectors.rows()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the key count does not match the vector row count");
        }
        return new Request<>(DenseIndex.INSERT, call -> {
            call.size("count", keys.length);
            call.keys("keys", keys);
            call.rows("vectors", vectors);
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return NOTHING;
        });
    }

    static Request<Long> refining(DenseRefineConfig config) {
        return new Request<>(DenseIndex.REFINE, call -> {
            MemorySegment changed = call.output("changed", Library.SIZE);
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> changed.get(Library.SIZE, 0);
        });
    }

    static Request<DenseReachability> auditing() {
        return new Request<>(DenseIndex.HEALTH, call -> {
            MemorySegment reachability = call.output("reachability", DenseIndex.REACHABILITY);
            return () -> DenseReachability.lift(reachability);
        });
    }

    static Request<Matches> searching(RowsView queries, DenseSearchConfig config) {
        Matches matches = config.landing(queries.rows());
        return new Request<>(DenseIndex.SEARCH, call -> {
            call.size("count", queries.rows());
            call.rows("queries", queries);
            matches.land(call, "matches");
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> matches;
        });
    }

    static Request<Matches> scoring(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config) {
        if (rowsPerQuery <= 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "a query cannot carry zero rows");
        }
        if (queries.rows() % rowsPerQuery != 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the token rows do not divide into whole queries");
        }
        long count = queries.rows() / rowsPerQuery;
        Matches matches = config.landing(count);
        return new Request<>(DenseIndex.MAXSIM, call -> {
            call.size("count", count);
            call.rows("queries", queries);
            call.size("rows_per_query", rowsPerQuery);
            matches.land(call, "matches");
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> matches;
        });
    }

    static Request<Products> multiplying(RowsView left, DenseMultiplyConfig config) {
        int rows = Types.Extents.toArrayLength(left.rows(), "the left rows");
        return new Request<>(DenseIndex.MULTIPLY, call -> {
            Arena arena = config.arena != null ? config.arena : new Arena();
            call.size("count", rows);
            call.rows("left", left);
            MemorySegment offsets = call.memory.allocate(Library.SIZE, rows + 1L);
            call.pointer("offsets", offsets);
            MemorySegment keys = call.output("keys", Library.POINTER);
            MemorySegment scores = call.output("scores", Library.POINTER);
            java.lang.foreign.Arena scope = call.arena("arena", arena);
            MemorySegment written = call.output("written", Library.SIZE);
            MemorySegment required = call.output("required", Library.SIZE);
            call.size("wanted", config.wanted);
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> {
                long pairs = written.get(Library.SIZE, 0);
                return new Products(
                        config.arena != null ? null : arena,
                        offsets.toArray(ValueLayout.JAVA_LONG),
                        arena.reach(scope, keys.get(Library.POINTER, 0), pairs * Key.BYTES),
                        arena.reach(scope, scores.get(Library.POINTER, 0), pairs * Double.BYTES),
                        required.get(Library.SIZE, 0));
            };
        });
    }

    static Request<RowsView> attending(RowsView queries, ValueLookup valueLookup, DenseAttendConfig config) {
        // The carry lands one state per query row, so a shorter array would be written past.
        if (config.carry != null && config.carry.length != queries.rows()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the carry holds one state per query row");
        }
        return new Request<>(DenseIndex.ATTEND, call -> {
            call.size("count", queries.rows());
            call.rows("queries", queries);
            call.size("head_count", config.headCount);
            call.pointer("value_lookup", valueLookup.upcall(call));
            call.rowsOut("outputs", config.outputs);
            call.size("expansion_factor", config.expansionFactor);
            call.real("score_scale", config.scoreScale);
            call.stats("stats", config.stats);
            DenseSoftmaxState[] carry = config.carry;
            if (carry == null) {
                return () -> config.outputs;
            }
            MemorySegment states =
                    call.output("carry", MemoryLayout.sequenceLayout(carry.length, DenseIndex.SOFTMAX_STATE));
            return () -> {
                for (int row = 0; row != carry.length; ++row) {
                    carry[row] = DenseSoftmaxState.lift(states.asSlice(row * DenseIndex.SOFTMAX_STATE.byteSize()));
                }
                return config.outputs;
            };
        });
    }

    static Request<Void> merging(DenseIndex[] donors) {
        return new Request<>(DenseIndex.MERGE, call -> {
            long index = call.arguments
                    .get(Library.POINTER, DenseIndex.MERGE.at("index"))
                    .address();
            for (DenseIndex donor : donors) {
                if (donor.handle.address().address() == index) {
                    throw new USearchException(Status.INVALID_ARGUMENT, "an index cannot absorb itself");
                }
            }
            MemorySegment slots = call.memory.allocate(Library.POINTER, donors.length);
            // Every surrendered donor is the call's to free once final, unless the library absorbed it.
            call.settle(() -> {
                for (int order = 0; order != donors.length; ++order) {
                    Library.invoke(DenseIndex.FREE, slots.getAtIndex(Library.POINTER, order));
                }
            });
            for (int order = 0; order != donors.length; ++order) {
                slots.setAtIndex(Library.POINTER, order, donors[order].handle.surrender());
            }
            call.size("count", donors.length);
            call.pointer("donors", slots);
            return NOTHING;
        });
    }

    static Request<Void> saving(String path) {
        return new Request<>(DenseIndex.SAVE, call -> {
            call.path("path", path);
            return NOTHING;
        });
    }

    static Request<Monolith> serializing(@Nullable Arena borrowed) {
        return new Request<>(DenseIndex.SAVE_BUFFER, call -> {
            Arena arena = borrowed != null ? borrowed : new Arena();
            MemorySegment bytes = call.output("bytes", Library.POINTER);
            MemorySegment length = call.output("length", Library.SIZE);
            java.lang.foreign.Arena scope = call.arena("arena", arena);
            return () -> new Monolith(
                    borrowed != null ? null : arena,
                    arena.reach(scope, bytes.get(Library.POINTER, 0), length.get(Library.SIZE, 0)));
        });
    }

    /** Refuses a negative breadth, which C would read as a size past any machine's. */
    static long nonNegative(long expansionFactor) {
        if (expansionFactor < 0) {
            throw new USearchException(Status.INVALID_ARGUMENT, "the expansion factor is negative");
        }
        return expansionFactor;
    }
}
