/**
 *  @file java/cloud/unum/usearch/DenseVerbs.java
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The dense verbs, written once for the index and for every receiver that runs them.
 */
package cloud.unum.usearch;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.util.function.Function;
import java.util.function.Supplier;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.DenseIndex.DenseAttendConfig;
import cloud.unum.usearch.DenseIndex.DenseCursor;
import cloud.unum.usearch.DenseIndex.DenseIndexDescription;
import cloud.unum.usearch.DenseIndex.DenseInsertConfig;
import cloud.unum.usearch.DenseIndex.DenseMaxsimConfig;
import cloud.unum.usearch.DenseIndex.DenseMultiplyConfig;
import cloud.unum.usearch.DenseIndex.DenseProducts;
import cloud.unum.usearch.DenseIndex.DenseReachability;
import cloud.unum.usearch.DenseIndex.DenseRefineConfig;
import cloud.unum.usearch.DenseIndex.DenseSampleConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchStream;
import cloud.unum.usearch.DenseIndex.DenseSoftmaxState;
import cloud.unum.usearch.DenseIndex.DenseSplitConfig;
import cloud.unum.usearch.DenseIndex.DenseVectors;
import cloud.unum.usearch.DenseIndex.RefineReport;
import cloud.unum.usearch.DenseIndex.Sample;
import cloud.unum.usearch.DenseIndex.ValueLookup;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Handle;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.Residency;
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
 * {@code Dense*Config}, named after the C structure it fills. Every verb whose output scales with
 * the call takes an {@link Arena} last, borrows it until it returns, and answers a view over it that
 * the arena's next call invalidates.
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
     * Claims capacity for a number of incoming vectors ahead of insertion, a hint, since insertions
     * grow the index on demand. The capacity is the desired total, including the current size.
     * Calls {@code usearch_dense_reserve}.
     *
     * @param capacity the desired total number of entries, including the current size.
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

    /**
     * How many entries are stored under a key; zero means the index holds none. Calls {@code usearch_dense_count}.
     *
     * @param key the key to look up.
     */
    public long count(Key key) {
        return block(counting(key));
    }

    /**
     * Whether the index holds at least one entry under a key. Binding-only sugar over {@link #count}.
     *
     * @param key the key to look up.
     */
    public boolean contains(Key key) {
        return count(key) != 0;
    }

    /**
     * Retrieves every vector stored under a key into {@code arena}, one row per entry sharing it, in
     * insertion order, packed at the natural width of the kind asked for - one call, however many
     * there turn out to be. Calls {@code usearch_dense_get}.
     *
     * @param key the key to look up.
     * @param outputsScalar the scalar kind the rows are converted into.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @return the rows read in place, none when the key holds nothing.
     * @throws USearchException UNKNOWN_KIND on a scalar the engine cannot convert into, and
     *     CONTENDED when the arena serves another call.
     */
    public DenseVectors get(Key key, Scalar outputsScalar, Arena arena) {
        return block(getting(key, outputsScalar, arena));
    }

    /**
     * Inserts a matrix of vectors under their keys, with every knob derived.
     *
     * @param keys one key per vector row.
     * @param vectors the rows to insert, of any scalar kind the engine converts.
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
     * ingest. A batch is all-or-nothing: a returning call published every entry, a failing one
     * published none. An accelerator fault severe enough to poison the device context is the sole
     * exception, reporting DEVICE and latching the index unusable rather than partially applied.
     * Calls {@code usearch_dense_insert}.
     *
     * @param keys one key per vector row.
     * @param vectors the rows to insert, of any scalar kind the engine converts.
     * @param config the insertion knobs.
     * @throws USearchException SHAPE_MISMATCH when the key count and the row count disagree or the
     *     rows are not as wide as the frozen geometry, UNKNOWN_KIND on a scalar the engine cannot convert, READ_ONLY on a view, and
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
    public RefineReport refine() {
        return block(refining(new DenseRefineConfig()));
    }

    /**
     * Re-selects the neighbors of every stored entry against the finished graph, improving neighbors
     * chosen when the graph around them was still incomplete.
     *
     * <p>
     * Searches may run throughout on a RAM handle, since re-selection publishes one node at a time
     * and never removes one; insertions may not. Re-selecting a settled node changes nothing, so a
     * caller loops until the report's {@code changed} falls below whatever rate it considers settled.
     * Calls {@code usearch_dense_refine}.
     *
     * @param config the refinement knobs.
     * @return the nodes re-selected and changed, the islands found and reconnected, and the audit
     *     closing the heal; {@code changed} at zero means the graph has settled.
     * @throws USearchException READ_ONLY on a view.
     */
    public RefineReport refine(DenseRefineConfig config) {
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
     * Resolves a matrix of queries into up to {@code wanted} nearest matches apiece, with every other
     * knob derived.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param wanted the most matches kept per query.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses {@link #search(RowsView, DenseSearchConfig, Arena)} names.
     */
    public Matches search(RowsView queries, long wanted, Arena arena) {
        return block(searching(queries, new DenseSearchConfig(wanted), arena));
    }

    /**
     * Resolves a matrix of queries into {@code arena} - the native batch operation, parallelized
     * internally. Never writes to the graph, so any number of these may overlap with concurrent
     * insertions. Calls {@code usearch_dense_search}.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param config the wanted count and the other search knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException SHAPE_MISMATCH when the rows are not as wide as the frozen geometry,
     *     UNKNOWN_KIND on a scalar the engine cannot convert, and CONTENDED when the arena serves
     *     another call.
     */
    public Matches search(RowsView queries, DenseSearchConfig config, Arena arena) {
        return block(searching(queries, config, arena));
    }

    /**
     * Late-interaction scoring keeping up to {@code wanted} keys per query, zero keeping every key a
     * query scored, with every other knob derived.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param rowsPerQuery the token rows every query carries.
     * @param wanted the most keys kept per query, zero keeping every key it scored.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses {@link #maxsim(RowsView, long, DenseMaxsimConfig, Arena)}
     *     names.
     */
    public DenseProducts maxsim(RowsView queries, long rowsPerQuery, long wanted, Arena arena) {
        return block(scoring(queries, rowsPerQuery, new DenseMaxsimConfig(wanted), arena));
    }

    /**
     * Late-interaction scoring of multi-row queries against multi-vector documents: every query
     * row takes its best match within a key, and those maxima sum to that key's score. A document
     * is one key holding its token rows, so the wanted count is the "k" over keys rather than over
     * rows.
     *
     * <p>
     * Every query carries the same number of rows, so a padded token batch applies as is, and the
     * query count is the row count divided by that. Approximate in the same way a search is: a key
     * can score below its exact value but never above it. Runs on the inner-product kinds alone.
     * Each query's keys land in {@code arena} largest score first, as one ragged output like
     * {@link #multiply}'s. Calls {@code usearch_dense_maxsim}.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param rowsPerQuery the token rows every query carries.
     * @param config the wanted count and the other scoring knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind,
     *     SHAPE_MISMATCH when the token rows do not divide into whole queries or the rows are not as
     *     wide as the frozen geometry, and CONTENDED when the arena serves another call.
     */
    public DenseProducts maxsim(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
        return block(scoring(queries, rowsPerQuery, config, arena));
    }

    /**
     * Approximate sparse product keeping every pair, with every other knob derived.
     *
     * @param left the rows multiplied against the indexed vectors.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses {@link #multiply(RowsView, DenseMultiplyConfig, Arena)}
     *     names.
     */
    public DenseProducts multiply(RowsView left, Arena arena) {
        return block(multiplying(left, new DenseMultiplyConfig(), arena));
    }

    /**
     * Approximate sparse product of a matrix against the indexed vectors - the linear-algebra sibling
     * of {@link #search}, reporting every pair the traversal scored rather than only the nearest.
     * Pairs arrive in traversal order, and the pairs the walk never reaches are the low and negative
     * products. Runs on the inner-product kinds alone.
     *
     * <p>
     * Only the call knows how many pairs it scores, so the keys, the products and their offsets land
     * in {@code arena} as one ragged output. Calls {@code usearch_dense_multiply}.
     *
     * @param left the rows multiplied against the indexed vectors.
     * @param config the product knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException CONFIG_MISMATCH on a graph frozen under a distance kind,
     *     SHAPE_MISMATCH when the rows are not as wide as the frozen geometry, and CONTENDED when the
     *     arena serves another call.
     */
    public DenseProducts multiply(RowsView left, DenseMultiplyConfig config, Arena arena) {
        return block(multiplying(left, config, arena));
    }

    /**
     * Softmax attention with every knob derived.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param valueLookup resolves every scored key to its value row.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException the statuses
     *     {@link #attend(RowsView, ValueLookup, DenseAttendConfig, Arena)} names.
     */
    public DenseVectors attend(RowsView queries, ValueLookup valueLookup, Arena arena) {
        return block(attending(queries, valueLookup, new DenseAttendConfig(), arena));
    }

    /**
     * Softmax attention of every query row over every indexed key its walk scores, folded as Flash
     * Attention folds key tiles, with no top-k cut, writing one row per query into {@code arena}
     * already divided by the partition. Runs on the inner-product kinds alone, since a softmax needs
     * logits.
     *
     * <p>
     * The index holds one layer's keys for one key-value head and no values, so the lookup resolves
     * every scored key to its value row. Calls {@code usearch_dense_attend}.
     *
     * @param queries the query rows, as wide as the frozen geometry.
     * @param valueLookup resolves every scored key to its value row.
     * @param config the attention knobs.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException SHAPE_MISMATCH when a scored key resolves to no row or the head count
     *     does not divide the rows; CONFIG_MISMATCH on a graph frozen under a distance kind; CONTENDED
     *     when the arena serves another call; DEVICE on a VRAM arena, whose walk needs a device
     *     lookup; and whatever the lookup raised, which outranks the status it turned into.
     */
    public DenseVectors attend(RowsView queries, ValueLookup valueLookup, DenseAttendConfig config, Arena arena) {
        return block(attending(queries, valueLookup, config, arena));
    }

    /**
     * Absorbs every donor into this index, one after another, keeping their graphs rather than
     * re-inserting their vectors and linking each in across the appended part. Every donor refuses
     * every verb until the call is final; then each one absorbed stays closed, and one it did not
     * absorb is a live index again, which the caller still owns, though its entries may already have
     * moved here. Calls {@code usearch_dense_absorb}.
     *
     * @param donors the indexes absorbed, of this index's residency, scalar, metric and geometry.
     * @return what the closing refinement did to the graph.
     * @throws USearchException READ_ONLY on a view, CONFIG_MISMATCH on a donor of another residency,
     *     scalar, metric or geometry, and INVALID_ARGUMENT when the index is among its own donors or a
     *     donor is named twice, which consumes none.
     */
    public RefineReport absorb(DenseIndex... donors) {
        return block(absorbing(donors));
    }

    /**
     * Splits this index into the empty receivers, the inverse of {@link #absorb}: the entries are
     * partitioned along the graph, each part lands in one receiver with its edges kept, and a closing
     * refinement re-links every receiver, leaving this index empty and live. Calls
     * {@code usearch_dense_split}.
     *
     * @param config the partitioning knobs.
     * @param receivers at least two empty indexes matching this one, which take the parts.
     * @return what the closing refinements did, across the receivers.
     * @throws USearchException INVALID_ARGUMENT on fewer than two receivers, and CONFIG_MISMATCH on a
     *     receiver of another residency, scalar, metric, geometry or shard count, one holding entries,
     *     or this index itself.
     */
    public RefineReport split(DenseSplitConfig config, DenseIndex... receivers) {
        return block(splitting(config, receivers));
    }

    /**
     * Draws keys around the focus, or across the whole index, each hanging from a shown or earlier
     * draw and placed in three dimensions beside the shown ones - so a view grows one connected
     * sample as it zooms. Writes up to {@code wanted} draws, and a larger count under the same seed
     * draws a superset; only a RAM index samples yet. Calls {@code usearch_dense_sample}.
     *
     * @param wanted the most draws written.
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @param config the focus, seed and other sampling knobs.
     * @throws USearchException INVALID_ARGUMENT on a negative count, UNSUPPORTED on a GPU or NVMe
     *     handle, and CONTENDED when the arena serves another call.
     */
    public Sample sample(long wanted, Arena arena, DenseSampleConfig config) {
        return block(sampling(wanted, arena, config));
    }

    /**
     * Writes the index under a path, in the layout its extension picks - one {@code .safegraphs}
     * monolith, or a {@code .safegraphs.json} manifest beside its shards. Demands a mutation-free
     * window; concurrent readers keep searching throughout. Calls {@code usearch_dense_save}.
     *
     * @param path the destination; its extension picks the layout.
     * @throws USearchException INVALID_ARGUMENT on an empty path, and IO when the destination cannot
     *     be written - which leaves a file without its header, for the caller to remove.
     */
    public void save(String path) {
        block(saving(path));
    }

    /**
     * Writes the index as one monolith - every shard, entries in placement order - the same bytes
     * {@link #save} writes to a {@code .safegraphs} path, into {@code arena}. Calls
     * {@code usearch_dense_save_buffer}.
     *
     * @param arena the arena the output lands in, borrowed until the call returns.
     * @throws USearchException CONTENDED when the arena serves another call.
     */
    public Monolith saveBuffer(Arena arena) {
        return block(serializing(arena));
    }

    /**
     * Opens a walk over the current contents, pulled on this receiver's executor. Each shard's
     * publication watermark is read as the walk enters it, so an entry once passed is never revisited.
     * Without {@code starts} it yields every entry as stored; with them, the start keys' own entries
     * first in order, then the entries those link to breadth-first, each once. Calls
     * {@code usearch_dense_cursor_init}.
     *
     * @param starts the keys whose entries the walk starts from, or none to yield every entry.
     */
    public DenseCursor cursor(Key... starts) {
        DenseIndex index = index();
        return Call.block(DenseIndex.CURSOR_INIT, index.handle, null, call -> {
            MemorySegment opened = call.output("cursor", Library.POINTER);
            call.keyRun("starts_count", "starts", starts);
            return () -> new DenseCursor(opened.get(Library.POINTER, 0), index, executor());
        });
    }

    /**
     * Opens a resumable search for one query row, with the breadth of the candidates derived.
     *
     * @param query exactly one query row.
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
     * @param query exactly one query row.
     * @param expansionFactor the breadth of the candidates, zero to derive it.
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

    static Request<DenseVectors> getting(Key key, Scalar outputsScalar, Arena arena) {
        return new Request<>(DenseIndex.GET, call -> {
            call.key("key", key);
            call.enumeration("outputs_scalar", outputsScalar.code());
            MemorySegment vectors = call.output("vectors", Library.ROWS_VIEW);
            MemorySegment written = call.output("written", Library.SIZE);
            call.arena("arena", arena);
            return () -> DenseVectors.lift(call, vectors, written.get(Library.SIZE, 0));
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

    static Request<RefineReport> refining(DenseRefineConfig config) {
        return new Request<>(DenseIndex.REFINE, call -> {
            MemorySegment report = call.output("report", DenseIndex.REFINE_REPORT);
            call.size("refine.expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> RefineReport.lift(report);
        });
    }

    static Request<DenseReachability> auditing() {
        return new Request<>(DenseIndex.HEALTH, call -> {
            MemorySegment reachability = call.output("reachability", DenseIndex.REACHABILITY);
            return () -> DenseReachability.lift(reachability);
        });
    }

    static Request<Matches> searching(RowsView queries, DenseSearchConfig config, Arena arena) {
        return new Request<>(DenseIndex.SEARCH, call -> {
            call.size("count", queries.rows());
            call.rows("queries", queries);
            MemorySegment matches = call.output("matches", Library.MATCHES_VIEW);
            call.arena("arena", arena);
            call.size("wanted", config.wanted);
            call.size("expansion_factor", config.expansionFactor);
            call.enumeration("traversal", config.traversal.code());
            call.stats("stats", config.stats);
            return () -> Matches.lift(call, matches, queries.rows());
        });
    }

    static Request<DenseProducts> scoring(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
        if (rowsPerQuery <= 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "a query cannot carry zero rows");
        }
        if (queries.rows() % rowsPerQuery != 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the token rows do not divide into whole queries");
        }
        int count = Types.Extents.toIndex(queries.rows() / rowsPerQuery, "the query count");
        return new Request<>(DenseIndex.MAXSIM, call -> {
            call.size("count", count);
            call.rows("queries", queries);
            call.size("rows_per_query", rowsPerQuery);
            MemorySegment matches = call.output("matches", Library.MATCHES_TAPE_VIEW);
            MemorySegment written = call.output("written", Library.SIZE);
            MemorySegment required = call.output("required", Library.SIZE);
            call.arena("arena", arena);
            call.size("wanted", config.wanted);
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> DenseProducts.lift(
                    call, matches, count, written.get(Library.SIZE, 0), required.get(Library.SIZE, 0));
        });
    }

    static Request<DenseProducts> multiplying(RowsView left, DenseMultiplyConfig config, Arena arena) {
        int rows = Types.Extents.toIndex(left.rows(), "the left rows");
        return new Request<>(DenseIndex.MULTIPLY, call -> {
            call.size("count", rows);
            call.rows("left", left);
            MemorySegment products = call.output("products", Library.MATCHES_TAPE_VIEW);
            MemorySegment written = call.output("written", Library.SIZE);
            MemorySegment required = call.output("required", Library.SIZE);
            call.arena("arena", arena);
            call.size("wanted", config.wanted);
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> DenseProducts.lift(
                    call, products, rows, written.get(Library.SIZE, 0), required.get(Library.SIZE, 0));
        });
    }

    static Request<DenseVectors> attending(
            RowsView queries, ValueLookup valueLookup, DenseAttendConfig config, Arena arena) {
        // The carry lands one state per query row, so a shorter array would be written past.
        if (config.carry != null && config.carry.length != queries.rows()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the carry holds one state per query row");
        }
        return new Request<>(DenseIndex.ATTEND, call -> {
            call.size("count", queries.rows());
            call.rows("queries", queries);
            call.size("head_count", config.headCount);
            valueLookup.lend(call);
            call.enumeration("outputs_scalar", config.outputsScalar.code());
            MemorySegment outputs = call.output("outputs", Library.ROWS_VIEW);
            call.arena("arena", arena);
            // A VRAM arena serves only a VRAM call, whose walk would run this host upcall on the device.
            if (arena.describe().residency() == Residency.VRAM) {
                throw new USearchException(Status.DEVICE, "a VRAM walk needs a device-callable value lookup");
            }
            call.size("expansion_factor", config.expansionFactor);
            call.real("score_scale", config.scoreScale);
            call.stats("stats", config.stats);
            DenseSoftmaxState[] carry = config.carry;
            if (carry == null) {
                return () -> DenseVectors.lift(call, outputs, queries.rows());
            }
            MemorySegment states =
                    call.output("carry", MemoryLayout.sequenceLayout(carry.length, DenseIndex.SOFTMAX_STATE));
            return () -> {
                for (int row = 0; row != carry.length; ++row) {
                    carry[row] = DenseSoftmaxState.lift(states.asSlice(row * DenseIndex.SOFTMAX_STATE.byteSize()));
                }
                return DenseVectors.lift(call, outputs, queries.rows());
            };
        });
    }

    static Request<RefineReport> absorbing(DenseIndex[] donors) {
        return new Request<>(DenseIndex.ABSORB, call -> {
            long index = call.arguments
                    .get(Library.POINTER, DenseIndex.ABSORB.at("index"))
                    .address();
            // Every donor is checked before any is surrendered, so a refused absorb consumes none of them.
            for (int order = 0; order != donors.length; ++order) {
                if (donors[order].handle.address().address() == index) {
                    throw new USearchException(Status.INVALID_ARGUMENT, "an index cannot absorb itself");
                }
                for (int earlier = 0; earlier != order; ++earlier) {
                    if (donors[earlier] == donors[order]) {
                        throw new USearchException(Status.INVALID_ARGUMENT, "a donor is named twice");
                    }
                }
            }
            MemorySegment slots = call.memory.allocate(Library.POINTER, donors.length);
            // A slot the library left unnulled is a live handle once final, so it goes back to its donor.
            call.settle(() -> {
                for (int order = 0; order != donors.length; ++order) {
                    MemorySegment survivor = slots.getAtIndex(Library.POINTER, order);
                    if (!survivor.equals(MemorySegment.NULL)) {
                        donors[order].handle = new Handle(survivor, DenseIndex.FREE);
                    }
                }
            });
            for (int order = 0; order != donors.length; ++order) {
                slots.setAtIndex(Library.POINTER, order, donors[order].handle.surrender());
            }
            call.size("count", donors.length);
            call.pointer("donors", slots);
            MemorySegment report = call.output("report", DenseIndex.REFINE_REPORT);
            return () -> RefineReport.lift(report);
        });
    }

    static Request<RefineReport> splitting(DenseSplitConfig config, DenseIndex[] receivers) {
        return new Request<>(DenseIndex.SPLIT, call -> {
            MemorySegment handles = call.memory.allocate(Library.POINTER, Math.max(receivers.length, 1));
            for (int order = 0; order != receivers.length; ++order) {
                handles.setAtIndex(Library.POINTER, order, receivers[order].handle.address());
                call.lend(receivers[order]);
            }
            call.size("count", receivers.length);
            call.pointer("receivers", handles);
            MemorySegment report = call.output("report", DenseIndex.REFINE_REPORT);
            call.size("refine.expansion_factor", config.expansionFactor);
            call.size("seed", config.seed);
            call.stats("stats", config.stats);
            return () -> RefineReport.lift(report);
        });
    }

    static Request<Sample> sampling(long wanted, Arena arena, DenseSampleConfig config) {
        long bound = DenseIndex.nonNegative(wanted, "the wanted count is negative");
        return new Request<>(DenseIndex.SAMPLE, call -> {
            call.size("shown_count", config.shown.length);
            call.keys("shown", config.shown);
            call.rows("shown_coordinates", RowsView.of(config.shownCoordinates, 3));
            call.size("focus_count", config.focus.length);
            call.keys("focus", config.focus);
            MemorySegment keys = call.output("keys", Library.KEYS_VIEW);
            MemorySegment entryOrdinals = call.output("entry_ordinals", Library.POINTER);
            MemorySegment parents = call.output("parents", Library.MATCHES_VIEW);
            MemorySegment coordinates = call.output("coordinates", Library.ROWS_VIEW);
            MemorySegment written = call.output("written", Library.SIZE);
            call.arena("arena", arena);
            call.size("wanted", bound);
            call.size("seed", config.seed);
            call.size("expansion_factor", config.expansionFactor);
            call.stats("stats", config.stats);
            return () -> Sample.lift(call, keys, entryOrdinals, parents, coordinates, written.get(Library.SIZE, 0));
        });
    }

    static Request<Void> saving(String path) {
        return new Request<>(DenseIndex.SAVE, call -> {
            call.path("path", path);
            return NOTHING;
        });
    }

    static Request<Monolith> serializing(Arena arena) {
        return new Request<>(DenseIndex.SAVE_BUFFER, call -> {
            MemorySegment monolith = call.output("monolith", Library.BYTES_VIEW);
            call.arena("arena", arena);
            return () -> Monolith.lift(call, monolith);
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
