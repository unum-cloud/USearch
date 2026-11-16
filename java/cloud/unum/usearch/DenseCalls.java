/**
 *  @file java/cloud/unum/usearch/DenseCalls.java
 *  @author Ash Vardanian
 *  @date October 10, 2026
 *  @brief The one body of every dense verb, which each receiver runs blocking or deferred.
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
import cloud.unum.usearch.DenseIndex.DenseRefinePlan;
import cloud.unum.usearch.DenseIndex.DenseReshardConfig;
import cloud.unum.usearch.DenseIndex.DenseReshardPlan;
import cloud.unum.usearch.DenseIndex.DenseSampleConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchConfig;
import cloud.unum.usearch.DenseIndex.DenseSearchStream;
import cloud.unum.usearch.DenseIndex.DenseSoftmaxState;
import cloud.unum.usearch.DenseIndex.DenseVectors;
import cloud.unum.usearch.DenseIndex.RefineReport;
import cloud.unum.usearch.DenseIndex.Sample;
import cloud.unum.usearch.DenseIndex.ValueLookup;
import cloud.unum.usearch.Types.Arena;
import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.Matches;
import cloud.unum.usearch.Types.Monolith;
import cloud.unum.usearch.Types.Progress;
import cloud.unum.usearch.Types.RowsView;
import cloud.unum.usearch.Types.Scalar;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * The dense verbs, each written once as the structure it fills and what it answers, for
 * {@link DenseIndex}, {@link DenseIndex.On} and {@link DenseIndex.DeferredOn} to run blocking or
 * deferred. Binding-only.
 */
final class DenseCalls {

    private DenseCalls() {}

    /** One verb's argument structure and how a call fills it, whichever completion runs it. */
    record Request<Result>(Verb verb, Function<Call, Supplier<Result>> fill) {}

    /** What a verb answering nothing answers once final. */
    private static final Supplier<Void> NOTHING = () -> null;

    static Request<Void> reserving(long capacity) {
        return new Request<>(DenseIndex.RESERVE, call -> {
            call.size(DenseIndex.RESERVE_CAPACITY, capacity);
            return NOTHING;
        });
    }

    static Request<Void> clearing() {
        return new Request<>(DenseIndex.CLEAR, call -> NOTHING);
    }

    static Request<DenseIndexDescription> describing() {
        return new Request<>(DenseIndex.DESCRIBE, call -> {
            MemorySegment description = call.output(DenseIndex.DESCRIBE_DESCRIPTION, DenseIndex.DESCRIPTION);
            return () -> DenseIndexDescription.lift(description);
        });
    }

    static Request<Long> counting(Key key) {
        return new Request<>(DenseIndex.COUNT, call -> {
            call.key(DenseIndex.COUNT_KEY, key);
            MemorySegment entries = call.output(DenseIndex.COUNT_ENTRIES, Library.SIZE);
            return () -> entries.get(Library.SIZE, 0);
        });
    }

    static Request<DenseVectors> getting(Key key, Scalar outputsScalar, Arena arena) {
        return new Request<>(DenseIndex.GET, call -> {
            call.key(DenseIndex.GET_KEY, key);
            call.enumeration(DenseIndex.GET_OUTPUTS_SCALAR, outputsScalar.code());
            MemorySegment vectors = call.output(DenseIndex.GET_VECTORS, Library.ROWS_VIEW);
            MemorySegment written = call.output(DenseIndex.GET_WRITTEN, Library.SIZE);
            call.arena(DenseIndex.GET_ARENA, arena);
            return () -> DenseVectors.lift(call, vectors, written.get(Library.SIZE, 0));
        });
    }

    static Request<Void> inserting(Key[] keys, RowsView vectors, DenseInsertConfig config) {
        // The call has one count, so only this side sees the keys and the rows disagree.
        if (keys.length != vectors.rows()) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the key count does not match the vector row count");
        }
        return new Request<>(DenseIndex.INSERT, call -> {
            call.size(DenseIndex.INSERT_COUNT, keys.length);
            call.keys(DenseIndex.INSERT_KEYS, keys);
            call.rows(DenseIndex.INSERT_VECTORS, vectors);
            call.size(DenseIndex.INSERT_EXPANSION_FACTOR, config.expansionFactor);
            call.progress(DenseIndex.INSERT_PROGRESS, config.progress);
            call.stats(DenseIndex.INSERT_STATS, config.stats);
            return NOTHING;
        });
    }

    static Request<RefineReport> refining(DenseRefineConfig config) {
        return new Request<>(DenseIndex.REFINE, call -> {
            MemorySegment report = call.output(DenseIndex.REFINE_REPORT, DenseIndex.REPORT);
            call.size(DenseIndex.REFINE_EXPANSION_FACTOR, config.expansionFactor);
            call.size(DenseIndex.REFINE_MEMORY_LIMIT, config.memoryLimit);
            call.progress(DenseIndex.REFINE_PROGRESS, config.progress);
            call.stats(DenseIndex.REFINE_STATS, config.stats);
            if (config.plan != null) {
                config.plan.lend(call);
            }
            return () -> RefineReport.lift(report);
        });
    }

    static Request<RefineReport> refinePlanFinishing(DenseRefinePlan plan) {
        return new Request<>(DenseIndex.REFINE_PLAN_FINISH, call -> {
            call.pointer(DenseIndex.REFINE_PLAN_FINISH_PLAN, plan.take());
            MemorySegment report = call.output(DenseIndex.REFINE_PLAN_FINISH_REPORT, DenseIndex.REPORT);
            return () -> RefineReport.lift(report);
        });
    }

    static Request<DenseReachability> auditing() {
        return new Request<>(DenseIndex.HEALTH, call -> {
            MemorySegment reachability = call.output(DenseIndex.HEALTH_REACHABILITY, DenseIndex.REACHABILITY);
            return () -> DenseReachability.lift(reachability);
        });
    }

    static Request<Matches> searching(RowsView queries, DenseSearchConfig config, Arena arena) {
        return new Request<>(DenseIndex.SEARCH, call -> {
            call.size(DenseIndex.SEARCH_COUNT, queries.rows());
            call.rows(DenseIndex.SEARCH_QUERIES, queries);
            MemorySegment matches = call.output(DenseIndex.SEARCH_MATCHES, Library.MATCHES_VIEW);
            call.arena(DenseIndex.SEARCH_ARENA, arena);
            call.size(DenseIndex.SEARCH_WANTED, config.wanted);
            call.size(DenseIndex.SEARCH_EXPANSION_FACTOR, config.expansionFactor);
            call.enumeration(DenseIndex.SEARCH_TRAVERSAL, config.traversal.code());
            call.predicate(DenseIndex.SEARCH_PREDICATE, config.predicate);
            call.progress(DenseIndex.SEARCH_PROGRESS, config.progress);
            call.stats(DenseIndex.SEARCH_STATS, config.stats);
            return () -> Matches.lift(call, matches, queries.rows());
        });
    }

    static Request<DenseProducts> scoring(RowsView queries, long rowsPerQuery, DenseMaxsimConfig config, Arena arena) {
        // The call counts queries, so only this side sees the token rows fail to divide into them.
        if (rowsPerQuery <= 0 || queries.rows() % rowsPerQuery != 0) {
            throw new USearchException(Status.SHAPE_MISMATCH, "the token rows do not divide into whole queries");
        }
        int count = Types.Extents.toIndex(queries.rows() / rowsPerQuery, "the query count");
        return new Request<>(DenseIndex.MAXSIM, call -> {
            call.size(DenseIndex.MAXSIM_COUNT, count);
            call.rows(DenseIndex.MAXSIM_QUERIES, queries);
            call.size(DenseIndex.MAXSIM_ROWS_PER_QUERY, rowsPerQuery);
            MemorySegment matches = call.output(DenseIndex.MAXSIM_MATCHES, Library.MATCHES_TAPE_VIEW);
            MemorySegment written = call.output(DenseIndex.MAXSIM_WRITTEN, Library.SIZE);
            MemorySegment required = call.output(DenseIndex.MAXSIM_REQUIRED, Library.SIZE);
            call.arena(DenseIndex.MAXSIM_ARENA, arena);
            call.size(DenseIndex.MAXSIM_WANTED, config.wanted);
            call.size(DenseIndex.MAXSIM_EXPANSION_FACTOR, config.expansionFactor);
            call.predicate(DenseIndex.MAXSIM_PREDICATE, config.predicate);
            call.progress(DenseIndex.MAXSIM_PROGRESS, config.progress);
            call.stats(DenseIndex.MAXSIM_STATS, config.stats);
            return () -> DenseProducts.lift(
                    call, matches, count, written.get(Library.SIZE, 0), required.get(Library.SIZE, 0));
        });
    }

    static Request<DenseProducts> multiplying(RowsView left, DenseMultiplyConfig config, Arena arena) {
        int rows = Types.Extents.toIndex(left.rows(), "the left rows");
        return new Request<>(DenseIndex.MULTIPLY, call -> {
            call.size(DenseIndex.MULTIPLY_COUNT, rows);
            call.rows(DenseIndex.MULTIPLY_LEFT, left);
            MemorySegment products = call.output(DenseIndex.MULTIPLY_PRODUCTS, Library.MATCHES_TAPE_VIEW);
            MemorySegment written = call.output(DenseIndex.MULTIPLY_WRITTEN, Library.SIZE);
            MemorySegment required = call.output(DenseIndex.MULTIPLY_REQUIRED, Library.SIZE);
            call.arena(DenseIndex.MULTIPLY_ARENA, arena);
            call.size(DenseIndex.MULTIPLY_WANTED, config.wanted);
            call.size(DenseIndex.MULTIPLY_EXPANSION_FACTOR, config.expansionFactor);
            call.predicate(DenseIndex.MULTIPLY_PREDICATE, config.predicate);
            call.progress(DenseIndex.MULTIPLY_PROGRESS, config.progress);
            call.stats(DenseIndex.MULTIPLY_STATS, config.stats);
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
            call.size(DenseIndex.ATTEND_COUNT, queries.rows());
            call.rows(DenseIndex.ATTEND_QUERIES, queries);
            call.size(DenseIndex.ATTEND_HEAD_COUNT, config.headCount);
            valueLookup.lend(call);
            call.enumeration(DenseIndex.ATTEND_OUTPUTS_SCALAR, config.outputsScalar.code());
            MemorySegment outputs = call.output(DenseIndex.ATTEND_OUTPUTS, Library.ROWS_VIEW);
            call.arena(DenseIndex.ATTEND_ARENA, arena);
            call.size(DenseIndex.ATTEND_EXPANSION_FACTOR, config.expansionFactor);
            call.real(DenseIndex.ATTEND_SCORE_SCALE, config.scoreScale);
            call.predicate(DenseIndex.ATTEND_PREDICATE, config.predicate);
            call.progress(DenseIndex.ATTEND_PROGRESS, config.progress);
            call.stats(DenseIndex.ATTEND_STATS, config.stats);
            DenseSoftmaxState[] carry = config.carry;
            if (carry == null) {
                return () -> DenseVectors.lift(call, outputs, queries.rows());
            }
            MemorySegment states = call.output(
                    DenseIndex.ATTEND_CARRY, MemoryLayout.sequenceLayout(carry.length, DenseIndex.SOFTMAX_STATE));
            return () -> {
                for (int row = 0; row != carry.length; ++row) {
                    carry[row] = DenseSoftmaxState.lift(states.asSlice(row * DenseIndex.SOFTMAX_STATE.byteSize()));
                }
                return DenseVectors.lift(call, outputs, queries.rows());
            };
        });
    }

    static Request<RefineReport> resharding(DenseIndex[] sources, DenseIndex[] targets, DenseReshardConfig config) {
        return new Request<>(DenseIndex.RESHARD, call -> {
            call.size(DenseIndex.RESHARD_SOURCES_COUNT, sources.length);
            call.pointer(DenseIndex.RESHARD_SOURCES, handles(call, sources));
            call.size(DenseIndex.RESHARD_TARGETS_COUNT, targets.length);
            call.pointer(DenseIndex.RESHARD_TARGETS, handles(call, targets));
            if (config.memoryDomains != null) {
                MemorySegment ram = call.output(DenseIndex.RESHARD_RAM, Library.RAM_OPTIONS);
                call.ram(ram, 0, config.memoryDomains, null);
                Library.RAM_SHARD_CAPACITY.set(ram, 0L, Call.count(config.shardCapacity));
            }
            if (config.devices != null) {
                MemorySegment vram = call.output(DenseIndex.RESHARD_VRAM, Library.VRAM_OPTIONS);
                call.vram(vram, 0, config.devices, null);
                Library.VRAM_SHARD_CAPACITY.set(vram, 0L, Call.count(config.shardCapacity));
            }
            if (config.paths != null) {
                call.nvme(call.output(DenseIndex.RESHARD_NVME, Library.NVME_OPTIONS), 0, config.paths, null);
            }
            MemorySegment report = call.output(DenseIndex.RESHARD_REPORT, DenseIndex.REPORT);
            call.size(DenseIndex.RESHARD_EXPANSION_FACTOR, config.expansionFactor);
            call.size(DenseIndex.RESHARD_MEMORY_LIMIT, config.memoryLimit);
            call.seed(DenseIndex.RESHARD_SEED, config.seed);
            call.progress(DenseIndex.RESHARD_PROGRESS, config.progress);
            call.stats(DenseIndex.RESHARD_STATS, config.stats);
            if (config.plan != null) {
                config.plan.lend(call);
            }
            return () -> RefineReport.lift(report);
        });
    }

    static Request<RefineReport> reshardPlanFinishing(DenseReshardPlan plan) {
        return new Request<>(DenseIndex.RESHARD_PLAN_FINISH, call -> {
            call.pointer(DenseIndex.RESHARD_PLAN_FINISH_PLAN, plan.take());
            MemorySegment report = call.output(DenseIndex.RESHARD_PLAN_FINISH_REPORT, DenseIndex.REPORT);
            return () -> RefineReport.lift(report);
        });
    }

    /** Lends every index to the call and copies their handles into the array C reads. */
    private static MemorySegment handles(Call call, DenseIndex[] indexes) {
        MemorySegment[] addresses = new MemorySegment[indexes.length];
        for (int order = 0; order != indexes.length; ++order) {
            addresses[order] = indexes[order].handle.address();
            call.lend(indexes[order]);
        }
        return call.addresses(addresses);
    }

    static Request<Sample> sampling(long wanted, Arena arena, DenseSampleConfig config) {
        return new Request<>(DenseIndex.SAMPLE, call -> {
            call.size(DenseIndex.SAMPLE_SHOWN_COUNT, config.shown.length);
            call.keys(DenseIndex.SAMPLE_SHOWN, config.shown);
            call.rows(DenseIndex.SAMPLE_SHOWN_COORDINATES, RowsView.of(config.shownCoordinates, 3));
            call.size(DenseIndex.SAMPLE_FOCUS_COUNT, config.focus.length);
            call.keys(DenseIndex.SAMPLE_FOCUS, config.focus);
            MemorySegment keys = call.output(DenseIndex.SAMPLE_KEYS, Library.KEYS_VIEW);
            MemorySegment entryOrdinals = call.output(DenseIndex.SAMPLE_ENTRY_ORDINALS, Library.POINTER);
            MemorySegment parents = call.output(DenseIndex.SAMPLE_PARENTS, Library.MATCHES_VIEW);
            MemorySegment coordinates = call.output(DenseIndex.SAMPLE_COORDINATES, Library.ROWS_VIEW);
            MemorySegment written = call.output(DenseIndex.SAMPLE_WRITTEN, Library.SIZE);
            call.arena(DenseIndex.SAMPLE_ARENA, arena);
            call.size(DenseIndex.SAMPLE_WANTED, wanted);
            call.seed(DenseIndex.SAMPLE_SEED, config.seed);
            call.size(DenseIndex.SAMPLE_EXPANSION_FACTOR, config.expansionFactor);
            call.progress(DenseIndex.SAMPLE_PROGRESS, config.progress);
            call.stats(DenseIndex.SAMPLE_STATS, config.stats);
            return () -> Sample.lift(call, keys, entryOrdinals, parents, coordinates, written.get(Library.SIZE, 0));
        });
    }

    static Request<Void> saving(String path, @Nullable Progress progress) {
        return new Request<>(DenseIndex.SAVE, call -> {
            call.path(DenseIndex.SAVE_PATH, path);
            call.progress(DenseIndex.SAVE_PROGRESS, progress);
            return NOTHING;
        });
    }

    static Request<Monolith> serializing(Arena arena, @Nullable Progress progress) {
        return new Request<>(DenseIndex.SAVE_BUFFER, call -> {
            MemorySegment monolith = call.output(DenseIndex.SAVE_BUFFER_MONOLITH, Library.BYTES_VIEW);
            call.arena(DenseIndex.SAVE_BUFFER_ARENA, arena);
            call.progress(DenseIndex.SAVE_BUFFER_PROGRESS, progress);
            return () -> Monolith.lift(call, monolith);
        });
    }

    static Request<DenseCursor> cursorOpening(DenseIndex index, @Nullable Executor executor, Key[] starts) {
        return new Request<>(DenseIndex.CURSOR_INIT, call -> {
            MemorySegment opened = call.output(DenseIndex.CURSOR_INIT_CURSOR, Library.POINTER);
            if (starts.length != 0) {
                call.size(DenseIndex.CURSOR_INIT_STARTS_COUNT, starts.length);
                call.pointer(DenseIndex.CURSOR_INIT_STARTS, call.staged(starts));
            }
            return () -> new DenseCursor(opened.get(Library.POINTER, 0), index, executor);
        });
    }

    static Request<DenseSearchStream> searchStreamOpening(
            DenseIndex index, @Nullable Executor executor, RowsView query, long expansionFactor) {
        return new Request<>(DenseIndex.SEARCH_STREAM_INIT, call -> {
            MemorySegment opened = call.output(DenseIndex.SEARCH_STREAM_INIT_STREAM, Library.POINTER);
            call.rows(DenseIndex.SEARCH_STREAM_INIT_QUERY, query);
            call.size(DenseIndex.SEARCH_STREAM_INIT_EXPANSION_FACTOR, expansionFactor);
            return () -> new DenseSearchStream(opened.get(Library.POINTER, 0), index, executor);
        });
    }
}
