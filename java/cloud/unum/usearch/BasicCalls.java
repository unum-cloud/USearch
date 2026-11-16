/**
 *  @file java/cloud/unum/usearch/BasicCalls.java
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief The one body of every verb the sequence, shapes and geospatial indexes share.
 */
package cloud.unum.usearch;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.StructLayout;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.VarHandle;
import java.nio.charset.StandardCharsets;
import java.util.function.Consumer;
import java.util.function.Function;

import org.jspecify.annotations.Nullable;

import cloud.unum.usearch.DenseCalls.Request;
import cloud.unum.usearch.DenseIndex.DenseTraversalStats;
import cloud.unum.usearch.DenseIndex.RefineReport;
import cloud.unum.usearch.Types.Call;
import cloud.unum.usearch.Types.Executor;
import cloud.unum.usearch.Types.Handle;
import cloud.unum.usearch.Types.Key;
import cloud.unum.usearch.Types.PhaseProgress;
import cloud.unum.usearch.Types.Progress;
import cloud.unum.usearch.Types.Status;
import cloud.unum.usearch.Types.USearchException;
import cloud.unum.usearch.Types.Verb;

/**
 * The verbs {@link SequenceIndex}, {@link ShapesIndex} and {@link GeospatialIndex} spell alike, each
 * written once over the {@link Family} naming one modality's symbols, for every receiver of each to
 * run blocking or deferred. Where a structure matches the dense one field for field, its layout is
 * the dense one's. Binding-only.
 */
final class BasicCalls {

    private BasicCalls() {}

    /** The symbols of the verbs one basic modality shares with the others, as {@code usearch_<family>_<verb>}. */
    record Family(
            Verb clear,
            Verb describe,
            Verb count,
            Verb refine,
            Verb refinePlanFinish,
            Verb health,
            Verb reshard,
            Verb reshardPlanFinish,
            Verb cursorInit,
            Verb cursorNext,
            Verb sample,
            Verb save,
            Verb saveBuffer,
            Verb metadataRead,
            Verb metadataReadBuffer,
            MethodHandle free,
            MethodHandle cursorFree) {

        /** The verbs of {@code family}: sequence, shapes or geospatial. */
        static Family of(String family) {
            String stem = "usearch_" + family + "_";
            return new Family(
                    new Verb(stem + "clear", DenseIndex.CLEAR.arguments()),
                    new Verb(stem + "describe", DenseIndex.DESCRIBE.arguments()),
                    new Verb(stem + "count", COUNTING),
                    new Verb(stem + "refine", DenseIndex.REFINE.arguments()),
                    new Verb(stem + "refine_plan_finish", DenseIndex.REFINE_PLAN_FINISH.arguments()),
                    new Verb(stem + "health", DenseIndex.HEALTH.arguments()),
                    new Verb(stem + "reshard", RESHARDING),
                    new Verb(stem + "reshard_plan_finish", DenseIndex.RESHARD_PLAN_FINISH.arguments()),
                    new Verb(stem + "cursor_init", DenseIndex.CURSOR_INIT.arguments()),
                    new Verb(stem + "cursor_next", DenseIndex.CURSOR_NEXT.arguments()),
                    new Verb(stem + "sample", DenseIndex.SAMPLE.arguments()),
                    new Verb(stem + "save", DenseIndex.SAVE.arguments()),
                    new Verb(stem + "save_buffer", DenseIndex.SAVE_BUFFER.arguments()),
                    new Verb(stem + "metadata_read", DenseIndex.METADATA_READ.arguments()),
                    new Verb(stem + "metadata_read_buffer", DenseIndex.METADATA_READ.arguments()),
                    Library.downcall(stem + "free", FunctionDescriptor.ofVoid(Library.POINTER)),
                    Library.downcall(stem + "cursor_free", FunctionDescriptor.ofVoid(Library.POINTER)));
        }
    }

    /**
     * The knobs every refining call shares, and the plan it opens or continues: what a modality's
     * refine and reshard configurations hold, whatever their plan's type.
     */
    record Rounds(
            long expansionFactor,
            long memoryLimit,
            @Nullable PlanCell plan,
            @Nullable Progress progress,
            @Nullable DenseTraversalStats stats) {}

    /**
     * The two slots a plan lends every call naming it: the lazy handle C opens, continues and a
     * finish frees, and where the last call stopped. Each modality's plan types hold one.
     */
    static final class PlanCell {

        private final MemorySegment plan = java.lang.foreign.Arena.ofAuto().allocate(Library.POINTER);
        private final MemorySegment phase = java.lang.foreign.Arena.ofAuto().allocate(Library.PHASE_PROGRESS);

        /** Where the last call naming the plan stopped. */
        PhaseProgress phase() {
            return PhaseProgress.lift(phase);
        }

        void lend(Call call, VarHandle planField, VarHandle phaseField) {
            call.pointer(planField, plan);
            call.pointer(phaseField, phase);
            call.lend(this);
        }

        /** The open plan a finish frees, leaving this one to open afresh on its next call. */
        MemorySegment take() {
            MemorySegment open = plan.get(Library.POINTER, 0);
            plan.set(Library.POINTER, 0, MemorySegment.NULL);
            return open;
        }
    }

    static <Description> Request<Description> describing(
            Family family, StructLayout layout, Function<MemorySegment, Description> lift) {
        return new Request<>(family.describe(), call -> {
            MemorySegment description = call.output(DenseIndex.DESCRIBE_DESCRIPTION, layout);
            return () -> lift.apply(description);
        });
    }

    static <Stats> Request<Stats> counting(
            Family family, Key key, MemoryLayout layout, Function<MemorySegment, Stats> lift) {
        return new Request<>(family.count(), call -> {
            call.key(COUNT_KEY, key);
            MemorySegment stats = call.output(COUNT_KEY_STATS, layout);
            return () -> lift.apply(stats);
        });
    }

    static Request<RefineReport> refining(Family family, Rounds rounds) {
        return new Request<>(family.refine(), call -> {
            MemorySegment report = call.output(DenseIndex.REFINE_REPORT, DenseIndex.REPORT);
            call.size(DenseIndex.REFINE_EXPANSION_FACTOR, rounds.expansionFactor());
            call.size(DenseIndex.REFINE_MEMORY_LIMIT, rounds.memoryLimit());
            call.progress(DenseIndex.REFINE_PROGRESS, rounds.progress());
            call.stats(DenseIndex.REFINE_STATS, rounds.stats());
            if (rounds.plan() != null) {
                rounds.plan().lend(call, DenseIndex.REFINE_PLAN, DenseIndex.REFINE_PHASE);
            }
            return () -> RefineReport.lift(report);
        });
    }

    static Request<RefineReport> refinePlanFinishing(Family family, PlanCell plan) {
        return new Request<>(family.refinePlanFinish(), call -> {
            call.pointer(DenseIndex.REFINE_PLAN_FINISH_PLAN, plan.take());
            MemorySegment report = call.output(DenseIndex.REFINE_PLAN_FINISH_REPORT, DenseIndex.REPORT);
            return () -> RefineReport.lift(report);
        });
    }

    static Request<RefineReport> resharding(
            Family family, Handle[] sources, Handle[] targets, long seed, Rounds rounds) {
        return new Request<>(family.reshard(), call -> {
            call.size(RESHARD_SOURCES_COUNT, sources.length);
            call.pointer(RESHARD_SOURCES, handles(call, sources));
            call.size(RESHARD_TARGETS_COUNT, targets.length);
            call.pointer(RESHARD_TARGETS, handles(call, targets));
            MemorySegment report = call.output(RESHARD_REPORT, DenseIndex.REPORT);
            call.size(RESHARD_EXPANSION_FACTOR, rounds.expansionFactor());
            call.size(RESHARD_MEMORY_LIMIT, rounds.memoryLimit());
            call.seed(RESHARD_SEED, seed);
            call.progress(RESHARD_PROGRESS, rounds.progress());
            call.stats(RESHARD_STATS, rounds.stats());
            if (rounds.plan() != null) {
                rounds.plan().lend(call, RESHARD_PLAN, RESHARD_PHASE);
            }
            return () -> RefineReport.lift(report);
        });
    }

    static Request<RefineReport> reshardPlanFinishing(Family family, PlanCell plan) {
        return new Request<>(family.reshardPlanFinish(), call -> {
            call.pointer(DenseIndex.RESHARD_PLAN_FINISH_PLAN, plan.take());
            MemorySegment report = call.output(DenseIndex.RESHARD_PLAN_FINISH_REPORT, DenseIndex.REPORT);
            return () -> RefineReport.lift(report);
        });
    }

    /** Opens a cursor, which {@code adopt} wraps as the modality's own cursor type. */
    static <Cursor> Request<Cursor> cursorOpening(Family family, Key[] starts, Function<MemorySegment, Cursor> adopt) {
        return new Request<>(family.cursorInit(), call -> {
            MemorySegment opened = call.output(DenseIndex.CURSOR_INIT_CURSOR, Library.POINTER);
            if (starts.length != 0) {
                call.size(DenseIndex.CURSOR_INIT_STARTS_COUNT, starts.length);
                call.pointer(DenseIndex.CURSOR_INIT_STARTS, call.staged(starts));
            }
            return () -> adopt.apply(opened.get(Library.POINTER, 0));
        });
    }

    /** Reads what a checkpoint froze from a path or a buffer, as {@code lift} reads its metadata. */
    static <Metadata> Metadata metadata(
            Verb verb,
            @Nullable Executor executor,
            Consumer<Call> source,
            StructLayout layout,
            Function<MemorySegment, Metadata> lift) {
        return Call.block(verb, null, executor, call -> {
            source.accept(call);
            MemorySegment metadata = call.output(READ_METADATA, layout);
            return () -> lift.apply(metadata);
        });
    }

    /** One {@code usearch_<family>_metric_from_name}, which reads a spelling into an enumerator. */
    static MethodHandle nameReader(String name) {
        return Library.downcall(name, FunctionDescriptor.of(Library.ENUM, Library.BYTES_VIEW, Library.POINTER));
    }

    /**
     * Reads the code of the enumerator {@code reader} spells as {@code name}.
     *
     * @throws USearchException INVALID_ARGUMENT on a name the library does not spell.
     */
    static int fromName(MethodHandle reader, String name) {
        byte[] encoded = name.getBytes(StandardCharsets.UTF_8);
        try (java.lang.foreign.Arena memory = java.lang.foreign.Arena.ofConfined()) {
            MemorySegment view = memory.allocate(Library.BYTES_VIEW);
            Library.BYTES_DATA.set(view, 0L, memory.allocateFrom(ValueLayout.JAVA_BYTE, encoded));
            Library.BYTES_LENGTH.set(view, 0L, (long) encoded.length);
            MemorySegment code = memory.allocate(Library.ENUM);
            int status;
            try {
                status = (int) reader.invokeExact(view, code);
            } catch (Throwable failure) {
                throw Library.rethrow(failure);
            }
            if (status != Status.SUCCESS.code()) {
                throw new USearchException(Status.of(status), "no kind is spelled \"" + name + "\"");
            }
            return code.get(Library.ENUM, 0L);
        }
    }

    /** Members of {@code shared}, then {@code last}, which handles of {@code shared} address. */
    static StructLayout followedBy(StructLayout shared, MemoryLayout last) {
        MemoryLayout[] members = shared.memberLayouts()
                .toArray(new MemoryLayout[shared.memberLayouts().size() + 1]);
        members[members.length - 1] = last;
        return MemoryLayout.structLayout(members);
    }

    /** Lends every handle to the call and copies their addresses into the array C reads. */
    private static MemorySegment handles(Call call, Handle[] handles) {
        MemorySegment[] addresses = new MemorySegment[handles.length];
        for (int order = 0; order != handles.length; ++order) {
            addresses[order] = handles[order].address();
            call.lend(handles[order]);
        }
        return call.addresses(addresses);
    }

    /** {@code usearch_tape_view_t}, whose offsets this binding always spells 64 bits wide. */
    static final StructLayout TAPE_VIEW = MemoryLayout.structLayout(
                    Library.POINTER.withName("data"),
                    Library.ENUM.withName("layout"),
                    Library.padding(),
                    Library.POINTER.withName("offsets"),
                    Library.ENUM.withName("scalar"),
                    Library.padding())
            .withName("usearch_tape_view_t");

    /** The section of a metadata read naming where it reads and where the metadata lands. */
    static final long READ_SOURCE = DenseIndex.METADATA_READ.at("source");

    private static final VarHandle READ_METADATA = DenseIndex.METADATA_READ.field("metadata");

    // The key follows the completion unpadded, since sixteen bytes align to one.
    private static final StructLayout COUNTING =
            Library.opening(Library.KEY.withName("key"), Library.padding(), Library.POINTER.withName("key_stats"));

    private static final long COUNT_KEY = Library.offset(COUNTING, "key");
    private static final VarHandle COUNT_KEY_STATS = Library.field(COUNTING, "key_stats");

    /** {@code usearch_<family>_reshard_t}, the dense one without its placements. */
    private static final StructLayout RESHARDING = MemoryLayout.structLayout(
            Library.SIZE.withName("sources_count"),
            Library.POINTER.withName("sources"),
            Library.SIZE.withName("targets_count"),
            Library.POINTER.withName("targets"),
            Library.POINTER.withName("executor"),
            Library.POINTER.withName("outcome"),
            Library.ENUM.withName("completion"),
            Library.padding(),
            Library.POINTER.withName("report"),
            MemoryLayout.structLayout(Library.SIZE.withName("expansion_factor"), Library.SIZE.withName("memory_limit"))
                    .withName("refine"),
            ValueLayout.JAVA_LONG.withName("seed"),
            Library.PROGRESS,
            Library.POINTER.withName("stats"),
            Library.POINTER.withName("plan"),
            Library.POINTER.withName("phase"));

    private static final VarHandle RESHARD_SOURCES_COUNT = Library.field(RESHARDING, "sources_count");
    private static final VarHandle RESHARD_SOURCES = Library.field(RESHARDING, "sources");
    private static final VarHandle RESHARD_TARGETS_COUNT = Library.field(RESHARDING, "targets_count");
    private static final VarHandle RESHARD_TARGETS = Library.field(RESHARDING, "targets");
    private static final VarHandle RESHARD_REPORT = Library.field(RESHARDING, "report");
    private static final VarHandle RESHARD_EXPANSION_FACTOR = Library.field(RESHARDING, "refine", "expansion_factor");
    private static final VarHandle RESHARD_MEMORY_LIMIT = Library.field(RESHARDING, "refine", "memory_limit");
    private static final VarHandle RESHARD_SEED = Library.field(RESHARDING, "seed");
    private static final long RESHARD_PROGRESS = Library.offset(RESHARDING, "progress");
    private static final VarHandle RESHARD_STATS = Library.field(RESHARDING, "stats");
    private static final VarHandle RESHARD_PLAN = Library.field(RESHARDING, "plan");
    private static final VarHandle RESHARD_PHASE = Library.field(RESHARDING, "phase");
}
