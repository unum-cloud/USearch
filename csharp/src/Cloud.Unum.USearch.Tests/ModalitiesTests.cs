using System.Text;

namespace Cloud.Unum.USearch.Tests;

/// <summary>
/// The verbs every modality beside the dense one shares, ported from <c>test/modalities.py</c>: init,
/// refinement and its plans, resharding and its refusals, and checkpoints. Each concrete class below
/// runs them over one modality. A list holding something other than an index, which Python refuses
/// with a <c>TypeError</c>, cannot be written here, since the arrays are typed.
/// </summary>
/// <typeparam name="TIndex">The modality's index handle.</typeparam>
/// <typeparam name="TRefinePlan">The plan its refinement takes.</typeparam>
/// <typeparam name="TReshardPlan">The plan its reshard takes.</typeparam>
public abstract class ModalityTests<TIndex, TRefinePlan, TReshardPlan> : IDisposable
    where TIndex : class, IDisposable
    where TRefinePlan : class, new()
    where TReshardPlan : class, new()
{
    protected readonly Executor Pool = Executor.ForRam(threads: 1);
    protected readonly Arena Arena = new();

    /// <summary>Releases the executor every test in this class fans out over, and the arena its
    /// blocking calls land in.</summary>
    public void Dispose()
    {
        this.Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the arena, then the executor, when called from <see cref="Dispose()"/>.</summary>
    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            this.Arena.Dispose();
            this.Pool.Dispose();
        }
    }

    protected abstract TIndex InitRam();

    protected abstract TIndex InitVram();

    /// <summary>Inserts one random entry under each key, drawn from <paramref name="seed"/>.</summary>
    protected abstract void Insert(TIndex index, Executor executor, Key[] keys, int seed);

    protected abstract (ulong Size, string CpuClass, string GpuClass) Describe(TIndex index);

    protected abstract bool Contains(TIndex index, Key key);

    protected abstract RefineReport Refine(TIndex index, Executor? executor, TRefinePlan? plan, Progress? progress);

    protected abstract RefineReport RefinePlanFinish(TIndex index, TRefinePlan plan);

    protected abstract PhaseProgress RefinePhase(TRefinePlan plan);

    protected abstract RefineReport Reshard(
        TIndex index, TIndex[] sources, TIndex[] targets, ulong seed, Progress? progress, TReshardPlan? plan);

    protected abstract Deferred<RefineReport> ReshardPlanFinishDeferred(TIndex index, TReshardPlan plan);

    protected abstract PhaseProgress ReshardPhase(TReshardPlan plan);

    protected abstract void Save(TIndex index, string path);

    protected abstract TIndex LoadRam(string path);

    protected abstract byte[] SaveBuffer(TIndex index);

    protected abstract TIndex LoadRamBuffer(byte[] buffer);

    protected static Key[] KeysFrom(ulong first, int count) =>
        [.. Enumerable.Range(0, count).Select(offset => (Key)(first + (ulong)offset))];

    /// <summary>A host index holding <paramref name="count"/> random entries under consecutive keys
    /// from <paramref name="firstKey"/>.</summary>
    protected TIndex Build(int count, ulong firstKey = 0, int seed = 0)
    {
        TIndex index = this.InitRam();
        if (count != 0)
        {
            this.Insert(index, this.Pool, KeysFrom(firstKey, count), seed);
        }

        return index;
    }

    protected ulong Size(TIndex index) => this.Describe(index).Size;

    /// <summary>Runs a call whose progress answers false, which the library may finish before it hears.</summary>
    private static void Stopped(Action call)
    {
        try
        {
            call();
        }
        catch (USearchException refused) when (refused.Status == Status.Cancelled)
        {
        }
    }

    #region Construction

    [Fact]
    public void InitRamStartsEmptyOnACpuClass()
    {
        using TIndex index = this.InitRam();
        (ulong size, string cpuClass, string gpuClass) = this.Describe(index);

        Assert.Equal(0UL, size);
        Assert.NotEqual("unknown", cpuClass);
        Assert.Equal("unknown", gpuClass);
    }

    [Fact]
    public void InitVramSurfacesTheLibraryStatus()
    {
        TIndex index;
        try
        {
            index = this.InitVram();
        }
        catch (USearchException refused)
        {
            Assert.NotEqual(Status.Success, refused.Status);
            Assert.NotEqual(Status.Deferred, refused.Status);
            return;
        }

        using (index)
        {
            Assert.NotEqual("unknown", this.Describe(index).GpuClass);
        }
    }

    #endregion Construction

    #region Refinement

    [Fact]
    public void RefineReportsARoundOverEveryEntry()
    {
        using TIndex index = this.Build(64);

        RefineReport report = this.Refine(index, this.Pool, null, null);

        Assert.True(report.Changed <= report.Reselected);
        Assert.Equal(64UL, report.Reachability.Audited);
        Assert.Equal(64UL, this.Size(index));
        Assert.True(this.Contains(index, 17UL));
    }

    [Fact]
    public void RefinePlanResumesAndFinishes()
    {
        using TIndex index = this.Build(64);
        TRefinePlan plan = new();

        Stopped(() => this.Refine(index, this.Pool, plan, (_, _) => false));
        Assert.NotEqual(Phase.Done, this.RefinePhase(plan).Phase);
        Assert.True(this.RefinePhase(plan).Completed <= this.RefinePhase(plan).Total);

        this.Refine(index, this.Pool, plan, null);
        Assert.Equal(Phase.Done, this.RefinePhase(plan).Phase);
        Assert.Equal(64UL, this.RefinePlanFinish(index, plan).Reachability.Audited);
        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => this.RefinePlanFinish(index, plan)).Status);
    }

    [Fact]
    public void RefinePlanNeedsAnExecutor()
    {
        using TIndex index = this.Build(0);

        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => this.Refine(index, null, new TRefinePlan(), null)).Status);
    }

    #endregion Refinement

    #region Composition

    [Fact]
    public void ReshardGathersEverySourceIntoTheTarget()
    {
        using TIndex first = this.Build(32);
        using TIndex second = this.Build(32, firstKey: 32, seed: 1);
        using TIndex third = this.Build(32, firstKey: 64, seed: 2);

        this.Reshard(first, [first, second, third], [first], 0, null, null);

        Assert.Equal(96UL, this.Size(first));
        Assert.True(this.Contains(first, 40UL));
        Assert.True(this.Contains(first, 80UL));
        Assert.Equal(0UL, this.Size(second) + this.Size(third));
    }

    [Fact]
    public void ReshardCutsOneSourceIntoSeveralTargets()
    {
        using TIndex source = this.Build(64);
        using TIndex left = this.InitRam();
        using TIndex right = this.InitRam();

        this.Reshard(source, [source], [left, right], 7, null, null);

        Assert.Equal(64UL, this.Size(left) + this.Size(right));
        Assert.NotEqual(0UL, this.Size(left));
        Assert.NotEqual(0UL, this.Size(right));
        Assert.Equal(0UL, this.Size(source));
    }

    [Fact]
    public void ReshardCutsThroughAPlanAndADeferredFinish()
    {
        using TIndex source = this.Build(64);
        using TIndex left = this.InitRam();
        using TIndex right = this.InitRam();
        TReshardPlan plan = new();

        Stopped(() => this.Reshard(source, [source], [left, right], 0, (_, _) => false, plan));
        Assert.Equal(64UL, this.Size(left) + this.Size(right));
        Assert.NotEqual(Phase.Done, this.ReshardPhase(plan).Phase);

        this.Reshard(source, [source], [left, right], 0, null, plan);
        Assert.Equal(Phase.Done, this.ReshardPhase(plan).Phase);
        using Deferred<RefineReport> finished = this.ReshardPlanFinishDeferred(source, plan);
        finished.Wait();
    }

    [Fact]
    public void ReshardRecutOfOneHandleIsUnsupported()
    {
        using TIndex index = this.Build(16);

        Assert.Equal(
            Status.Unsupported,
            Assert.Throws<USearchException>(() => this.Reshard(index, [index], [index], 0, null, null)).Status);
        Assert.Equal(16UL, this.Size(index));
    }

    [Fact]
    public void ReshardRefusesMalformedListsAndMovesNothing()
    {
        using TIndex first = this.Build(8);
        using TIndex second = this.Build(8, firstKey: 8, seed: 1);

        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => this.Reshard(first, [], [first], 0, null, null)).Status);
        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => this.Reshard(first, [first, first], [first], 0, null, null)).Status);
        Assert.Equal(
            Status.ConfigMismatch,
            Assert.Throws<USearchException>(() => this.Reshard(first, [first], [second], 0, null, null)).Status);
        Assert.Equal(8UL, this.Size(first));
        Assert.Equal(8UL, this.Size(second));
    }

    #endregion Composition

    #region Serialization

    [Fact]
    public void CheckpointsRoundTripThroughAPath()
    {
        string path = Path.Combine(Path.GetTempPath(), $"usearch-{Guid.NewGuid():N}.safegraphs");
        try
        {
            using (TIndex index = this.Build(32))
            {
                this.Save(index, path);
            }

            using TIndex restored = this.LoadRam(path);
            Assert.Equal(32UL, this.Size(restored));
            Assert.True(this.Contains(restored, 17UL));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CheckpointsRoundTripThroughABuffer()
    {
        using TIndex index = this.Build(32);

        using TIndex restored = this.LoadRamBuffer(this.SaveBuffer(index));

        Assert.Equal(32UL, this.Size(restored));
        Assert.True(this.Contains(restored, 17UL));
        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => this.LoadRamBuffer([])).Status);
    }

    #endregion Serialization
}

/// <summary>The shared modality verbs over byte strings, and the sequence verbs of their own.</summary>
public sealed class SequenceIndexTests : ModalityTests<SequenceIndex, SequenceRefinePlan, SequenceReshardPlan>
{
    protected override SequenceIndex InitRam() => SequenceIndex.InitRam(SequenceOptions.Default);

    protected override SequenceIndex InitVram() => SequenceIndex.InitVram(SequenceOptions.Default);

    protected override void Insert(SequenceIndex index, Executor executor, Key[] keys, int seed)
    {
        Random random = new(seed);
        byte[][] words = new byte[keys.Length][];
        for (int word = 0; word < words.Length; word++)
        {
            words[word] = new byte[random.Next(8, 17)];
            for (int letter = 0; letter < words[word].Length; letter++)
            {
                words[word][letter] = (byte)random.Next('a', 'z' + 1);
            }
        }

        index.On(executor).Insert(keys, Tape.Bytes(words));
    }

    protected override (ulong Size, string CpuClass, string GpuClass) Describe(SequenceIndex index)
    {
        SequenceIndexDescription description = index.Describe();
        return (description.Size, description.CpuClass, description.GpuClass);
    }

    protected override bool Contains(SequenceIndex index, Key key) => index.Contains(key);

    protected override RefineReport Refine(
        SequenceIndex index, Executor? executor, SequenceRefinePlan? plan, Progress? progress) =>
        executor is null
            ? index.Refine(progress: progress, plan: plan)
            : index.On(executor).Refine(progress: progress, plan: plan);

    protected override RefineReport RefinePlanFinish(SequenceIndex index, SequenceRefinePlan plan) =>
        index.On(this.Pool).RefinePlanFinish(plan);

    protected override PhaseProgress RefinePhase(SequenceRefinePlan plan) => plan.Phase;

    protected override RefineReport Reshard(
        SequenceIndex index, SequenceIndex[] sources, SequenceIndex[] targets, ulong seed, Progress? progress,
        SequenceReshardPlan? plan) =>
        index.On(this.Pool).Reshard(sources, targets, seed, progress: progress, plan: plan);

    protected override Deferred<RefineReport> ReshardPlanFinishDeferred(
        SequenceIndex index, SequenceReshardPlan plan) =>
        index.Deferred(this.Pool).ReshardPlanFinish(plan);

    protected override PhaseProgress ReshardPhase(SequenceReshardPlan plan) => plan.Phase;

    protected override void Save(SequenceIndex index, string path) => index.On(this.Pool).Save(path);

    protected override SequenceIndex LoadRam(string path) => SequenceIndex.LoadRam(path, executor: this.Pool);

    protected override byte[] SaveBuffer(SequenceIndex index) =>
        index.On(this.Pool).SaveBuffer(this.Arena).Bytes.ToArray();

    protected override SequenceIndex LoadRamBuffer(byte[] buffer) =>
        SequenceIndex.LoadRamBuffer(buffer, executor: this.Pool);

    [Fact]
    public void InitRamKeepsTheMetric()
    {
        using SequenceIndex index = SequenceIndex.InitRam(new SequenceOptions { Metric = SequenceMetric.DnaOverlap });

        Assert.Equal(SequenceMetric.DnaOverlap, index.Describe().Metadata.Metric);
    }

    [Fact]
    public void ContainmentTakesNoCosts()
    {
        USearchException refused = Assert.Throws<USearchException>(() => SequenceIndex.InitRam(new SequenceOptions
        {
            Metric = SequenceMetric.Utf8Containment, Costs = new SequenceCosts(new sbyte[256 * 256], 1),
        }));

        Assert.Equal(Status.ConfigMismatch, refused.Status);
    }

    /// <summary>Scores each of A, C, G and T against itself at <paramref name="self"/>, and A and G
    /// against each other at <paramref name="near"/>.</summary>
    private static sbyte[] NucleotideSimilarities(sbyte self, sbyte near)
    {
        sbyte[] similarities = new sbyte[256 * 256];
        foreach (char nucleotide in "ACGT")
        {
            similarities[nucleotide * 256 + nucleotide] = self;
        }

        similarities['A' * 256 + 'G'] = near;
        similarities['G' * 256 + 'A'] = near;
        return similarities;
    }

    [Fact]
    public void OverlapFindsTheReverseComplement()
    {
        using SequenceIndex index = SequenceIndex.InitRam(new SequenceOptions { Metric = SequenceMetric.DnaOverlap });
        index.Insert(KeysFrom(1, 3), Tape.Utf8(
            "ACGTTGCAAGGCTTAACCGGTTACGATCGA", "GGCTTAACCGGTTACGATCGATTTGCAGCA", "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"));

        Matches found = index.Search(Tape.Utf8("TGCTGCAAATCGATCGTAACCGGTTAAGCC"), wanted: 2, this.Arena);

        Assert.Equal(Key.FromUInt64(2), found[0].Keys[0]);
        Assert.Equal(0.0, found[0].Distances[0]);
    }

    [Fact]
    public void AlignmentRanksByCosts()
    {
        using SequenceIndex index = SequenceIndex.InitRam(new SequenceOptions
        {
            Metric = SequenceMetric.ProteinAlignment,
            Costs = new SequenceCosts(NucleotideSimilarities(4, 3), 4),
        });
        index.Insert(KeysFrom(1, 2), Tape.Utf8("GCGTTGCAACGT", "CCGTTGCAACGT"));

        Matches found = index.Search(Tape.Utf8("ACGTTGCAACGT"), wanted: 2, this.Arena);

        Assert.Equal(Key.FromUInt64(1), found[0].Keys[0]);
        Assert.True(found[0].Distances[0] < found[0].Distances[1]);
    }

    [Fact]
    public void ContainmentFindsTheHoldingText()
    {
        using SequenceIndex index = SequenceIndex.InitRam(
            new SequenceOptions { Metric = SequenceMetric.Utf8Containment });
        index.Insert(KeysFrom(1, 2), Tape.Utf8(
            "the quick brown fox jumps over the lazy dog", "a journey of a thousand miles begins with a single step"));

        Matches found = index.Search(Tape.Utf8("BROWN FOX jumps"), wanted: 2, this.Arena);

        Assert.Equal(Key.FromUInt64(1), found[0].Keys[0]);
    }

    [Fact]
    public void SearchFindsTheInsertedEntry()
    {
        using SequenceIndex index = this.InitRam();
        index.Insert(KeysFrom(1, 3), Tape.Utf8("kitten", "sitting", "mitten"));

        Matches found = index.Search(Tape.Utf8("sitting"), wanted: 1, this.Arena);

        Assert.Equal(3UL, index.Describe().Size);
        Assert.Equal(Key.FromUInt64(2), found[0].Keys[0]);
        Assert.Equal(0.0, found[0].Distances[0]);
    }

    [Fact]
    public void GetAndCountReadTheStoredText()
    {
        using SequenceIndex index = SequenceIndex.InitRam(
            new SequenceOptions { Metric = SequenceMetric.Utf8Containment });
        index.Insert(KeysFrom(1, 3), Tape.Utf8("kitten", "sitting", "mitten"));

        TapeEntries entries = index.Get(2UL, this.Arena);

        Assert.Equal(1, entries.Count);
        Assert.Equal("sitting", Encoding.UTF8.GetString(entries.Entry<byte>(0)));
        Assert.Equal(new SequenceKeyStats(1, 7, 7), index.Count(2UL));
        Assert.Equal(0, index.Get(9UL, this.Arena).Count);
    }

    /// <summary>Mirrors the example under <c>## Sequence Indexes</c> in <c>csharp/README.md</c>; the
    /// arena is this class's own.</summary>
    [Fact]
    public void ReadmeExampleFindsTheHoldingText()
    {
        Arena arena = this.Arena;

        using SequenceIndex index = SequenceIndex.InitRam(
            new SequenceOptions { Metric = SequenceMetric.Utf8Containment });
        Key[] keys = [1UL, 2UL, 3UL];
        index.Insert(keys, Tape.Utf8("The quick brown fox", "A lazy dog", "Pack my box"));

        Matches found = index.Search(Tape.Utf8("BROWN FOX"), wanted: 1, arena);
        Key nearest = found[0].Keys[0];
        double distance = found[0].Distances[0];

        Assert.Equal(Key.FromUInt64(1), nearest);
        Assert.InRange(distance, 0.0, 1.0 / 32768);
    }
}

/// <summary>The shared modality verbs over point clouds, and the shape verbs of their own.</summary>
public sealed class ShapesIndexTests : ModalityTests<ShapesIndex, ShapesRefinePlan, ShapesReshardPlan>
{
    protected override ShapesIndex InitRam() => ShapesIndex.InitRam(ShapesOptions.Default);

    protected override ShapesIndex InitVram() => ShapesIndex.InitVram(ShapesOptions.Default);

    private static float[] Cloud(Random random, int points) =>
        [.. Enumerable.Range(0, points * 3).Select(_ => random.NextSingle())];

    protected override void Insert(ShapesIndex index, Executor executor, Key[] keys, int seed)
    {
        Random random = new(seed);
        float[][] clouds = [.. keys.Select(_ => Cloud(random, random.Next(4, 13)))];
        index.On(executor).Insert(keys, Tape.Points(clouds));
    }

    protected override (ulong Size, string CpuClass, string GpuClass) Describe(ShapesIndex index)
    {
        ShapesIndexDescription description = index.Describe();
        return (description.Size, description.CpuClass, description.GpuClass);
    }

    protected override bool Contains(ShapesIndex index, Key key) => index.Contains(key);

    protected override RefineReport Refine(
        ShapesIndex index, Executor? executor, ShapesRefinePlan? plan, Progress? progress) =>
        executor is null
            ? index.Refine(progress: progress, plan: plan)
            : index.On(executor).Refine(progress: progress, plan: plan);

    protected override RefineReport RefinePlanFinish(ShapesIndex index, ShapesRefinePlan plan) =>
        index.On(this.Pool).RefinePlanFinish(plan);

    protected override PhaseProgress RefinePhase(ShapesRefinePlan plan) => plan.Phase;

    protected override RefineReport Reshard(
        ShapesIndex index, ShapesIndex[] sources, ShapesIndex[] targets, ulong seed, Progress? progress,
        ShapesReshardPlan? plan) =>
        index.On(this.Pool).Reshard(sources, targets, seed, progress: progress, plan: plan);

    protected override Deferred<RefineReport> ReshardPlanFinishDeferred(
        ShapesIndex index, ShapesReshardPlan plan) =>
        index.Deferred(this.Pool).ReshardPlanFinish(plan);

    protected override PhaseProgress ReshardPhase(ShapesReshardPlan plan) => plan.Phase;

    protected override void Save(ShapesIndex index, string path) => index.On(this.Pool).Save(path);

    protected override ShapesIndex LoadRam(string path) => ShapesIndex.LoadRam(path, executor: this.Pool);

    protected override byte[] SaveBuffer(ShapesIndex index) =>
        index.On(this.Pool).SaveBuffer(this.Arena).Bytes.ToArray();

    protected override ShapesIndex LoadRamBuffer(byte[] buffer) =>
        ShapesIndex.LoadRamBuffer(buffer, executor: this.Pool);

    [Fact]
    public void InitRamKeepsTheMetric()
    {
        using ShapesIndex index = ShapesIndex.InitRam(ShapesOptions.Default with { Metric = ShapesMetric.Hausdorff });

        Assert.Equal(ShapesMetric.Hausdorff, index.Describe().Metadata.Metric);
    }

    [Fact]
    public void SearchFindsTheInsertedEntry()
    {
        Random random = new(42);
        float[][] clouds = [Cloud(random, 5), Cloud(random, 9), Cloud(random, 13)];
        using ShapesIndex index = this.InitRam();
        index.Insert(KeysFrom(1, 3), Tape.Points(clouds));

        Matches found = index.Search(Tape.Points(clouds[2]), wanted: 1, this.Arena);

        Assert.Equal(3UL, index.Describe().Size);
        Assert.Equal(27UL, index.Describe().TotalPoints);
        Assert.Equal(Key.FromUInt64(3), found[0].Keys[0]);
    }

    [Fact]
    public void GetAndCountReadThePointsUnderAKey()
    {
        Random random = new(42);
        float[][] clouds = [Cloud(random, 5), Cloud(random, 9), Cloud(random, 13)];
        using ShapesIndex index = this.InitRam();
        index.Insert(KeysFrom(1, 3), Tape.Points(clouds));

        TapeEntries entries = index.Get<F32>(3UL, this.Arena);

        Assert.Equal(1, entries.Count);
        float[] restored = entries.Entry<float>(0).ToArray();
        Assert.Equal(clouds[2].Length, restored.Length);
        for (int coordinate = 0; coordinate < restored.Length; coordinate++)
        {
            Assert.Equal(clouds[2][coordinate], restored[coordinate], 1e-5f); // The inserted pose, re-rotated
        }
        Assert.Equal(new ShapesKeyStats(1, 13, 13), index.Count(3UL));
        Assert.Equal(0, index.Get<F32>(9UL, this.Arena).Count);
    }

    /// <summary>Mirrors the example under <c>## Shape Indexes</c> in <c>csharp/README.md</c>; the arena
    /// is this class's own.</summary>
    [Fact]
    public void ReadmeExampleFindsTheNearerMolecule()
    {
        Arena arena = this.Arena;

        using ShapesIndex index = ShapesIndex.InitRam(ShapesOptions.Default);
        Key[] keys = [1UL, 2UL];
        float[] pair = [0, 0, 0, 1.1f, 0, 0];
        float[] triple = [0, 0, 0, 1.5f, 0, 0, 0, 1.5f, 0];
        index.Insert(keys, Tape.Points(pair, triple));

        float[] query = [0, 0, 0, 1.0f, 0, 0];
        Matches found = index.Search(Tape.Points(query), wanted: 1, arena);
        Key nearest = found[0].Keys[0];

        Assert.Equal(Key.FromUInt64(1), nearest);
    }
}

/// <summary>The shared modality verbs over observations, and the geospatial verbs of their own.</summary>
public sealed class GeospatialIndexTests
    : ModalityTests<GeospatialIndex, GeospatialRefinePlan, GeospatialReshardPlan>
{
    protected override GeospatialIndex InitRam() => GeospatialIndex.InitRam(new GeospatialOptions());

    protected override GeospatialIndex InitVram() => GeospatialIndex.InitVram(new GeospatialOptions());

    protected override void Insert(GeospatialIndex index, Executor executor, Key[] keys, int seed)
    {
        Random random = new(seed);
        GeospatialObservation[] batch =
        [
            .. keys.Select(_ => GeospatialObservation.At(
                (random.NextDouble() * 120) - 60, (random.NextDouble() * 360) - 180)),
        ];
        index.On(executor).Insert(keys, batch);
    }

    protected override (ulong Size, string CpuClass, string GpuClass) Describe(GeospatialIndex index)
    {
        GeospatialIndexDescription description = index.Describe();
        return (description.Size, description.CpuClass, description.GpuClass);
    }

    protected override bool Contains(GeospatialIndex index, Key key) => index.Contains(key);

    protected override RefineReport Refine(
        GeospatialIndex index, Executor? executor, GeospatialRefinePlan? plan, Progress? progress) =>
        executor is null
            ? index.Refine(progress: progress, plan: plan)
            : index.On(executor).Refine(progress: progress, plan: plan);

    protected override RefineReport RefinePlanFinish(GeospatialIndex index, GeospatialRefinePlan plan) =>
        index.On(this.Pool).RefinePlanFinish(plan);

    protected override PhaseProgress RefinePhase(GeospatialRefinePlan plan) => plan.Phase;

    protected override RefineReport Reshard(
        GeospatialIndex index, GeospatialIndex[] sources, GeospatialIndex[] targets, ulong seed, Progress? progress,
        GeospatialReshardPlan? plan) =>
        index.On(this.Pool).Reshard(sources, targets, seed, progress: progress, plan: plan);

    protected override Deferred<RefineReport> ReshardPlanFinishDeferred(
        GeospatialIndex index, GeospatialReshardPlan plan) =>
        index.Deferred(this.Pool).ReshardPlanFinish(plan);

    protected override PhaseProgress ReshardPhase(GeospatialReshardPlan plan) => plan.Phase;

    protected override void Save(GeospatialIndex index, string path) => index.On(this.Pool).Save(path);

    protected override GeospatialIndex LoadRam(string path) => GeospatialIndex.LoadRam(path, executor: this.Pool);

    protected override byte[] SaveBuffer(GeospatialIndex index) =>
        index.On(this.Pool).SaveBuffer(this.Arena).Bytes.ToArray();

    protected override GeospatialIndex LoadRamBuffer(byte[] buffer) =>
        GeospatialIndex.LoadRamBuffer(buffer, executor: this.Pool);

    /// <summary>The integer identities of a run of keys, in ascending order.</summary>
    private static ulong[] Identities(ReadOnlySpan<Key> keys)
    {
        ulong[] identities = new ulong[keys.Length];
        for (int key = 0; key < keys.Length; key++)
        {
            Assert.True(keys[key].TryGetUInt64(out identities[key]));
        }

        Array.Sort(identities);
        return identities;
    }

    private static readonly GeospatialObservation[] Capitals =
    [
        GeospatialObservation.At(48.8566, 2.3522),
        GeospatialObservation.At(52.52, 13.405),
        GeospatialObservation.At(40.7128, -74.006),
    ];

    [Fact]
    public void SearchFindsTheInsertedEntry()
    {
        using GeospatialIndex index = this.InitRam();
        index.Insert(KeysFrom(1, 3), Capitals);

        Matches found = index.Search(Capitals.AsMemory(1, 1), wanted: 1, this.Arena, GeospatialRelation.Near);

        Assert.Equal(3UL, index.Describe().Size);
        Assert.Equal(Key.FromUInt64(2), found[0].Keys[0]);
        Assert.Equal(0.0, found[0].Distances[0]);
    }

    [Fact]
    public void WithinReportsOnlyTheEntriesInsideTheRadius()
    {
        GeospatialObservation[] batch =
        [
            GeospatialObservation.At(48.8566, 2.3522),
            GeospatialObservation.At(48.8656, 2.3522),
            GeospatialObservation.At(52.52, 13.405),
        ];
        using GeospatialIndex index = this.InitRam();
        index.Insert(KeysFrom(1, 3), batch);

        DenseProducts report = index.Within(batch.AsMemory(0, 1), this.Arena, new GeospatialTolerance(2000.0));

        Assert.Equal(new ulong[] { 1, 2 }, Identities(report.Keys(0)));
    }

    [Fact]
    public void GetAndCountReadTheObservationsUnderAKey()
    {
        using GeospatialIndex index = this.InitRam();
        index.Insert(KeysFrom(1, 3), Capitals);

        GeospatialObservations stored = index.Get(2UL, this.Arena);

        Assert.Equal(1, stored.Count);
        Assert.Equal(52.52, stored.Observations[0].LatitudeDegrees, 3);
        Assert.Equal(13.405, stored.Observations[0].LongitudeDegrees, 3);
        Assert.Equal(1UL, index.Count(2UL));
        Assert.Equal(0, index.Get(9UL, this.Arena).Count);
    }

    /// <summary>Mirrors the example under <c>## Geospatial Indexes</c> in <c>csharp/README.md</c>; the
    /// arena is this class's own.</summary>
    [Fact]
    public void ReadmeExampleFindsTheStopsWithinTheRadius()
    {
        Arena arena = this.Arena;

        using GeospatialIndex index = GeospatialIndex.InitRam(new GeospatialOptions());
        GeospatialObservation[] stops =
        [
            GeospatialObservation.At(51.5308, -0.1238),
            GeospatialObservation.At(51.5320, -0.1233),
            GeospatialObservation.At(51.5033, -0.1196),
        ];
        Key[] keys = [1UL, 2UL, 3UL];
        index.Insert(keys, stops);

        DenseProducts nearby = index.Within(stops.AsMemory(0, 1), arena, new GeospatialTolerance(Metres: 500));
        ReadOnlySpan<Key> found = nearby.Keys(0);
        ReadOnlySpan<double> chords = nearby.Scores(0);

        Assert.Equal(new ulong[] { 1, 2 }, Identities(found));
        Assert.All(chords.ToArray(), chord => Assert.InRange(chord, 0.0, 500.0));
    }
}
