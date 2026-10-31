using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Cloud.Unum.USearch.Tests;

/// <summary>
/// The boundary of the binding: marshalling of the argument structs, key byte order, stride
/// arithmetic, handle ownership, arena lifetimes, and deferred calls. That vector search works is
/// the C library's job, and is not re-tested here.
/// </summary>
public sealed class DenseIndexTests : IDisposable
{
    private const int Dimensions = 4;

    private readonly Executor _pool = Executor.ForRam(threads: 1);
    private readonly Arena _arena = new();

    /// <summary>Releases the executor every test in this class fans out over, and the arena its
    /// blocking calls land in.</summary>
    public void Dispose()
    {
        this._arena.Dispose();
        this._pool.Dispose();
    }

    private static DenseOptions SmallOptions() => DenseOptions.Default with
    {
        Dimensions = Dimensions,
        Metric = DenseMetric.Euclidean,
        Quantization = Scalar.BF16,
        Connectivity = 16,
    };

    private static DenseIndex SmallIndex()
    {
        DenseIndex index = DenseIndex.InitRam(SmallOptions());
        index.Reserve(64);
        return index;
    }

    private static float[] RowOf(float first) => [first, first + 1f, first + 2f, first + 3f];

    private static RowsView Rows(float[] values) => RowsView.RowsOf<float>(values, Dimensions);

    private static Key[] KeysOf(params ulong[] identities) => [.. identities.Select(identity => (Key)identity)];

    #region Layouts

    [Fact]
    public void NativeStructsMatchTheirLayouts()
    {
        int word = IntPtr.Size;
        Assert.Equal(2 * word, (int)Marshal.OffsetOf<NativeRamOptions>(nameof(NativeRamOptions.ShardCapacity)));
        Assert.Equal(3 * word, (int)Marshal.OffsetOf<NativeRamOptions>(nameof(NativeRamOptions.CpuClass)));
        Assert.Equal(4 * word, Unsafe.SizeOf<NativeRamOptions>());
        Assert.Equal(3 * word, (int)Marshal.OffsetOf<NativeVramOptions>(nameof(NativeVramOptions.GpuClass)));
        Assert.Equal(4 * word, Unsafe.SizeOf<NativeVramOptions>());
        Assert.Equal(2 * word, (int)Marshal.OffsetOf<NativeNvmeOptions>(nameof(NativeNvmeOptions.CpuClass)));
        Assert.Equal(3 * word, Unsafe.SizeOf<NativeNvmeOptions>());
        Assert.Equal(7 * word, (int)Marshal.OffsetOf<NativeExecutorForNvme>(nameof(NativeExecutorForNvme.NvmeClass)));
        Assert.Equal(3 * word, Unsafe.SizeOf<NativeDenseTraversalStats>());
        int cpuClass = (int)Marshal.OffsetOf<NativeDenseDescription>(nameof(NativeDenseDescription.CpuClass));
        Assert.Equal(Unsafe.SizeOf<NativeDenseMetadata>() + 3 * word, cpuClass);
        Assert.Equal(cpuClass + 4, (int)Marshal.OffsetOf<NativeDenseDescription>(nameof(NativeDenseDescription.GpuClass)));
        Assert.Equal(cpuClass + 8,
                     (int)Marshal.OffsetOf<NativeDenseDescription>(nameof(NativeDenseDescription.ShardsCount)));
        Assert.Equal(cpuClass + 8 + word, Unsafe.SizeOf<NativeDenseDescription>());
        Assert.True(Unsafe.SizeOf<NativeDenseDescription>() <= 12 * sizeof(ulong), "the report slots hold it");
        Assert.True(Unsafe.SizeOf<SampleReports>() <= 12 * sizeof(ulong), "the report slots hold a sample's views");
        // The traversal enum packs right after the float scale, with no padding between them.
        Assert.Equal((int)Marshal.OffsetOf<NativeDenseAttend>(nameof(NativeDenseAttend.ScoreScale)) + 4,
                     (int)Marshal.OffsetOf<NativeDenseAttend>(nameof(NativeDenseAttend.Traversal)));
        Assert.Equal(4 * word, (int)Marshal.OffsetOf<NativeDenseCursorInit>(nameof(NativeDenseCursorInit.Cursor)));
    }

    #endregion Layouts

    #region Keys

    [Fact]
    public void KeyRendersIntegerIdentitiesInTheLowBytes()
    {
        Key key = Key.FromUInt64(42);

        Assert.Equal("00000000-0000-0000-0000-00000000002a", key.ToString());
        Assert.Equal(0x2a, key[15]);
        Assert.Equal(0, key[0]);
        Assert.True(key.TryGetUInt64(out ulong identity));
        Assert.Equal(42UL, identity);
    }

    [Fact]
    public void KeyParsesTheRenderingItPrints()
    {
        Key wide = Key.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        Assert.Equal("0123abcd-4567-89ef-fedc-ba9876543210", wide.ToString());
        Assert.Equal(0x01, wide[0]);
        Assert.Equal(0x10, wide[15]);
        Assert.False(wide.TryGetUInt64(out _));
        Assert.Equal(wide, Key.Parse("0123ABCD456789EFFEDCBA9876543210"));
        Assert.False(Key.TryParse("0123abcd", out _));
        Assert.False(Key.TryParse("0123abcd-4567-89ef-fedc-ba98765432zz", out _));
    }

    [Fact]
    public void KeyEqualityAgreesWithHashing()
    {
        Key first = 7UL;
        Key same = Key.FromUInt64(7);
        Key other = Key.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        Assert.True(first == same);
        Assert.True(first != other);
        Assert.Equal(first.GetHashCode(), same.GetHashCode());
        Assert.NotEqual(first.GetHashCode(), other.GetHashCode());
        Assert.Single(new HashSet<Key> { first, same });
    }

    [Fact]
    public void KeysCrossTheBoundaryIntact()
    {
        using DenseIndex index = SmallIndex();
        Key wide = Key.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        index.On(this._pool).Insert(new[] { wide }, Rows(RowOf(1f)));
        Assert.Equal(1UL, index.Count(wide));
        Assert.True(index.Contains(wide));
        Assert.False(index.Contains(Key.FromUInt64(1)));

        Matches found = index.On(this._pool).Search(Rows(RowOf(1f)), wanted: 1, this._arena);
        Assert.Equal(wide, found[0].Keys[0]);
    }

    #endregion Keys

    #region Names

    [Fact]
    public void ClosedSetsAreSpelledByTheLibraryTables()
    {
        Assert.Equal("shape-mismatch", Status.ShapeMismatch.Name());
        Assert.Equal("deferred", Status.Deferred.Name());
        Assert.Equal("unrecognized", ((Status)99).Name());
        Assert.Equal("f32", Scalar.F32.Name());
        Assert.Equal("i32", Scalar.I32.Name());
        Assert.Equal("spinning", Mood.Spinning.Name());
        Assert.Equal("inner", DenseMetric.Inner.Name());
        Assert.Equal("vram", Residency.Vram.Name());
        Assert.Equal("scan", Traversal.Scan.Name());

        Assert.True(Names.TryParse("invalid-argument", out Status status));
        Assert.Equal(Status.InvalidArgument, status);
        Assert.True(Names.TryParse("parked", out Mood mood));
        Assert.Equal(Mood.Parked, mood);
        Assert.True(Names.TryParse("u1x8", out Scalar scalar));
        Assert.Equal(Scalar.U1x8, scalar);
        Assert.False(Names.TryParse("lukewarm", out Mood _));
        Assert.False(Names.TryParse("cosine", out DenseMetric _));
        Assert.True(Names.TryParse("nvme", out Residency residency));
        Assert.Equal(Residency.Nvme, residency);
        Assert.True(Names.TryParse("walk", out Traversal traversal));
        Assert.Equal(Traversal.Walk, traversal);
        Assert.False(Names.TryParse("exact", out Traversal _));

        USearchException failure = new(Status.ReadOnly, "detail");
        Assert.Equal("usearch: read-only: detail", failure.Message);
    }

    /// <summary>Reads <paramref name="name"/> back through <paramref name="parse"/>.</summary>
    private static int ReadBack(string name, Func<NativeBytesView, int> parse)
    {
        IntPtr data = Marshal.StringToCoTaskMemUTF8(name);
        try
        {
            return parse(new NativeBytesView { Data = data, Length = (nuint)System.Text.Encoding.UTF8.GetByteCount(name) });
        }
        finally
        {
            Marshal.FreeCoTaskMem(data);
        }
    }

    [Fact]
    public void EveryClassNameReadsBackThroughTheLibrary()
    {
        (Func<NativeBytesView, int> Parse, Func<int, IntPtr> Spell, int Last)[] classes =
        [
            (NativeMethods.usearch_cpu_class_from_name, NativeMethods.usearch_cpu_class_name, 17),
            (NativeMethods.usearch_gpu_class_from_name, NativeMethods.usearch_gpu_class_name, 12),
            (NativeMethods.usearch_nvme_class_from_name, NativeMethods.usearch_nvme_class_name, 6),
        ];
        foreach ((Func<NativeBytesView, int> parse, Func<int, IntPtr> spell, int last) in classes)
        {
            for (int code = 1; code <= last; ++code)
                Assert.Equal(code, ReadBack(Marshal.PtrToStringUTF8(spell(code)) ?? string.Empty, parse));
            Assert.Equal("unknown", Marshal.PtrToStringUTF8(spell(last + 1)));
        }
    }

    #endregion Names

    #region Constructors

    [Fact]
    public void OptionsReachTheLibraryAsTheyWereWritten()
    {
        using DenseIndex index = DenseIndex.InitRam(SmallOptions() with { ValuesPerKey = 3 }, memoryDomains: [0]);
        DenseIndexDescription description = index.Describe();

        Assert.Equal((ulong)Dimensions, description.Metadata.Dimensions);
        Assert.Equal(16UL, description.Metadata.Connectivity);
        Assert.Equal(DenseMetric.Euclidean, description.Metadata.Metric);
        Assert.Equal(Scalar.BF16, description.Metadata.Quantization);
        Assert.Equal(0UL, description.Size);
        Assert.NotEmpty(description.CpuClass);
        Assert.Equal("unknown", description.GpuClass);
    }

    [Fact]
    public void ResidenciesTheBuildLacksAreRefusedByTheirOwnConstructors()
    {
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(
                () => DenseIndex.InitRam(SmallOptions() with { Metric = (DenseMetric)9 })).Status);
        Assert.Equal(
            Status.Device,
            Assert.Throws<USearchException>(() => DenseIndex.InitVram(SmallOptions(), devices: [0])).Status);
        // A build without the NVMe tier refuses the view; one with it fails to open the absent checkpoint.
        Assert.Contains(
            Assert.Throws<USearchException>(
                () => DenseIndex.ViewNvme("checkpoint.safegraphs", paths: ["/mnt/nvme0"])).Status,
            new[] { Status.Unsupported, Status.Io });
    }

    #endregion Constructors

    #region Executors

    [Fact]
    public void ExecutorsDrainAndRefuseEveryVerbOnceDisposed()
    {
        Executor pool = Executor.ForRam(threads: 1, memoryDomains: [0], deferredLimit: 4);
        pool.Drain(Mood.Spinning);
        pool.Dispose();
        pool.Dispose();

        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => pool.Drain()).Status);

        using DenseIndex index = SmallIndex();
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(() => index.On(pool).Insert(KeysOf(1), Rows(RowOf(1f)))).Status);
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(() => index.Deferred(pool).Insert(KeysOf(1), Rows(RowOf(1f)))).Status);
    }

    [Fact]
    public void BorrowedStreamsMustPairOnePerDevice() =>
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(() => Executor.ForVram(devices: [0], streams: [1, 2])).Status);

    [Fact]
    public void TheIndexItselfRunsOnTheCallersThread()
    {
        using DenseIndex index = SmallIndex();
        index.Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));

        Matches found = index.Search(Rows(RowOf(2f)), wanted: 1, this._arena);
        Assert.Equal(Key.FromUInt64(2), found[0].Keys[0]);
        Assert.Equal(0UL, index.Refine().Changed);
    }

    #endregion Executors

    #region Deferred Calls

    [Fact]
    public void DeferredCallsStayDeferredUntilWaitedFor()
    {
        using DenseIndex index = SmallIndex();
        DenseIndexDeferredOn deferred = index.Deferred(this._pool);

        using Deferred inserted = deferred.Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));
        using Deferred<Matches> searched = deferred.Search(Rows(RowOf(2f)), wanted: 1, this._arena);
        Assert.Equal(Status.Deferred, inserted.Status);
        Assert.Equal(Status.Deferred, searched.Status);

        Matches found = searched.Wait();
        Assert.Equal(Status.Success, searched.Status);
        Assert.Equal(Status.Success, inserted.Status);
        inserted.Wait();
        Assert.Equal(Key.FromUInt64(2), found[0].Keys[0]);

        // The waited view borrows the arena the call named, so the arena's next call retires it.
        index.Get(1UL, this._arena);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => found[0].Count).Status);
    }

    [Fact]
    public async Task DeferredCallsAreAwaitableAsync()
    {
        using DenseIndex index = SmallIndex();
        index.Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));
        DenseTraversalStats stats = new();

        Deferred<Matches> searched =
            index.Deferred(this._pool).Search(Rows(RowOf(1f)), wanted: 2, this._arena, stats: stats);
        Deferred<DenseReachability> audited = index.Deferred(this._pool).Health();

        Matches found = await searched;
        DenseReachability reachability = await audited;
        Assert.Equal(Key.FromUInt64(1), found[0].Keys[0]);
        Assert.Equal(Key.FromUInt64(1), (await searched)[0].Keys[0]);
        Assert.True(stats.ComputedDistances > 0);
        Assert.Equal(2UL, reachability.Reachable);
    }

    [Fact]
    public void DeferredFailuresArriveAtTheWait()
    {
        using DenseIndex index = SmallIndex();
        RowsView narrowRows = RowsView.RowsOf<float>(new float[3], 3);

        using Deferred<Matches> mismatched = index.Deferred(this._pool).Search(narrowRows, wanted: 1, this._arena);
        USearchException failure = Assert.Throws<USearchException>(() => mismatched.Wait());
        Assert.Equal(Status.ShapeMismatch, failure.Status);
        Assert.Same(failure, Assert.Throws<USearchException>(() => mismatched.Wait()));
        Assert.Equal(Status.ShapeMismatch, mismatched.Status);
    }

    [Fact]
    public void FreeingAHandleWaitsForTheDeferredCallsNamingIt()
    {
        DenseIndex index = SmallIndex();
        index.Insert(KeysOf(1), Rows(RowOf(1f)));
        Arena arena = new();

        using Deferred<DenseVectors> got = index.Deferred(this._pool).Get(1UL, arena);
        using Deferred<DenseIndexDescription> described = index.Deferred(this._pool).Describe();
        arena.Dispose();
        index.Dispose();

        Assert.Equal(Status.Success, got.Status);
        Assert.Equal(1UL, described.Wait().Size);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => got.Wait().Row<float>(0).Length).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Describe()).Status);
    }

    [Fact]
    public void ABusyArenaRefusesOtherCallsUntilItsDeferredCallIsFinal()
    {
        using DenseIndex index = SmallIndex();
        index.Insert(KeysOf(1), Rows(RowOf(1f)));
        using Arena arena = new();

        Deferred<DenseVectors> pending = index.Deferred(this._pool).Get(1UL, arena);
        Assert.Equal(Status.Contended, Assert.Throws<USearchException>(() => index.Get(1UL, arena)).Status);
        using Deferred<DenseVectors> refused = index.Deferred(this._pool).Get(1UL, arena);
        Assert.Equal(Status.Contended, refused.Status);

        // A refused call never rewound the arena, so the busy call's view stays readable.
        Assert.Equal(RowOf(1f), pending.Wait().Row<float>(0).ToArray());
        pending.Dispose();
        Assert.Equal(RowOf(1f), index.Get(1UL, arena).Row<float>(0).ToArray());

        SubmitAndDrop(index, this._pool, arena);
        Collect();
        this._pool.Drain();
        Collect();
        Assert.Equal(RowOf(1f), index.Get(1UL, arena).Row<float>(0).ToArray());
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void SubmitAndDrop(DenseIndex index, Executor pool, Arena arena) =>
        index.Deferred(pool).Get(1UL, arena);

    private static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
    }

    #endregion Deferred Calls

    #region Scalars & Rows

    [Fact]
    public void PackedKindsPitchByTheValuesTheyHoldRatherThanTheDimensions()
    {
        Assert.Equal(256UL, Scalar.F32.RowBytes(64));
        Assert.Equal(32UL, Scalar.U4x2.RowBytes(64));
        Assert.Equal(8UL, Scalar.U1x8.RowBytes(64));
        Assert.Equal(8UL, Scalar.U1x8.RowBytes(60));

        byte[] bits = new byte[16];
        RowsView packed = RowsView.RowsOf<byte>(bits, 64, Scalar.U1x8);
        Assert.Equal(2, packed.Rows);
        Assert.Equal(8, packed.StrideBytes);
        Assert.Equal(64, packed.Dimensions);
        Assert.Equal(Scalar.U1x8, packed.Scalar);
    }

    [Fact]
    public void PackedBitSetsReachTheIndexAndComeBack()
    {
        DenseOptions options = DenseOptions.Default with
        {
            Dimensions = 64,
            Metric = DenseMetric.Hamming,
            Quantization = Scalar.U1x8,
        };
        using DenseIndex index = DenseIndex.InitRam(options);
        index.Reserve(4);

        byte[] stored = [0b0000_0001, 0, 0, 0, 0, 0, 0, 0b1000_0000];
        index.On(this._pool).Insert(KeysOf(1), RowsView.RowsOf<byte>(stored, 64, Scalar.U1x8));

        DenseVectors retrieved = index.On(this._pool).Get(1UL, this._arena, Scalar.U1x8);
        Assert.Equal(1, retrieved.Rows);
        Assert.Equal(8, retrieved.StrideBytes);
        Assert.Equal(64, retrieved.Columns);
        Assert.Equal(stored, retrieved.Row<byte>(0).ToArray());
    }

    [Fact]
    public void StridedRowsSkipThePaddingBetweenThem()
    {
        using DenseIndex index = SmallIndex();
        float[] padded = [.. RowOf(10f), 0f, 0f, .. RowOf(20f), 0f, 0f];

        RowsView rows = RowsView.StridedRowsOf<float>(
            padded, rows: 2, dimensions: Dimensions, strideBytes: 6 * sizeof(float), Scalar.F32);
        index.On(this._pool).Insert(KeysOf(1, 2), rows);

        Matches found = index.On(this._pool).Search(Rows(RowOf(20f)), wanted: 1, this._arena);
        Assert.Equal(Key.FromUInt64(2), found[0].Keys[0]);
    }

    [Fact]
    public void RowsViewsRefuseLayoutsNoSlabCouldHold()
    {
        float[] slab = new float[8];

        Assert.Equal(
            Status.UnsupportedLayout,
            Assert.Throws<USearchException>(
                () => RowsView.StridedRowsOf<float>(slab, 2, Dimensions, 8, Scalar.F32)).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => RowsView.StridedRowsOf<float>(slab, 3, Dimensions, 16, Scalar.F32)).Status);
        Assert.Equal(
            Status.UnsupportedLayout,
            Assert.Throws<USearchException>(() => RowsView.RowsOf<float>(slab, Dimensions, Scalar.F64)).Status);
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => RowsView.RowsOf<short>(new short[4], Dimensions)).Status);
    }

    #endregion Scalars & Rows

    #region Geometry

    [Fact]
    public void ZeroDimensionsAreRefusedAtInit()
    {
        Assert.Equal(
            Status.ConfigMismatch,
            Assert.Throws<USearchException>(() => DenseIndex.InitRam(DenseOptions.Default)).Status);
    }

    [Fact]
    public void FreshIndexInsertsWithoutAReservation()
    {
        using DenseIndex index = DenseIndex.InitRam(SmallOptions());

        index.On(this._pool).Insert(KeysOf(1), Rows(RowOf(1f)));
        Assert.Equal(1UL, index.Describe().Size);
        Assert.Equal(1UL, index.On(this._pool).Count(1UL));
    }

    [Fact]
    public void MismatchedRowWidthsAreRefusedByTheLibrary()
    {
        using DenseIndex index = SmallIndex();
        RowsView narrow = RowsView.RowsOf<float>(new float[] { 1f, 2f, 3f }, 3);

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(() => index.On(this._pool).Insert(KeysOf(1), narrow)).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(() => index.On(this._pool).Search(narrow, wanted: 1, this._arena)).Status);
    }

    [Fact]
    public void KeyRunsAreCheckedAgainstTheRowsTheyAccompany()
    {
        using DenseIndex index = SmallIndex();

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.On(this._pool).Insert(KeysOf(1), Rows([.. RowOf(1f), .. RowOf(2f)]))).Status);
    }

    [Fact]
    public void NegativeBoundsAndMissingArenasAreRefusedBeforeTheBoundary()
    {
        using DenseIndex index = SmallIndex();

        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => index.Search(Rows(RowOf(1f)), wanted: -1, this._arena)).Status);
        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => index.Search(Rows(RowOf(1f)), wanted: 1, null!)).Status);
    }

    #endregion Geometry

    #region Batches & Outputs

    [Fact]
    public void BatchedRowsAndQueriesKeepTheirLanes()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3), Rows([.. RowOf(10f), .. RowOf(20f), .. RowOf(30f)]));
        Assert.Equal(3UL, index.Describe().Size);

        Matches found = index.On(this._pool).Search(Rows([.. RowOf(30f), .. RowOf(10f)]), wanted: 2, this._arena);

        Assert.Equal(2, found.Queries);
        Assert.Equal(2, found[0].Count);
        Assert.Equal(2, found[1].Count);
        Assert.Equal(Key.FromUInt64(3), found[0].Keys[0]);
        Assert.Equal(Key.FromUInt64(1), found[1].Keys[0]);
        Assert.Equal(2, found[0].Distances.Length);
        Assert.Equal(0d, found[0].Distances[0], 6);
    }

    [Fact]
    public void SearchesReportFewerMatchesThanWantedWithoutReadingPastThem()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1), Rows(RowOf(1f)));

        Matches found = index.On(this._pool).Search(Rows(RowOf(1f)), wanted: 8, this._arena);

        Assert.Equal(8, found.Columns);
        Assert.Equal(1, found[0].Count);
        Assert.Single(found[0].Keys.ToArray());
        Assert.Single(found[0].Distances.ToArray());

        // A zero wanted lands only zeroed counts, with no keys or distances behind them.
        Matches none = index.On(this._pool).Search(Rows(RowOf(1f)), wanted: 0, this._arena);
        Assert.Equal(0, none.Columns);
        Assert.Equal(0, none[0].Count);
        Assert.True(none[0].Keys.IsEmpty);
    }

    [Fact]
    public void ScansScoreEveryRowSoEveryQueryFindsItself()
    {
        using DenseIndex index = DenseIndex.InitRam(DenseOptions.Default with { Dimensions = Dimensions });
        index.Reserve(4);
        float[] rows = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0.5f, 0.5f, 0.5f, 0.5f];
        index.On(this._pool).Insert(KeysOf(1, 2, 3, 4), Rows(rows));

        Matches found = index.On(this._pool).Search(Rows(rows), wanted: 1, this._arena, traversal: Traversal.Scan);

        Assert.Equal(4, found.Queries);
        for (int query = 0; query < found.Queries; query++)
        {
            Assert.Equal(Key.FromUInt64((ulong)query + 1), found[query].Keys[0]);
        }
    }

    [Fact]
    public void TraversalStatisticsAreReportedAndAccumulate()
    {
        using DenseIndex index = SmallIndex();
        DenseTraversalStats insertion = new();
        index.On(this._pool).Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]), stats: insertion);

        DenseTraversalStats query = new();
        index.On(this._pool).Search(Rows(RowOf(1f)), wanted: 2, this._arena, stats: query);
        Assert.True(query.ComputedDistances > 0);

        ulong before = query.ComputedDistances;
        query.Accumulate(insertion);
        Assert.Equal(before + insertion.ComputedDistances, query.ComputedDistances);
    }

    [Fact]
    public void ScalarKindsOtherThanFloatCross()
    {
        using DenseIndex index = DenseIndex.InitRam(SmallOptions() with { Quantization = Scalar.I8 });
        index.Reserve(4);

        index.On(this._pool).Insert(KeysOf(1), RowsView.RowsOf<sbyte>(new sbyte[] { 1, 2, 3, 4 }, Dimensions));
        index.On(this._pool).Insert(KeysOf(2), RowsView.RowsOf<byte>(new byte[] { 1, 2, 3, 4 }, Dimensions));

        DenseVectors retrieved = index.Get(2UL, this._arena, Scalar.F64);
        Assert.Equal(new double[] { 1, 2, 3, 4 }, retrieved.Row<double>(0).ToArray());
    }

    #endregion Batches & Outputs

    #region Refinement & Linear Algebra

    [Fact]
    public void RefinementAndAuditsLeaveTheGraphSearchable()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3), Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]));

        RefineReport report = index.On(this._pool).Refine(expansionFactor: 64);
        DenseReachability reachability = index.On(this._pool).Health();

        Assert.True(report.Changed <= report.Reselected);
        Assert.Equal(3UL, report.Reachability.Audited);
        Assert.Equal(3UL, reachability.Audited);
        Assert.Equal(3UL, reachability.Reachable);
        Assert.Equal(Key.FromUInt64(3), index.Search(Rows(RowOf(3f)), wanted: 1, this._arena)[0].Keys[0]);
    }

    [Fact]
    public void SplitEmptiesTheSourceIntoItsReceivers()
    {
        using DenseIndex index = SmallIndex();
        using DenseIndex left = SmallIndex();
        using DenseIndex right = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3), Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]));

        index.On(this._pool).Split([left, right], seed: 7);

        Assert.Equal(3UL, left.Describe().Size + right.Describe().Size);
        Assert.Equal(0UL, index.Describe().Size);
    }

    [Fact]
    public void SampleDrawsEveryEntryOfASmallIndex()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3), Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]));

        Sample sample = index.On(this._pool).Sample(8, this._arena, new DenseSampleConfig { Seed = 42 });

        Assert.Equal(3, sample.Count);
        Assert.Equal(3, sample.Parents.Queries);
        Assert.Equal(9, sample.Coordinates.Length);
    }

    [Fact]
    public void LateInteractionSumsTheMaximaOfEveryQueryRow()
    {
        DenseOptions options = DenseOptions.Default with
        {
            Dimensions = Dimensions,
            Metric = DenseMetric.Inner,
            ValuesPerKey = 2,
        };
        using DenseIndex index = DenseIndex.InitRam(options);
        index.Reserve(8);

        float[] tokens = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f];
        index.On(this._pool).Insert(KeysOf(1, 1, 2, 2), Rows(tokens));

        float[] query = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f];
        DenseProducts scored = index.On(this._pool).Maxsim(Rows(query), rowsPerQuery: 2, wanted: 2, this._arena);

        Assert.Equal(2, scored.Keys(0).Length);
        Assert.Equal(Key.FromUInt64(1), scored.Keys(0)[0]);
        Assert.Equal(2d, scored.Scores(0)[0], 6);
        Assert.Equal(Key.FromUInt64(2), scored.Keys(0)[1]);
        int bounded = scored.Keys(0).Length;
        Assert.True(index.Maxsim(Rows(query), rowsPerQuery: 2, wanted: 0, this._arena).Keys(0).Length >= bounded);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Maxsim(
                    Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]), rowsPerQuery: 2, wanted: 1, this._arena)).Status);
    }

    [Fact]
    public void ProductsReportEveryPairTheWalkScored()
    {
        using DenseIndex index = DenseIndex.InitRam(SmallOptions() with { Metric = DenseMetric.Inner });
        index.Reserve(2);
        index.On(this._pool).Insert(KeysOf(1, 2), Rows([1f, 0f, 0f, 0f, 0f, 2f, 0f, 0f]));

        DenseProducts products = index.On(this._pool).Multiply(Rows([1f, 1f, 0f, 0f, 0f, 3f, 0f, 0f]), this._arena);
        Assert.Equal(2, products.Queries);
        Assert.Equal(4UL, products.Required);

        Dictionary<Key, double> second = [];
        for (int pair = 0; pair < products.Keys(1).Length; pair++)
        {
            second[products.Keys(1)[pair]] = products.Scores(1)[pair];
        }

        Assert.Equal(0d, second[1UL], 6);
        Assert.Equal(6d, second[2UL], 6);

        DenseProducts capped = index.Multiply(Rows(RowOf(1f)), this._arena, wanted: 1);
        Assert.Single(capped.Keys(0).ToArray());
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => products.Keys(1).Length).Status);

        using DenseIndex distances = SmallIndex();
        Assert.Equal(
            Status.ConfigMismatch,
            Assert.Throws<USearchException>(() => distances.Multiply(Rows(RowOf(1f)), this._arena)).Status);
    }

    [Fact]
    public void AttentionResolvesValuesThroughTheCallersRows()
    {
        using DenseIndex index = DenseIndex.InitRam(SmallOptions() with { Metric = DenseMetric.Inner });
        index.Reserve(2);
        index.Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));

        // Value rows come in the index's own quantization, here the raw bits of bf16 ones and zeros.
        const ushort One = 0x3F80;
        RowsView values = RowsView.RowsOf<ushort>(new ushort[] { One, 0, 0, 0, 0, One, 0, 0 }, Dimensions, Scalar.BF16);
        DenseSoftmaxState[] carry = new DenseSoftmaxState[1];

        // A zero query scores every key alike, so the output is the plain mean of the value rows.
        DenseVectors attended = index.On(this._pool).Attend(
            Rows(new float[Dimensions]), values, key => key.TryGetUInt64(out ulong identity) ? (int)identity - 1 : -1,
            this._arena, carry: carry);
        Assert.Equal(new[] { 0.5f, 0.5f, 0f, 0f }, attended.Row<float>(0).ToArray());
        Assert.Equal(2d, carry[0].Partition, 4);

        DenseVectors widened = index.Attend(
            Rows(new float[Dimensions]), values, _ => 0, this._arena, outputsScalar: Scalar.F64);
        Assert.Equal(Scalar.F64, widened.Scalar);
        Assert.Equal(1d, widened.Row<double>(0)[0], 6);

        RowsView narrow = RowsView.RowsOf<ushort>(new ushort[6], 3, Scalar.BF16);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Attend(Rows(new float[Dimensions]), narrow, _ => 0, this._arena)).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Attend(Rows(new float[Dimensions]), values, _ => 7, this._arena)).Status);
        Assert.Throws<InvalidOperationException>(
            () => index.Attend(
                Rows(new float[Dimensions]), values, _ => throw new InvalidOperationException(), this._arena));
    }

    #endregion Refinement & Linear Algebra

    #region Retrieval & Arenas

    [Fact]
    public void GetLandsEveryRowSharingTheKey()
    {
        using DenseIndex index = DenseIndex.InitRam(SmallOptions() with { ValuesPerKey = 3 });
        index.On(this._pool).Reserve(64);

        Key shared = 5UL;
        index.On(this._pool).Insert(new[] { shared, shared, shared }, Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]));

        using Arena other = new(this._pool);
        DenseVectors first = index.On(this._pool).Get(shared, this._arena);
        DenseVectors second = index.Get(1UL, other);
        Assert.Equal(3, first.Rows);
        Assert.Equal(Dimensions, first.Columns);
        Assert.Equal(Dimensions * sizeof(float), first.StrideBytes);
        Assert.Equal(RowOf(3f), first.Row<float>(2).ToArray());
        Assert.Equal(0, second.Rows);
        Assert.Equal(Residency.Ram, other.Residency);
        Assert.Equal(3UL, index.On(this._pool).Count(shared));
    }

    [Fact]
    public void ArenasAreRewoundByEveryCallThatReusesThem()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));

        using Arena arena = new();
        Matches found = index.On(this._pool).Search(Rows(RowOf(1f)), wanted: 1, arena);
        QueryMatches query = found[0];
        Assert.Equal(Key.FromUInt64(1), query.Keys[0]);

        // Any verb reuses any arena, and the next call's rewind retires every earlier view.
        DenseVectors second = index.On(this._pool).Get(2UL, arena);
        Assert.Equal(RowOf(2f), second.Row<float>(0).ToArray());
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => query.Count).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => found[0].Keys.Length).Status);

        // A refused call rewinds the arena all the same.
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => index.Get(1UL, arena, Scalar.Unknown)).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => second.Row<float>(0).Length).Status);

        Monolith monolith = index.SaveBuffer(arena);
        arena.Dispose();
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => monolith.Bytes.Length).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Get(1UL, arena)).Status);
    }

    #endregion Retrieval & Arenas

    #region Filtered Search

    [Fact]
    public void FilteredSearchesAdmitOnlyWhatThePredicateApproves()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3, 4), Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f), .. RowOf(4f)]));

        Matches found = index.On(this._pool).Search(
            Rows(RowOf(1f)), wanted: 4, this._arena,
            predicate: key => key.TryGetUInt64(out ulong identity) && identity % 2 == 0);

        Assert.Equal(2, found[0].Count);
        foreach (Key key in found[0].Keys)
        {
            Assert.True(key.TryGetUInt64(out ulong identity));
            Assert.Equal(0UL, identity % 2);
        }
    }

    [Fact]
    public void PredicateFailuresSurfaceAfterTheCallRatherThanUnwindingThroughIt()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));

        Assert.Throws<InvalidOperationException>(
            () => index.On(this._pool).Search(
                Rows(RowOf(1f)), wanted: 2, this._arena, predicate: _ => throw new InvalidOperationException()));

        using Deferred<Matches> deferred = index.Deferred(this._pool).Search(
            Rows(RowOf(1f)), wanted: 2, this._arena, predicate: _ => throw new InvalidOperationException());
        Assert.Throws<InvalidOperationException>(() => deferred.Wait());
    }

    #endregion Filtered Search

    #region Handles

    [Fact]
    public void DisposedHandlesRefuseEveryVerb()
    {
        DenseIndex index = SmallIndex();
        index.Dispose();
        index.Dispose();

        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Describe()).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Count(1UL)).Status);
    }

    [Fact]
    public void CursorsWalkEveryRowOnceAndKeepTheirIndexAlive()
    {
        DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3), Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]));

        using DenseCursor cursor = index.On(this._pool).Cursor();
        index.Dispose();

        DenseBatch batch = cursor.Next(2, this._arena);
        Assert.Equal(2, batch.Count);
        Assert.Equal(Key.FromUInt64(1), batch.Keys[0]);

        batch = cursor.Next(2, this._arena);
        Assert.Equal(1, batch.Count);
        Assert.Equal(Key.FromUInt64(3), batch.Keys[0]);
        Assert.Throws<InvalidOperationException>(() => batch.Links(0));
        Assert.Equal(0, cursor.Next(2, this._arena).Count);
    }

    [Fact]
    public void SearchStreamsYieldMatchesAcrossPulls()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1, 2, 3), Rows([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]));

        DenseSearchStream stream;
        try
        {
            stream = index.On(this._pool).SearchStream(Rows(RowOf(1f)));
        }
        catch (USearchException refused) when (refused.Status == Status.Unsupported)
        {
            return; // A residency that streams no matches yet refuses the stream itself.
        }

        using (stream)
        {
            QueryMatches pulled = stream.Next(2, this._arena);
            Assert.Equal(2, pulled.Count);
            Assert.Equal(Key.FromUInt64(1), pulled.Keys[0]);
            Assert.Equal(Key.FromUInt64(2), pulled.Keys[1]);
            pulled = stream.Next(2, this._arena);
            Assert.Equal(1, pulled.Count);
            Assert.Equal(Key.FromUInt64(3), pulled.Keys[0]);
            Assert.Equal(0, stream.Next(2, this._arena).Count);
        }
    }

    [Fact]
    public void AbsorbConsumesOnlyTheDonorsItAbsorbed()
    {
        using DenseIndex index = SmallIndex();
        DenseIndex donor = SmallIndex();
        donor.Insert(KeysOf(7), Rows(RowOf(7f)));

        index.On(this._pool).Absorb(donor);
        Assert.True(index.Contains(7UL));
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => donor.Count(7UL)).Status);
        donor.Dispose();

        using DenseIndex other = DenseIndex.InitRam(SmallOptions() with { Quantization = Scalar.I8 });
        other.Reserve(1);
        other.Insert(KeysOf(8), Rows(RowOf(8f)));
        try
        {
            index.On(this._pool).Absorb(other);
        }
        catch (USearchException refused)
        {
            // The engine refuses another scalar, which the stand-in absorbs, and hands the donor back live.
            Assert.Equal(Status.ConfigMismatch, refused.Status);
            Assert.Equal(1UL, other.Count(8UL));
        }
    }

    #endregion Handles

    #region Serialization

    [Fact]
    public void CheckpointsRoundTripThroughAPath()
    {
        string path = Path.Combine(Path.GetTempPath(), $"usearch-{Guid.NewGuid():N}.safegraphs");
        try
        {
            using (DenseIndex index = SmallIndex())
            {
                index.On(this._pool).Insert(KeysOf(1, 2), Rows([.. RowOf(1f), .. RowOf(2f)]));
                using Deferred saved = index.Deferred(this._pool).Save(path);
                saved.Wait();
            }

            Assert.True(File.Exists(path));

            DenseMetadata metadata = DenseIndex.MetadataRead(path, this._pool);
            Assert.Equal((ulong)Dimensions, metadata.Dimensions);
            Assert.Equal(DenseMetric.Euclidean, metadata.Metric);

            using DenseIndex restored = DenseIndex.LoadRam(path, executor: this._pool);
            Assert.Equal((ulong)Dimensions, restored.Describe().Metadata.Dimensions);
            Assert.Equal(2UL, restored.Describe().Size);
            Assert.Equal(
                Key.FromUInt64(2), restored.On(this._pool).Search(Rows(RowOf(2f)), wanted: 1, this._arena)[0].Keys[0]);

            DenseIndex viewed;
            try
            {
                viewed = DenseIndex.ViewRam(path);
            }
            catch (USearchException refused) when (refused.Status == Status.Unsupported)
            {
                return; // RAM serves no mapping in place yet.
            }

            using (viewed)
            {
                Assert.True(viewed.Contains(1UL));
                Assert.Equal(
                    Status.ReadOnly,
                    Assert.Throws<USearchException>(() => viewed.Insert(KeysOf(3), Rows(RowOf(3f)))).Status);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CheckpointsRoundTripThroughABuffer()
    {
        using DenseIndex index = SmallIndex();
        index.On(this._pool).Insert(KeysOf(1), Rows(RowOf(1f)));

        Monolith monolith = index.On(this._pool).SaveBuffer(this._arena);
        byte[] buffer = monolith.Bytes.ToArray();
        Assert.Equal(buffer.Length, monolith.Length);
        using Deferred<Monolith> again = index.Deferred(this._pool).SaveBuffer(this._arena);
        Assert.Equal(buffer, again.Wait().Bytes.ToArray());

        DenseMetadata metadata = DenseIndex.MetadataReadBuffer(buffer);
        Assert.Equal((ulong)Dimensions, metadata.Dimensions);

        using DenseIndex restored = DenseIndex.LoadRamBuffer(buffer, executor: this._pool);
        Assert.Equal(1UL, restored.Describe().Size);
        Assert.True(restored.Contains(1UL));

        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => DenseIndex.LoadRamBuffer(ReadOnlySpan<byte>.Empty)).Status);
    }

    [Fact]
    public void LibraryFailuresArriveAsStatusesRatherThanMessages()
    {
        USearchException failure = Assert.Throws<USearchException>(
            () => DenseIndex.LoadRam(
                Path.Combine(Path.GetTempPath(), "usearch-absent-checkpoint.safegraphs"), executor: this._pool));

        Assert.Equal(Status.Io, failure.Status);
        Assert.Equal(
            Status.InvalidArgument,
            Assert.Throws<USearchException>(() => DenseIndex.LoadRam(string.Empty)).Status);
    }

    #endregion Serialization

    #region Introspection

    [Fact]
    public void LibraryProbesReportSomething()
    {
        Assert.NotEmpty(USearchCapabilities.CapabilitiesCompiled());
        Assert.NotEmpty(USearchCapabilities.CapabilitiesAvailable());
    }

    #endregion Introspection
}
