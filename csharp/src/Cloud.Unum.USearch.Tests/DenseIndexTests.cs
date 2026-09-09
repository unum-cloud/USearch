using System.Runtime.InteropServices;

namespace Cloud.Unum.USearch.Tests;

/// <summary>
/// The boundary of the binding: marshalling, the options round trip through the residency union,
/// key byte order, stride arithmetic, handle ownership, and truncation reporting. That vector
/// search works is the C library's job, and is not re-tested here.
/// </summary>
public sealed class DenseIndexTests : IDisposable
{
    private const int Dimensions = 4;

    private readonly Executor _pool = Executor.ForRam(threads: 1);

    /// <summary>Releases the executor every test in this class fans out over.</summary>
    public void Dispose() => this._pool.Dispose();

    private static DenseOptions SmallOptions() => DenseOptions.Default with
    {
        Dimensions = Dimensions,
        Metric = DenseMetric.Euclidean,
        Quantization = ScalarKind.F32,
        Connectivity = 11,
        Residency = ResidencyKind.Ram,
    };

    private static DenseIndex SmallIndex()
    {
        DenseIndex index = new(SmallOptions());
        index.Reserve(64);
        return index;
    }

    private static float[] RowOf(float first) => [first, first + 1f, first + 2f, first + 3f];

    #region Keys

    [Fact]
    public void KeyRendersIntegerIdentitiesInTheLowBytes()
    {
        USearchKey key = USearchKey.FromUInt64(42);

        Assert.Equal("00000000-0000-0000-0000-00000000002a", key.ToString());
        Assert.Equal(0x2a, key[15]);
        Assert.Equal(0, key[0]);
        Assert.True(key.TryGetUInt64(out ulong identity));
        Assert.Equal(42UL, identity);
    }

    [Fact]
    public void KeyParsesTheRenderingItPrints()
    {
        USearchKey wide = USearchKey.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        Assert.Equal("0123abcd-4567-89ef-fedc-ba9876543210", wide.ToString());
        Assert.Equal(0x01, wide[0]);
        Assert.Equal(0x10, wide[15]);
        Assert.False(wide.TryGetUInt64(out _));
        Assert.Equal(wide, USearchKey.Parse("0123ABCD456789EFFEDCBA9876543210"));
        Assert.False(USearchKey.TryParse("0123abcd", out _));
        Assert.False(USearchKey.TryParse("0123abcd-4567-89ef-fedc-ba98765432zz", out _));
    }

    [Fact]
    public void KeyEqualityAgreesWithHashing()
    {
        USearchKey first = 7UL;
        USearchKey same = USearchKey.FromUInt64(7);
        USearchKey other = USearchKey.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        Assert.True(first == same);
        Assert.True(first != other);
        Assert.Equal(first.GetHashCode(), same.GetHashCode());
        Assert.NotEqual(first.GetHashCode(), other.GetHashCode());
        Assert.Single(new HashSet<USearchKey> { first, same });
    }

    [Fact]
    public void KeysCrossTheBoundaryIntact()
    {
        using DenseIndex index = SmallIndex();
        USearchKey wide = USearchKey.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        Assert.Equal(1UL, index.Add(this._pool, [wide], Matrix.RowsOf<float>(RowOf(1f), Dimensions)));
        Assert.Equal(1UL, index.Count(wide));
        Assert.True(index.Contains(wide));
        Assert.False(index.Contains(USearchKey.FromUInt64(1)));

        MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 1);
        index.Search(this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions), found);
        Assert.Equal(wide, found[0].Keys[0]);
    }

    #endregion Keys

    #region Options & Residency

    [Fact]
    public void ResidencyUnionIsAsWideAsItsWidestArm()
    {
        int union = Marshal.SizeOf<NativeResidencyOptions>();

        Assert.Equal(Marshal.SizeOf<NativeRamOptions>(), union);
        Assert.True(union >= Marshal.SizeOf<NativeVramOptions>());
        Assert.True(union >= Marshal.SizeOf<NativeNvmeOptions>());
        Assert.Equal(0, Marshal.OffsetOf<NativeResidencyOptions>(nameof(NativeResidencyOptions.Ram)).ToInt32());
        Assert.Equal(0, Marshal.OffsetOf<NativeResidencyOptions>(nameof(NativeResidencyOptions.Vram)).ToInt32());
        Assert.Equal(0, Marshal.OffsetOf<NativeResidencyOptions>(nameof(NativeResidencyOptions.Nvme)).ToInt32());
    }

    [Fact]
    public void MetadataIsLaidOutAsTheCheckpointHeaderSpellsIt()
    {
        Assert.Equal(0, Marshal.OffsetOf<NativeDenseMetadata>(nameof(NativeDenseMetadata.Metric)).ToInt32());
        Assert.Equal(
            sizeof(int),
            Marshal.OffsetOf<NativeDenseMetadata>(nameof(NativeDenseMetadata.Quantization)).ToInt32());
        Assert.Equal(
            2 * sizeof(int),
            Marshal.OffsetOf<NativeDenseMetadata>(nameof(NativeDenseMetadata.Dimensions)).ToInt32());
        Assert.Equal(
            (2 * sizeof(int)) + IntPtr.Size,
            Marshal.OffsetOf<NativeDenseMetadata>(nameof(NativeDenseMetadata.Connectivity)).ToInt32());
        Assert.Equal((2 * sizeof(int)) + (2 * IntPtr.Size), Marshal.SizeOf<NativeDenseMetadata>());
    }

    [Fact]
    public void GeometrySurvivesTheRoundTrip()
    {
        DenseOptions options = DenseOptions.Default with
        {
            Dimensions = 96,
            Connectivity = 24,
            ValuesPerKey = 3,
        };

        DenseOptions reloaded = options.Reload();

        Assert.Equal(96UL, reloaded.Dimensions);
        Assert.Equal(24UL, reloaded.Connectivity);
        Assert.Equal(3UL, reloaded.ValuesPerKey);
        Assert.Equal(DenseMetric.Angular, reloaded.Metric);
        Assert.Equal(ScalarKind.BF16, reloaded.Quantization);
    }

    [Fact]
    public void PlacementIsNeverReadBackOutOfACheckpoint()
    {
        DenseOptions vram = SmallOptions() with { Residency = ResidencyKind.Vram };
        Assert.Equal(ResidencyKind.Ram, vram.Reload().Residency);

        DenseOptions nvme = SmallOptions() with { Residency = ResidencyKind.Nvme };
        Assert.Equal(ResidencyKind.Ram, nvme.Reload().Residency);
        Assert.Equal((ulong)Dimensions, nvme.Reload().Dimensions);
    }

    [Fact]
    public void UnknownTagsAreRejectedRatherThanGuessedAt()
    {
        DenseOptions unknownResidency = SmallOptions() with { Residency = (ResidencyKind)7 };
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => unknownResidency.Reload()).Status);

        DenseOptions unknownMetric = SmallOptions() with { Metric = (DenseMetric)9 };
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => unknownMetric.Reload()).Status);

        DenseOptions unknownQuantization = SmallOptions() with { Quantization = (ScalarKind)44 };
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => unknownQuantization.Reload()).Status);
    }

    [Fact]
    public void OptionsReachTheLibraryAsTheyWereWritten()
    {
        using DenseIndex index = new(SmallOptions());

        Assert.Equal((ulong)Dimensions, index.Dimensions());
        Assert.Equal(11UL, index.Connectivity());
        Assert.Equal(0UL, index.Size());
        Assert.NotEmpty(index.Capabilities());
    }

    #endregion Options & Residency

    #region Executors

    [Fact]
    public void ExecutorsDrainAndRefuseEveryVerbOnceDisposed()
    {
        Executor pool = Executor.ForRam(threads: 1);
        pool.Drain();
        pool.Dispose();
        pool.Dispose();

        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(pool.Drain).Status);

        using DenseIndex index = SmallIndex();
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(
                () => index.Add(pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions))).Status);
    }

    [Fact]
    public void RamExecutorsNameTheirMemoryDomains()
    {
        using Executor pool = Executor.ForRam([0], threads: 1);
        pool.Drain();
    }

    #endregion Executors

    #region Scalars & Matrices

    [Fact]
    public void PackedKindsPitchByTheValuesTheyHoldRatherThanTheDimensions()
    {
        Assert.Equal(256UL, ScalarKind.F32.RowBytes(64));
        Assert.Equal(32UL, ScalarKind.U4x2.RowBytes(64));
        Assert.Equal(8UL, ScalarKind.U1x8.RowBytes(64));

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(() => ScalarKind.U1x8.RowBytes(60)).Status);
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => ScalarKind.Unknown.RowBytes(64)).Status);

        byte[] bits = new byte[16];
        Matrix packed = Matrix.RowsOf<byte>(bits, 64, ScalarKind.U1x8);
        Assert.Equal(2, packed.Rows);
        Assert.Equal(8, packed.StrideBytes);
        Assert.Equal(64, packed.Dimensions);
        Assert.Equal(ScalarKind.U1x8, packed.Kind);
    }

    [Fact]
    public void PackedBitSetsReachTheIndexAndComeBack()
    {
        DenseOptions options = DenseOptions.Default with
        {
            Dimensions = 64,
            Metric = DenseMetric.Hamming,
            Quantization = ScalarKind.U1x8,
        };
        using DenseIndex index = new(options);
        index.Reserve(4);

        byte[] stored = [0b0000_0001, 0, 0, 0, 0, 0, 0, 0b1000_0000];
        Assert.Equal(1UL, index.Add(this._pool, [1UL], Matrix.RowsOf<byte>(stored, 64, ScalarKind.U1x8)));

        byte[] retrieved = new byte[8];
        Assert.Equal(1UL, index.Get(1UL, Matrix.RowsOf<byte>(retrieved, 64, ScalarKind.U1x8)));
        Assert.Equal(stored, retrieved);
    }

    [Fact]
    public void StridedRowsSkipThePaddingBetweenThem()
    {
        using DenseIndex index = SmallIndex();
        float[] padded = [.. RowOf(10f), 0f, 0f, .. RowOf(20f), 0f, 0f];

        Matrix rows = Matrix.StridedRowsOf<float>(
            padded, rows: 2, dimensions: Dimensions, strideBytes: 6 * sizeof(float), ScalarKind.F32);
        Assert.Equal(2UL, index.Add(this._pool, [1UL, 2UL], rows));

        MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 1);
        index.Search(this._pool, Matrix.RowsOf<float>(RowOf(20f), Dimensions), found);
        Assert.Equal(USearchKey.FromUInt64(2), found[0].Keys[0]);
    }

    [Fact]
    public void MatricesRefuseLayoutsNoSlabCouldHold()
    {
        float[] slab = new float[8];

        Assert.Equal(
            Status.UnsupportedLayout,
            Assert.Throws<USearchException>(
                () => Matrix.StridedRowsOf<float>(slab, 2, Dimensions, 8, ScalarKind.F32)).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => Matrix.StridedRowsOf<float>(slab, 3, Dimensions, 16, ScalarKind.F32)).Status);
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => Matrix.RowsOf<float>(slab, Dimensions, ScalarKind.F64)).Status);
    }

    #endregion Scalars & Matrices

    #region Geometry

    [Fact]
    public void UnfrozenGeometryIsRefusedRatherThanSizedToZero()
    {
        DenseOptions deferred = DenseOptions.Default with { Quantization = ScalarKind.F32 };
        using DenseIndex index = new(deferred);

        Assert.Equal(0UL, index.Dimensions());
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(
                () => index.Add(this._pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions))).Status);
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(
                () => index.Search(
                    this._pool,
                    Matrix.RowsOf<float>(RowOf(1f), Dimensions),
                    MatchesSpan.Packed(1, 1))).Status);
    }

    [Fact]
    public void MismatchedRowWidthsAreRefused()
    {
        using DenseIndex index = SmallIndex();

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Add(this._pool, [1UL], Matrix.RowsOf<float>([1f, 2f, 3f], 3))).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Search(
                    this._pool, Matrix.RowsOf<float>([1f, 2f, 3f], 3), MatchesSpan.Packed(1, 1))).Status);
    }

    [Fact]
    public void KeyRunsAreCheckedAgainstTheRowsTheyAccompany()
    {
        using DenseIndex index = SmallIndex();
        float[] rows = [.. RowOf(1f), .. RowOf(2f)];

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Add(this._pool, [1UL], Matrix.RowsOf<float>(rows, Dimensions))).Status);
    }

    [Fact]
    public void UnnameableScalarTypesAreRefused()
    {
        short[] row = [1, 2, 3, 4];

        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => Matrix.RowsOf<short>(row, Dimensions)).Status);
    }

    [Fact]
    public void OversizedExtentsAreRefusedBeforeAllocating()
    {
        Assert.Equal(
            Status.Overflow,
            Assert.Throws<USearchException>(
                () => Extents.CheckedRowMajor(ulong.MaxValue / 2, 4, 16, "probe")).Status);
        Assert.Equal(
            Status.Overflow,
            Assert.Throws<USearchException>(
                () => Extents.CheckedRowMajor(1, ulong.MaxValue / 8, 16, "probe")).Status);
        Assert.Equal(
            Status.Overflow,
            Assert.Throws<USearchException>(() => MatchesSpan.Packed(2, int.MaxValue)).Status);
    }

    #endregion Geometry

    #region Batches & Outputs

    [Fact]
    public void BatchedRowsAndQueriesKeepTheirLanes()
    {
        using DenseIndex index = SmallIndex();
        USearchKey[] keys = [1UL, 2UL, 3UL];
        float[] rows = [.. RowOf(10f), .. RowOf(20f), .. RowOf(30f)];

        Assert.Equal(3UL, index.Add(this._pool, keys, Matrix.RowsOf<float>(rows, Dimensions)));
        Assert.Equal(3UL, index.Size());

        float[] queries = [.. RowOf(30f), .. RowOf(10f)];
        MatchesSpan found = MatchesSpan.Packed(queries: 2, wanted: 2);
        index.Search(this._pool, Matrix.RowsOf<float>(queries, Dimensions), found);

        Assert.Equal(2, found.Queries);
        Assert.Equal(2, found[0].Count);
        Assert.Equal(2, found[1].Count);
        Assert.Equal(USearchKey.FromUInt64(3), found[0].Keys[0]);
        Assert.Equal(USearchKey.FromUInt64(1), found[1].Keys[0]);
        Assert.Equal(2, found[0].Distances.Length);
        Assert.Equal(0d, found[0].Distances[0], 6);
    }

    [Fact]
    public void LentSlabsAreFilledInPlaceRatherThanReplaced()
    {
        using DenseIndex index = SmallIndex();
        index.Add(this._pool, [1UL, 2UL], Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f)], Dimensions));

        USearchKey[] keys = new USearchKey[2];
        double[] distances = new double[2];
        nuint[] counts = new nuint[1];
        MatchesSpan lent = MatchesSpan.Of(keys, distances, counts);

        index.Search(this._pool, Matrix.RowsOf<float>(RowOf(2f), Dimensions), lent);

        Assert.Equal(USearchKey.FromUInt64(2), keys[0]);
        Assert.Equal((nuint)2, counts[0]);
        Assert.Equal(0d, distances[0], 6);
        Assert.Equal(keys[0], lent[0].Keys[0]);

        // The same slabs serve the next point of a sweep, with nothing allocated between them.
        index.Search(this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions), lent);
        Assert.Equal(USearchKey.FromUInt64(1), keys[0]);
    }

    [Fact]
    public void SlabsThatDoNotReadAsOneShapeAreRefused()
    {
        using DenseIndex index = SmallIndex();
        index.Add(this._pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions));

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => MatchesSpan.Of(new USearchKey[4], new double[8], new nuint[2])).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => MatchesSpan.Of(new USearchKey[5], new double[5], new nuint[2])).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => MatchesSpan.Of(Array.Empty<USearchKey>(), Array.Empty<double>(), new nuint[1])).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => MatchesSpan.Of(new USearchKey[4], new double[4], Array.Empty<nuint>())).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.Search(
                    this._pool,
                    Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f)], Dimensions),
                    MatchesSpan.Packed(queries: 1, wanted: 1))).Status);
    }

    [Fact]
    public void SearchesReportFewerMatchesThanWantedWithoutReadingPastThem()
    {
        using DenseIndex index = SmallIndex();
        index.Add(this._pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions));

        MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 8);
        index.Search(this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions), found);

        Assert.Equal(8, found.Wanted);
        Assert.Equal(1, found[0].Count);
        Assert.Single(found[0].Keys.ToArray());
        Assert.Single(found[0].Distances.ToArray());
    }

    [Fact]
    public void ReportedCountsAboveTheRowWidthAreClamped()
    {
        // The sibling verbs disagree on what a count means, so the slicing clamps rather than
        // trusting it: entries past the row width were never written by anyone.
        MatchesSpan results = MatchesSpan.Of(new USearchKey[4], new double[4], new nuint[] { 5, 1 });

        Assert.Equal(2, results[0].Count);
        Assert.True(results[0].WasTruncated);
        Assert.Equal(2, results[0].Keys.Length);
        Assert.Equal(1, results[1].Count);
        Assert.False(results[1].WasTruncated);
        Assert.Equal(1, results[1].Distances.Length);
    }

    [Fact]
    public void TraversalStatisticsAreReportedAndAccumulate()
    {
        using DenseIndex index = SmallIndex();
        TraversalStats insertion = new();
        index.Add(
            this._pool, [1UL, 2UL], Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f)], Dimensions),
            stats: insertion);

        TraversalStats query = new();
        index.Search(
            this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions),
            MatchesSpan.Packed(1, 2), stats: query);

        Assert.True(insertion.VisitedMembers > 0);
        Assert.True(query.ComputedDistances > 0);

        ulong before = query.ComputedDistances;
        query.Accumulate(insertion);
        Assert.Equal(before + insertion.ComputedDistances, query.ComputedDistances);
    }

    [Fact]
    public void ScalarKindsOtherThanFloatCross()
    {
        DenseOptions options = SmallOptions() with { Quantization = ScalarKind.I8 };
        using DenseIndex index = new(options);
        index.Reserve(4);

        sbyte[] row = [1, 2, 3, 4];
        Assert.Equal(1UL, index.Add(this._pool, [1UL], Matrix.RowsOf<sbyte>(row, Dimensions)));

        byte[] unsigned = [1, 2, 3, 4];
        Assert.Equal(1UL, index.Add(this._pool, [2UL], Matrix.RowsOf<byte>(unsigned, Dimensions)));

        double[] retrieved = new double[Dimensions];
        Assert.Equal(1UL, index.Get(2UL, Matrix.RowsOf<double>(retrieved, Dimensions)));
        Assert.Equal(new double[] { 1, 2, 3, 4 }, retrieved);
    }

    #endregion Batches & Outputs

    #region Refinement & Late Interaction

    [Fact]
    public void RefinementLeavesTheGraphSearchable()
    {
        using DenseIndex index = SmallIndex();
        index.Add(
            this._pool, [1UL, 2UL, 3UL],
            Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)], Dimensions));

        index.Refine(this._pool, new DenseInsertionOptions { ExpansionFactor = 64 });

        MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 1);
        index.Search(this._pool, Matrix.RowsOf<float>(RowOf(3f), Dimensions), found);
        Assert.Equal(USearchKey.FromUInt64(3), found[0].Keys[0]);
        Assert.Equal(3UL, index.Size());
    }

    [Fact]
    public void LateInteractionSumsTheMaximaOfEveryQueryRow()
    {
        DenseOptions options = DenseOptions.Default with
        {
            Dimensions = Dimensions,
            Metric = DenseMetric.Inner,
            Quantization = ScalarKind.F32,
            ValuesPerKey = 2,
        };
        using DenseIndex index = new(options);
        index.Reserve(8);

        float[] tokens =
        [
            1f, 0f, 0f, 0f,
            0f, 1f, 0f, 0f,
            0f, 0f, 1f, 0f,
            0f, 0f, 0f, 1f,
        ];
        index.Add(this._pool, [1UL, 1UL, 2UL, 2UL], Matrix.RowsOf<float>(tokens, Dimensions));

        float[] query = [1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f];
        MatchesSpan scored = MatchesSpan.Packed(queries: 1, wanted: 2);
        index.MaxSim(this._pool, Matrix.RowsOf<float>(query, Dimensions), rowsPerQuery: 2, scored);

        Assert.Equal(2, scored[0].Count);
        Assert.Equal(USearchKey.FromUInt64(1), scored[0].Keys[0]);
        Assert.Equal(2d, scored[0].Distances[0], 6);
        Assert.Equal(USearchKey.FromUInt64(2), scored[0].Keys[1]);
    }

    [Fact]
    public void LateInteractionRefusesQueriesThatDoNotDivide()
    {
        DenseOptions options = DenseOptions.Default with
        {
            Dimensions = Dimensions,
            Metric = DenseMetric.Inner,
            Quantization = ScalarKind.F32,
        };
        using DenseIndex index = new(options);
        index.Reserve(4);
        index.Add(this._pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions));

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(
                () => index.MaxSim(
                    this._pool,
                    Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)], Dimensions),
                    rowsPerQuery: 2,
                    MatchesSpan.Packed(2, 1))).Status);
    }

    #endregion Refinement & Late Interaction

    #region Retrieval

    [Fact]
    public void GetReportsTheTrueCountAndTruncatesTheCopy()
    {
        DenseOptions options = SmallOptions() with { ValuesPerKey = 3 };
        using DenseIndex index = new(options);
        index.Reserve(64);

        USearchKey shared = 5UL;
        index.Add(
            this._pool, [shared, shared, shared],
            Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)], Dimensions));

        Assert.Equal(3UL, index.Count(shared));
        Assert.Equal(3UL, index.Get(shared));

        float[] roomForOne = new float[Dimensions];
        Assert.Equal(3UL, index.Get(shared, Matrix.RowsOf<float>(roomForOne, Dimensions)));
        Assert.Equal(RowOf(1f), roomForOne);

        float[] roomForAll = new float[3 * Dimensions];
        Assert.Equal(3UL, index.Get(shared, Matrix.RowsOf<float>(roomForAll, Dimensions)));
        Assert.Equal(RowOf(3f), roomForAll[(2 * Dimensions)..]);
    }

    [Fact]
    public void GetOfAnAbsentKeyWritesNothing()
    {
        using DenseIndex index = SmallIndex();
        index.Add(this._pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions));

        float[] untouched = new float[Dimensions];
        Assert.Equal(0UL, index.Get(99UL, Matrix.RowsOf<float>(untouched, Dimensions)));
        Assert.Equal(0UL, index.Count(99UL));
        Assert.Equal(new float[Dimensions], untouched);
    }

    #endregion Retrieval

    #region Filtered Search

    [Fact]
    public void FilteredSearchesAdmitOnlyWhatThePredicateApproves()
    {
        using DenseIndex index = SmallIndex();
        USearchKey[] keys = [1UL, 2UL, 3UL, 4UL];
        float[] rows = [.. RowOf(1f), .. RowOf(2f), .. RowOf(3f), .. RowOf(4f)];
        index.Add(this._pool, keys, Matrix.RowsOf<float>(rows, Dimensions));

        MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 4);
        index.SearchIf(
            this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions), found,
            key => key.TryGetUInt64(out ulong identity) && identity % 2 == 0
                ? Admission.Admit
                : Admission.Reject);

        Assert.Equal(2, found[0].Count);
        foreach (USearchKey key in found[0].Keys)
        {
            Assert.True(key.TryGetUInt64(out ulong identity));
            Assert.Equal(0UL, identity % 2);
        }
    }

    [Fact]
    public void PredicateFailuresSurfaceAfterTheCallRatherThanUnwindingThroughIt()
    {
        using DenseIndex index = SmallIndex();
        index.Add(this._pool, [1UL, 2UL], Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f)], Dimensions));

        USearchException failure = Assert.Throws<USearchException>(
            () => index.SearchIf(
                this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions), MatchesSpan.Packed(1, 2),
                _ => throw new InvalidOperationException()));

        Assert.Equal(Status.Internal, failure.Status);
        Assert.IsType<InvalidOperationException>(failure.InnerException);
    }

    #endregion Filtered Search

    #region Handles

    [Fact]
    public void DisposedHandlesRefuseEveryVerb()
    {
        DenseIndex index = SmallIndex();
        index.Dispose();
        index.Dispose();

        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Size()).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Count(1UL)).Status);
    }

    [Fact]
    public void CursorsWalkTheSnapshotTheyCaptured()
    {
        using DenseIndex index = SmallIndex();
        index.Add(
            this._pool, [1UL, 2UL, 3UL],
            Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)], Dimensions));

        using USearchCursor cursor = index.Cursor();
        index.Add(this._pool, [4UL], Matrix.RowsOf<float>(RowOf(4f), Dimensions));

        USearchKey[] keys = new USearchKey[2];
        float[] vectors = new float[2 * Dimensions];
        Assert.Equal(2, cursor.Next(keys, Matrix.RowsOf<float>(vectors, Dimensions)));
        Assert.Equal(USearchKey.FromUInt64(1), keys[0]);
        Assert.Equal(RowOf(2f), vectors[Dimensions..]);

        Assert.Equal(1, cursor.Next(keys));
        Assert.Equal(USearchKey.FromUInt64(3), keys[0]);
        Assert.Equal(0, cursor.Next(keys));
    }

    [Fact]
    public void FindersStreamMatchesAcrossPulls()
    {
        using DenseIndex index = SmallIndex();
        index.Add(
            this._pool, [1UL, 2UL, 3UL],
            Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)], Dimensions));

        using USearchFinder finder = index.Find(this._pool, Matrix.RowsOf<float>(RowOf(1f), Dimensions));
        USearchKey[] keys = new USearchKey[2];
        double[] distances = new double[2];

        Assert.Equal(2, finder.Next(keys, distances));
        Assert.Equal(USearchKey.FromUInt64(1), keys[0]);
        Assert.Equal(1, finder.Next(keys, distances));
        Assert.Equal(USearchKey.FromUInt64(3), keys[0]);
        Assert.Equal(0, finder.Next(keys, distances));
    }

    #endregion Handles

    #region Serialization

    [Fact]
    public void CheckpointsRoundTripThroughAPathAndFreezeTheGeometry()
    {
        string path = Path.Combine(Path.GetTempPath(), $"usearch-{Guid.NewGuid():N}.safegraphs");
        try
        {
            using (DenseIndex index = SmallIndex())
            {
                index.Add(
                    this._pool, [1UL, 2UL], Matrix.RowsOf<float>([.. RowOf(1f), .. RowOf(2f)], Dimensions));
                index.Save(this._pool, path);
            }

            Assert.True(File.Exists(path));

            DenseMetadata metadata = DenseIndex.Metadata(path);
            Assert.Equal((ulong)Dimensions, metadata.Dimensions);
            Assert.Equal(DenseMetric.Euclidean, metadata.Metric);

            DenseOptions deferred = DenseOptions.Default with { Quantization = ScalarKind.F32 };
            using DenseIndex restored = new(deferred);
            Assert.Equal(0UL, restored.Dimensions());

            restored.Load(this._pool, path);
            Assert.Equal((ulong)Dimensions, restored.Dimensions());
            Assert.Equal(2UL, restored.Size());

            MatchesSpan found = MatchesSpan.Packed(queries: 1, wanted: 1);
            restored.Search(this._pool, Matrix.RowsOf<float>(RowOf(2f), Dimensions), found);
            Assert.Equal(USearchKey.FromUInt64(2), found[0].Keys[0]);
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
        index.Add(this._pool, [1UL], Matrix.RowsOf<float>(RowOf(1f), Dimensions));

        byte[] buffer = new byte[index.StorageRequirements()];
        Assert.NotEmpty(buffer);
        index.SaveToBuffer(this._pool, buffer);

        DenseMetadata metadata = DenseIndex.MetadataFromBuffer(buffer);
        Assert.Equal((ulong)Dimensions, metadata.Dimensions);

        using DenseIndex restored = new(SmallOptions());
        restored.LoadFromBuffer(this._pool, buffer);
        Assert.Equal(1UL, restored.Size());
        Assert.True(restored.Contains(1UL));

        Assert.Equal(
            Status.MalformedInput,
            Assert.Throws<USearchException>(
                () => restored.LoadFromBuffer(this._pool, ReadOnlySpan<byte>.Empty)).Status);
    }

    [Fact]
    public void LibraryFailuresArriveAsStatusesRatherThanMessages()
    {
        using DenseIndex index = new(SmallOptions());

        USearchException failure = Assert.Throws<USearchException>(
            () => index.Load(this._pool, Path.Combine(Path.GetTempPath(), "usearch-absent-checkpoint.safegraphs")));

        Assert.Equal(Status.IOFailure, failure.Status);
        Assert.Contains("could not be opened", failure.Message, StringComparison.Ordinal);
    }

    #endregion Serialization

    #region Introspection

    [Fact]
    public void LibraryProbesReportSomething()
    {
        Assert.NotEmpty(USearchCapabilities.Version());
        Assert.NotEmpty(USearchCapabilities.CapabilitiesCompiled());
        Assert.NotEmpty(USearchCapabilities.CapabilitiesAvailable());
    }

    #endregion Introspection
}
