using System.Runtime.InteropServices;

namespace Cloud.Unum.USearch.Tests;

/// <summary>
/// The boundary of the binding: marshalling, the options round trip through the residency union,
/// key byte order, stride arithmetic, handle ownership, and truncation reporting. That vector
/// search works is the C library's job, and is not re-tested here.
/// </summary>
public class USearchIndexTests
{
    private const ulong Dimensions = 4;

    private static DenseOptions SmallOptions() => new()
    {
        Dimensions = Dimensions,
        Metric = DenseMetric.Euclidean,
        Quantization = ScalarKind.F32,
        Connectivity = 11,
        Residency = Residency.Default,
    };

    private static USearchIndex SmallIndex()
    {
        USearchIndex index = new(SmallOptions());
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
        using USearchIndex index = SmallIndex();
        USearchKey wide = USearchKey.Parse("0123abcd-4567-89ef-fedc-ba9876543210");

        Assert.Equal(1UL, index.Add<float>(wide, RowOf(1f)));
        Assert.Equal(1UL, index.Count(wide));
        Assert.True(index.Contains(wide));
        Assert.False(index.Contains(USearchKey.FromUInt64(1)));

        SearchResults found = index.Search<float>(RowOf(1f), wanted: 1);
        Assert.Equal(wide, found.KeysOf(0)[0]);
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
    public void RamOptionsSurviveTheRoundTrip()
    {
        DenseOptions options = DenseOptions.Default;
        options.Dimensions = 96;
        options.Connectivity = 24;
        options.DimensionPadding = DimensionPadding.SimdQuantumDimensions;
        options.Residency = new Residency
        {
            Kind = ResidencyKind.Ram,
            Threads = 12,
            HugePages = HugePages.RequireHugePages,
        };

        DenseOptions reloaded = options.Reload();

        Assert.Equal(96UL, reloaded.Dimensions);
        Assert.Equal(24UL, reloaded.Connectivity);
        Assert.Equal(DimensionPadding.SimdQuantumDimensions, reloaded.DimensionPadding);
        Assert.Equal(DenseMetric.Angular, reloaded.Metric);
        Assert.Equal(ScalarKind.BF16, reloaded.Quantization);
        Assert.Equal(ResidencyKind.Ram, reloaded.Residency.Kind);
        Assert.Equal(12UL, reloaded.Residency.Threads);
        Assert.Equal(HugePages.RequireHugePages, reloaded.Residency.HugePages);
    }

    [Fact]
    public void VramAndNvmeOptionsSurviveTheRoundTrip()
    {
        DenseOptions vram = SmallOptions();
        vram.Residency = Residency.OnDevice(-1, Addressing.UnifiedAddressing);

        DenseOptions reloadedVram = vram.Reload();
        Assert.Equal(ResidencyKind.Vram, reloadedVram.Residency.Kind);
        Assert.Equal(-1, reloadedVram.Residency.Device);
        Assert.Equal(Addressing.UnifiedAddressing, reloadedVram.Residency.Addressing);

        DenseOptions nvme = SmallOptions();
        nvme.Residency = new Residency
        {
            Kind = ResidencyKind.Nvme,
            SectorBytes = 512,
            QueueDepth = 64,
            IOMode = IOMode.PageCachedIO,
        };

        DenseOptions reloadedNvme = nvme.Reload();
        Assert.Equal(ResidencyKind.Nvme, reloadedNvme.Residency.Kind);
        Assert.Equal(512u, reloadedNvme.Residency.SectorBytes);
        Assert.Equal(64u, reloadedNvme.Residency.QueueDepth);
        Assert.Equal(IOMode.PageCachedIO, reloadedNvme.Residency.IOMode);
    }

    [Fact]
    public void UnknownTagsAreRejectedRatherThanGuessedAt()
    {
        DenseOptions unknownResidency = SmallOptions();
        unknownResidency.Residency = new Residency { Kind = (ResidencyKind)7 };
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => unknownResidency.Reload()).Status);

        DenseOptions unknownMetric = SmallOptions();
        unknownMetric.Metric = (DenseMetric)9;
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => unknownMetric.Reload()).Status);

        DenseOptions unknownQuantization = SmallOptions();
        unknownQuantization.Quantization = (ScalarKind)44;
        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => unknownQuantization.Reload()).Status);
    }

    [Fact]
    public void OptionsReachTheLibraryAsTheyWereWritten()
    {
        using USearchIndex index = new(SmallOptions());

        Assert.Equal(Dimensions, index.Dimensions());
        Assert.Equal(11UL, index.Connectivity());
        Assert.Equal(0UL, index.Size());
        Assert.NotEmpty(index.Capabilities());
    }

    #endregion Options & Residency

    #region Geometry

    [Fact]
    public void UnfrozenGeometryIsRefusedRatherThanSizedToZero()
    {
        DenseOptions deferred = DenseOptions.Default;
        deferred.Quantization = ScalarKind.F32;
        using USearchIndex index = new(deferred);

        Assert.Equal(0UL, index.Dimensions());
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(() => index.Add<float>(1UL, RowOf(1f))).Status);
        Assert.Equal(
            Status.Uninitialized,
            Assert.Throws<USearchException>(() => index.Search<float>(RowOf(1f), 1)).Status);
    }

    [Fact]
    public void MismatchedRowWidthsAreRefused()
    {
        using USearchIndex index = SmallIndex();
        float[] tooNarrow = [1f, 2f, 3f];

        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(() => index.Add<float>(1UL, tooNarrow)).Status);
        Assert.Equal(
            Status.ShapeMismatch,
            Assert.Throws<USearchException>(() => index.Search<float>(tooNarrow, 1)).Status);
    }

    [Fact]
    public void UnnameableScalarTypesAreRefused()
    {
        using USearchIndex index = SmallIndex();
        short[] row = [1, 2, 3, 4];

        Assert.Equal(
            Status.UnknownKind,
            Assert.Throws<USearchException>(() => index.Add<short>(1UL, row)).Status);
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

        using USearchIndex index = SmallIndex();
        float[] twoQueries = [.. RowOf(1f), .. RowOf(2f)];
        Assert.Equal(
            Status.Overflow,
            Assert.Throws<USearchException>(() => index.Search<float>(twoQueries, int.MaxValue)).Status);
    }

    #endregion Geometry

    #region Batches & Strides

    [Fact]
    public void BatchedRowsAndQueriesKeepTheirLanes()
    {
        using USearchIndex index = SmallIndex();
        USearchKey[] keys = [1UL, 2UL, 3UL];
        float[] rows = [.. RowOf(10f), .. RowOf(20f), .. RowOf(30f)];

        Assert.Equal(3UL, index.Add<float>(keys, rows));
        Assert.Equal(3UL, index.Size());

        float[] queries = [.. RowOf(30f), .. RowOf(10f)];
        SearchResults found = index.Search<float>(queries, wanted: 2);

        Assert.Equal(2, found.Queries);
        Assert.Equal(2, found.CountOf(0));
        Assert.Equal(2, found.CountOf(1));
        Assert.Equal(USearchKey.FromUInt64(3), found.KeysOf(0)[0]);
        Assert.Equal(USearchKey.FromUInt64(1), found.KeysOf(1)[0]);
        Assert.Equal(2, found.DistancesOf(0).Length);
        Assert.Equal(0d, found.DistancesOf(0)[0], 6);
    }

    [Fact]
    public void SearchesReportFewerMatchesThanWantedWithoutReadingPastThem()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>(1UL, RowOf(1f));

        SearchResults found = index.Search<float>(RowOf(1f), wanted: 8);

        Assert.Equal(8, found.Wanted);
        Assert.Equal(1, found.CountOf(0));
        Assert.Single(found.KeysOf(0).ToArray());
        Assert.Single(found.DistancesOf(0).ToArray());
    }

    [Fact]
    public void ReportedCountsAboveTheRowWidthAreClamped()
    {
        // The sibling verbs disagree on what a count means, so the slicing clamps rather than
        // trusting it: entries past the row width were never written by anyone.
        SearchResults results = new(new USearchKey[4], new double[4], [(nuint)5, (nuint)1], wanted: 2);

        Assert.Equal(2, results.CountOf(0));
        Assert.Equal(2, results.KeysOf(0).Length);
        Assert.Equal(1, results.CountOf(1));
        Assert.Equal(1, results.DistancesOf(1).Length);
    }

    [Fact]
    public void TraversalStatisticsAreReportedAndAccumulate()
    {
        using USearchIndex index = SmallIndex();
        TraversalStats insertion = new();
        index.Add<float>([1UL, 2UL], [.. RowOf(1f), .. RowOf(2f)], stats: insertion);

        TraversalStats query = new();
        index.Search<float>(RowOf(1f), wanted: 2, stats: query);

        Assert.True(insertion.VisitedMembers > 0);
        Assert.True(query.ComputedDistances > 0);

        ulong before = query.ComputedDistances;
        query.Accumulate(insertion);
        Assert.Equal(before + insertion.ComputedDistances, query.ComputedDistances);
    }

    [Fact]
    public void ScalarKindsOtherThanFloatCross()
    {
        DenseOptions options = SmallOptions();
        options.Quantization = ScalarKind.I8;
        using USearchIndex index = new(options);
        index.Reserve(4);

        sbyte[] row = [1, 2, 3, 4];
        Assert.Equal(1UL, index.Add<sbyte>(1UL, row));

        Half[] halves = [(Half)1f, (Half)2f, (Half)3f, (Half)4f];
        Assert.Equal(1UL, index.Add<Half>(2UL, halves));

        double[] retrieved = new double[Dimensions];
        Assert.Equal(1UL, index.Get<double>(2UL, retrieved));
        Assert.Equal(new double[] { 1, 2, 3, 4 }, retrieved);
    }

    #endregion Batches & Strides

    #region Retrieval

    [Fact]
    public void GetReportsTheTrueCountAndTruncatesTheCopy()
    {
        using USearchIndex index = SmallIndex();
        USearchKey shared = 5UL;
        index.Add<float>([shared, shared, shared], [.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]);

        Assert.Equal(3UL, index.Count(shared));

        float[] roomForOne = new float[Dimensions];
        Assert.Equal(3UL, index.Get<float>(shared, roomForOne));
        Assert.Equal(RowOf(1f), roomForOne);

        float[] roomForAll = new float[3 * Dimensions];
        Assert.Equal(3UL, index.Get<float>(shared, roomForAll));
        Assert.Equal(RowOf(3f), roomForAll[(int)(2 * Dimensions)..]);
    }

    [Fact]
    public void GetOfAnAbsentKeyWritesNothing()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>(1UL, RowOf(1f));

        float[] untouched = new float[Dimensions];
        Assert.Equal(0UL, index.Get<float>(99UL, untouched));
        Assert.Equal(0UL, index.Count(99UL));
        Assert.Equal(new float[Dimensions], untouched);
    }

    #endregion Retrieval

    #region Filtered Search

    [Fact]
    public void FilteredSearchesAdmitOnlyWhatThePredicateApproves()
    {
        using USearchIndex index = SmallIndex();
        USearchKey[] keys = [1UL, 2UL, 3UL, 4UL];
        float[] rows = [.. RowOf(1f), .. RowOf(2f), .. RowOf(3f), .. RowOf(4f)];
        index.Add<float>(keys, rows);

        SearchResults found = index.SearchIf<float>(
            RowOf(1f), wanted: 4, admits: key => key.TryGetUInt64(out ulong identity) && identity % 2 == 0);

        Assert.Equal(2, found.CountOf(0));
        foreach (USearchKey key in found.KeysOf(0))
        {
            Assert.True(key.TryGetUInt64(out ulong identity));
            Assert.Equal(0UL, identity % 2);
        }
    }

    [Fact]
    public void PredicateFailuresSurfaceAfterTheCallRatherThanUnwindingThroughIt()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>([1UL, 2UL], [.. RowOf(1f), .. RowOf(2f)]);

        USearchException failure = Assert.Throws<USearchException>(
            () => index.SearchIf<float>(RowOf(1f), wanted: 2, admits: _ => throw new InvalidOperationException()));

        Assert.Equal(Status.Internal, failure.Status);
        Assert.IsType<InvalidOperationException>(failure.InnerException);
    }

    #endregion Filtered Search

    #region Handles

    [Fact]
    public void MergeConsumesTheDonor()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>(1UL, RowOf(1f));

        USearchIndex donor = SmallIndex();
        donor.Add<float>(2UL, RowOf(2f));
        index.Merge(donor);

        Assert.Equal(2UL, index.Size());
        Assert.True(index.Contains(2UL));
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => donor.Size()).Status);

        // A later dispose of the consumed donor must not double-free.
        donor.Dispose();
        donor.Dispose();
    }

    [Fact]
    public void DisposedHandlesRefuseEveryVerb()
    {
        USearchIndex index = SmallIndex();
        index.Dispose();
        index.Dispose();

        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Size()).Status);
        Assert.Equal(Status.Uninitialized, Assert.Throws<USearchException>(() => index.Count(1UL)).Status);
    }

    [Fact]
    public void CursorsWalkTheSnapshotTheyCaptured()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>([1UL, 2UL, 3UL], [.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]);

        using USearchCursor cursor = index.Snapshot();
        index.Add<float>(4UL, RowOf(4f));

        USearchKey[] keys = new USearchKey[2];
        float[] vectors = new float[2 * Dimensions];
        Assert.Equal(2, cursor.Next<float>(keys, vectors));
        Assert.Equal(USearchKey.FromUInt64(1), keys[0]);
        Assert.Equal(RowOf(2f), vectors[(int)Dimensions..]);

        Assert.Equal(1, cursor.Next(keys));
        Assert.Equal(USearchKey.FromUInt64(3), keys[0]);
        Assert.Equal(0, cursor.Next(keys));
    }

    [Fact]
    public void FindersStreamMatchesAcrossPulls()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>([1UL, 2UL, 3UL], [.. RowOf(1f), .. RowOf(2f), .. RowOf(3f)]);

        using USearchFinder finder = index.Find<float>(RowOf(1f));
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
            using (USearchIndex index = SmallIndex())
            {
                index.Add<float>([1UL, 2UL], [.. RowOf(1f), .. RowOf(2f)]);
                index.Save(path);
            }

            Assert.True(File.Exists(path));

            DenseOptions metadata = USearchIndex.Metadata(path);
            Assert.Equal(Dimensions, metadata.Dimensions);
            Assert.Equal(DenseMetric.Euclidean, metadata.Metric);

            DenseOptions deferred = DenseOptions.Default;
            deferred.Quantization = ScalarKind.F32;
            using USearchIndex restored = new(deferred);
            Assert.Equal(0UL, restored.Dimensions());

            restored.Load(path);
            Assert.Equal(Dimensions, restored.Dimensions());
            Assert.Equal(2UL, restored.Size());
            Assert.Equal(USearchKey.FromUInt64(2), restored.Search<float>(RowOf(2f), 1).KeysOf(0)[0]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CheckpointsRoundTripThroughABuffer()
    {
        using USearchIndex index = SmallIndex();
        index.Add<float>(1UL, RowOf(1f));

        byte[] buffer = new byte[index.SerializedLength()];
        Assert.NotEmpty(buffer);
        index.SaveToBuffer(buffer);

        DenseOptions metadata = USearchIndex.MetadataFromBuffer(buffer);
        Assert.Equal(Dimensions, metadata.Dimensions);

        using USearchIndex restored = new(SmallOptions());
        restored.LoadFromBuffer(buffer);
        Assert.Equal(1UL, restored.Size());
        Assert.True(restored.Contains(1UL));

        Assert.Equal(
            Status.MalformedInput,
            Assert.Throws<USearchException>(() => restored.LoadFromBuffer(ReadOnlySpan<byte>.Empty)).Status);
    }

    [Fact]
    public void LibraryFailuresArriveAsStatusesRatherThanMessages()
    {
        using USearchIndex index = new(SmallOptions());

        USearchException failure = Assert.Throws<USearchException>(
            () => index.Load(Path.Combine(Path.GetTempPath(), "usearch-absent-checkpoint.safegraphs")));

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
