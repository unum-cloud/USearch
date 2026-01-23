using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using Cloud.Unum.USearch;

namespace Cloud.Unum.USearch.Tests;

public class UsearchIndexTests
{
    // Epsilon for real values comparisons
    private const float EqualTolerance = 1e-6f;
    private static readonly RealEqualityComparer<float> s_floatComparer;
    private static readonly RealEqualityComparer<double> s_doubleComparer;

    static UsearchIndexTests()
    {
        s_floatComparer = new RealEqualityComparer<float>(EqualTolerance);
        s_doubleComparer = new RealEqualityComparer<double>(EqualTolerance);
    }

    [Fact]
    public void IndexOptions_InitializesEmptyArguments_ConstructedWithMinimalOrUnknownValues()
    {
        // Arrange
        var controlIndexOptions = new IndexOptions(
            metricKind: MetricKind.Unknown,
            metric: default,
            quantization: ScalarKind.Unknown,
            dimensions: 0,
            connectivity: 0,
            expansionAdd: 0,
            expansionSearch: 0,
            multi: false
        );

        // Act
        var indexOptions = new IndexOptions();

        // Assert
        Assert.Equal(controlIndexOptions, indexOptions);
    }

    [Fact]
    public void PersistAndRestore()
    {

        string pathUsearch = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "savedVectorFolder");
        if (!Directory.Exists(pathUsearch))
        {
            Directory.CreateDirectory(pathUsearch);
        };

        var savedPath = Path.Combine(pathUsearch, "tmp.usearch");

        using var index = new USearchIndex(
            metricKind: MetricKind.Pearson, // Overwrite the default metric
            quantization: ScalarKind.Float64, // Don't quantize at all - max precision
            dimensions: 3,  // Define the number of dimensions in input vectors
            connectivity: 11, // How frequent should the connections in the graph be, optional
            expansionAdd: 15, // Control the recall of indexing, optional
            expansionSearch: 19 // Control the quality of search, optional
        );

        var vector = new double[] { 0.2f, 0.6f, 0.4f };
        index.Add(42, vector);
        index.Save(savedPath);

        Trace.Assert(File.Exists(savedPath));
        Trace.Assert(File.Exists(Path.Combine(pathUsearch, "tmp.usearch")));

        using var indexRestored = new USearchIndex(savedPath);
        Trace.Assert(indexRestored.Size() == 1);
        int matches = indexRestored.Search(vector, 10, out ulong[] keys, out float[] distances);
        Trace.Assert(matches == 1);
        Trace.Assert(keys[0] == 42);
        Trace.Assert(distances[0] <= 0.001f);

        // Clean-up
        File.Delete(savedPath);
    }

    [Fact]
    public void USearch_InitializesWithRequiredArgumentsSetToMinimalOrUnknownValue_UpdatesIndexOptions()
    {
        // Arrange
        const uint ExpectedDimensions = 0;
        const uint ExpectedConnectivity = 16;
        const uint ExpectedCapacity = 0;
        const uint ExpectedSize = 0;

        var initMetricKind = MetricKind.Unknown;
        ulong initDimensions = 0;
        var initQuantization = ScalarKind.Unknown;

        // Act
        using (var index = new USearchIndex(initMetricKind, initQuantization, initDimensions))
        {
            // Assert
            Assert.Equal(ExpectedDimensions, index.Dimensions());
            Assert.Equal(ExpectedConnectivity, index.Connectivity());
            Assert.True(ExpectedCapacity <= index.Capacity());
            Assert.Equal(ExpectedSize, index.Size());
        }
    }

    [Fact]
    public void USearch_InitializesWithIndexOptionsWithMinimalOrUnknownValue_UpdatesIndexOptions()
    {
        // Arrange
        const uint ExpectedDimensions = 0;
        const uint ExpectedConnectivity = 16;
        const uint ExpectedCapacity = 0;
        const uint ExpectedSize = 0;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Unknown,
            metric: default,
            quantization: ScalarKind.Unknown,
            dimensions: 0,
            connectivity: 0,
            expansionAdd: 0,
            expansionSearch: 0,
            multi: false
        );

        // Act
        using (var index = new USearchIndex(indexOptions))
        {
            // Assert
            Assert.Equal(ExpectedDimensions, index.Dimensions());
            Assert.Equal(ExpectedConnectivity, index.Connectivity());
            Assert.True(ExpectedCapacity <= index.Capacity());
            Assert.Equal(ExpectedSize, index.Size());
        }
    }

    [Fact]
    public void Add_FloatVector_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const uint AddKey = 1;
        const uint NonExistentKey = 2;
        const uint ExpectedSize = 1;
        const uint ExpectedCapacity = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions
        );

        var inputVector = GenerateFloatVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(AddKey, inputVector);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.False(index.Contains(NonExistentKey));
            Assert.Equal(ExpectedSize, index.Size());
            Assert.True(ExpectedCapacity <= index.Capacity());
        }
    }

    [Fact]
    public void Add_ByteVector_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const uint AddKey = 1;
        const uint NonExistentKey = 2;
        const uint ExpectedSize = 1;
        const uint ExpectedCapacity = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions
        );

        var inputVector = GenerateByteVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(AddKey, inputVector);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.False(index.Contains(NonExistentKey));
            Assert.Equal(ExpectedSize, index.Size());
            Assert.True(ExpectedCapacity <= index.Capacity());
        }
    }

    [Fact]
    public void Add_DoubleVector_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const uint AddKey = 1;
        const uint NonExistentKey = 2;
        const uint ExpectedSize = 1;
        const uint ExpectedCapacity = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions
        );

        var inputVector = GenerateDoubleVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(AddKey, inputVector);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.False(index.Contains(NonExistentKey));
            Assert.Equal(ExpectedSize, index.Size());
            Assert.True(ExpectedCapacity <= index.Capacity());
        }
    }


    [Fact]
    public void Add_BitsVector_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const uint AddKey = 1;
        const uint NonExistentKey = 2;
        const uint ExpectedSize = 1;
        const uint ExpectedCapacity = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Hamming,
            quantization: ScalarKind.Bits1,
            dimensions: Dimensions
        );

        var inputVector = GenerateBitsVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(AddKey, inputVector);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.False(index.Contains(NonExistentKey));
            Assert.Equal(ExpectedSize, index.Size());
            Assert.True(ExpectedCapacity <= index.Capacity());
        }
    }

    [Fact]
    public void Add_ManyFloatVectorsUnderSameKeySeparatelyInMultiKeyIndex_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int AddFactor = 5;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputVector = GenerateFloatVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            for (int i = 0; i < AddFactor; i++)
            {
                index.Add(AddKey, inputVector);
            }

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.Equal((uint)AddFactor, index.Size());
            Assert.True((uint)AddFactor <= index.Capacity());
        }
    }

    [Fact]
    public void Add_ManyByteVectorsUnderSameKeySeparatelyInMultiKeyIndex_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int AddFactor = 5;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions,
            multi: true
        );

        var inputVector = GenerateByteVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            for (int i = 0; i < AddFactor; i++)
            {
                index.Add(AddKey, inputVector);
            }

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.Equal((uint)AddFactor, index.Size());
            Assert.True((uint)AddFactor <= index.Capacity());
        }
    }

    [Fact]
    public void Add_ManyDoubleVectorsUnderSameKeySeparatelyInMultiKeyIndex_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int AddFactor = 5;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions,
            multi: true
        );

        var inputVector = GenerateDoubleVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            for (int i = 0; i < AddFactor; i++)
            {
                index.Add(AddKey, inputVector);
            }

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.Equal((uint)AddFactor, index.Size());
            Assert.True((uint)AddFactor <= index.Capacity());
        }
    }

    [Fact]
    public void Add_ManyFloatVectorsUnderSameKeyInBatchInMultiKeyIndex_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const uint BatchSize = 5;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
                Enumerable.Repeat(AddKey, (int)BatchSize).ToArray(),
                GenerateManyFloatVectors((int)BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(inputKeys, inputVectors);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.Equal(BatchSize, index.Size());
            Assert.True(BatchSize <= index.Capacity());
            Assert.Equal((int)BatchSize, index.Count(AddKey));
        }
    }

    [Fact]
    public void Add_ManyByteVectorsUnderSameKeyInBatchInMultiKeyIndex_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const uint BatchSize = 5;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, (int)BatchSize).ToArray(),
            GenerateManyByteVectors((int)BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(inputKeys, inputVectors);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.Equal(BatchSize, index.Size());
            Assert.True(BatchSize <= index.Capacity());
            Assert.Equal((int)BatchSize, index.Count(AddKey));
        }
    }

    [Fact]
    public void Add_ManyDoubleVectorsUnderSameKeyInBatchInMultiKeyIndex_UpdatesIndexOptions()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const uint BatchSize = 5;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
                Enumerable.Repeat(AddKey, (int)BatchSize).ToArray(),
                GenerateManyDoubleVectors((int)BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            index.Add(inputKeys, inputVectors);

            // Assert
            Assert.True(index.Contains(AddKey));
            Assert.Equal(BatchSize, index.Size());
            Assert.True(BatchSize <= index.Capacity());
            Assert.Equal((int)BatchSize, index.Count(AddKey));
        }
    }

    [Fact]
    public void Get_ManyFloatVectorsUnderSameKeyInMultiKeyIndex_ReturnsCorrectValue()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int RetrieveCount = 5;
        const int BatchSize = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, BatchSize).ToArray(),
            GenerateManyFloatVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            int foundVectorsCount = index.Get(AddKey, RetrieveCount, out float[][] retrievedVectors);

            // Assert
            Assert.Equal(RetrieveCount, foundVectorsCount);
        }
    }

    [Fact]
    public void Get_ManyByteVectorsUnderSameKeyInMultiKeyIndex_ReturnsCorrectValue()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int RetrieveCount = 5;
        const int BatchSize = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, BatchSize).ToArray(),
            GenerateManyByteVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            int foundVectorsCount = index.Get(AddKey, RetrieveCount, out float[][] retrievedVectors);

            // Assert
            Assert.Equal(RetrieveCount, foundVectorsCount);
        }
    }

    [Fact]
    public void Get_ManyBitsVectorsUnderSameKeyInMultiKeyIndex_ReturnsCorrectValue()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int RetrieveCount = 5;
        const int BatchSize = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Hamming,
            quantization: ScalarKind.Bits1,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, BatchSize).ToArray(),
            GenerateManyBitsVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            int foundVectorsCount = index.Get(AddKey, RetrieveCount, out float[][] retrievedVectors);

            // Assert
            Assert.Equal(RetrieveCount, foundVectorsCount);
        }
    }

    [Fact]
    public void Get_ManyDoubleVectorsUnderSameKeyInMultiKeyIndex_ReturnsCorrectCountValue()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int RetrieveCount = 5;
        const int BatchSize = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
                Enumerable.Repeat(AddKey, BatchSize).ToArray(),
                GenerateManyDoubleVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            int foundVectorsCount = index.Get(AddKey, RetrieveCount, out double[][] retrievedVectors);

            // Assert
            Assert.Equal(RetrieveCount, foundVectorsCount);
        }
    }

    [Fact]
    public void Get_AfterAddingFloatVector_ReturnsEqualVector()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateFloatVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            index.Get(inputKey, out float[] retrievedVector);

            // Assert
            Assert.Equal(inputVector, retrievedVector, s_floatComparer);
        }
    }

    [Fact]
    public void Get_AfterAddingByteVector_ReturnsEqualVector()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateByteVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            index.Get(inputKey, out sbyte[] retrievedVector);

            // Assert
            Assert.Equal(inputVector, retrievedVector);
        }
    }

    [Fact]
    public void Get_AfterAddingDoubleVector_ReturnsEqualVector()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateDoubleVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            index.Get(inputKey, out double[] retrievedVector);

            // Assert
            Assert.Equal(inputVector, retrievedVector, s_doubleComparer);
        }
    }

    [Fact]
    public void Add_AddingTwoFloatVectorsUnderSameKey_ThrowsException()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateFloatVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            Action actual = () => index.Add(inputKey, inputVector);

            // Assert
            Assert.Throws<USearchException>(actual);
        }
    }

    [Fact]
    public void Add_AddingTwoByteVectorsUnderSameKey_ThrowsException()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateByteVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            Action actual = () => index.Add(inputKey, inputVector);

            // Assert
            Assert.Throws<USearchException>(actual);
        }
    }

    [Fact]
    public void Add_AddingTwoDoubleVectorsUnderSameKey_ThrowsException()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateDoubleVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            Action actual = () => index.Add(inputKey, inputVector);

            // Assert
            Assert.Throws<USearchException>(actual);
        }
    }

    [Fact]
    public void Remove_InsertedFloatVector_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int ExpectedRemoveReturn = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateFloatVector((int)Dimensions));


        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            var removedCount = index.Remove(AddKey);

            // Assert
            Assert.Equal(ExpectedRemoveReturn, removedCount);
        }
    }

    [Fact]
    public void Remove_InsertedByteVector_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int ExpectedRemoveReturn = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (AddKey, GenerateByteVector((int)Dimensions));


        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            var removedCount = index.Remove(AddKey);

            // Assert
            Assert.Equal(ExpectedRemoveReturn, removedCount);
        }
    }

    [Fact]
    public void Remove_InsertedDoubleVector_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong TestAddKey = 1;
        const int ExpectedRemoveReturn = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions
        );

        (var inputKey, var inputVector) = (TestAddKey, GenerateDoubleVector((int)Dimensions));

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKey, inputVector);

            // Act
            var removedCount = index.Remove(TestAddKey);

            // Assert
            Assert.Equal(ExpectedRemoveReturn, removedCount);
        }
    }

    [Fact]
    public void Remove_InsertedManyFloatVectorsUnderSameKeyInMultiKeyIndex_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int BatchSize = 2;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, BatchSize).ToArray(),
            GenerateManyFloatVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var removedCount = index.Remove(AddKey);

            // Assert
            Assert.Equal(BatchSize, removedCount);
        }
    }

    [Fact]
    public void Remove_InsertedManyByteVectorsUnderSameKeyInMultiKeyIndex_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int BatchSize = 2;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Int8,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, BatchSize).ToArray(),
            GenerateManyByteVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var removedCount = index.Remove(AddKey);

            // Assert
            Assert.Equal(BatchSize, removedCount);
        }
    }

    [Fact]
    public void Remove_InsertedManyDoubleVectorUnderSameKeyInMultiKeyIndex_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const int BatchSize = 2;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float64,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Repeat(AddKey, BatchSize).ToArray(),
            GenerateManyDoubleVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var removedCount = index.Remove(AddKey);

            // Assert
            Assert.Equal(BatchSize, removedCount);
        }
    }

    [Fact]
    public void Get_NonExistentKey_ReturnsZeroCountAndEmptyVector()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong NonExistentKey = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions
        );

        using (var index = new USearchIndex(indexOptions))
        {
            // Act
            var retrievedCount = index.Get(NonExistentKey, out float[] retrievedVector);

            // Assert
            Assert.True(retrievedCount == 0);
            Assert.Null(retrievedVector);
        }
    }

    [Fact]
    public void Rename_ExistingKey_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong TestAddKey = 1;
        const ulong TestRenameKey = 2;
        const int ExpectedRenamedCount = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: false
        );

        var inputVector = GenerateFloatVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(TestAddKey, inputVector);

            // Act
            var renamedCount = index.Rename(TestAddKey, TestRenameKey);

            // Assert
            Assert.Equal(ExpectedRenamedCount, renamedCount);
        }
    }

    [Fact]
    public void Rename_ExistingKey_UpdatesOptionsProperly()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const ulong RenameToKey = 2;
        const ulong ExpectedSize = 1;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: false
        );

        var inputVector = GenerateFloatVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(AddKey, inputVector);

            // Act
            index.Rename(AddKey, RenameToKey);

            // Assert
            Assert.False(index.Contains(AddKey));
            Assert.True(index.Contains(RenameToKey));
            Assert.Equal(ExpectedSize, index.Size());
        }
    }

    [Fact]
    public void Rename_ExistingKeyInMultiKeyIndex_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const ulong RenameKey = 2;
        const int BatchSize = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputKeys = Enumerable.Repeat((int)AddKey, BatchSize).Select(x => (ulong)x).ToArray();
        var inputVectors = GenerateManyFloatVectors(BatchSize, (int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var vectorsCount = index.Rename(AddKey, RenameKey);

            // Assert
            Assert.Equal(BatchSize, vectorsCount);
        }
    }

    [Fact]
    public void Rename_ExistingKeyInMultiKeyIndex_UpdatesOptionsProperly()
    {
        // Arrange
        const uint Dimensions = 10;
        const ulong AddKey = 1;
        const ulong RenameToKey = 2;
        const int BatchSize = 2;
        const ulong ExpectedSize = 2;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        (var inputKeys, var inputVectors) = (
            Enumerable.Range((int)AddKey, BatchSize).Select(x => (ulong)x).ToArray(),
            GenerateManyFloatVectors(BatchSize, (int)Dimensions)
        );

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(
                inputKeys,
                inputVectors
            );

            // Act
            var vectorsCount = index.Rename(AddKey, RenameToKey);

            // Assert
            Assert.False(index.Contains(AddKey));
            Assert.True(index.Contains(RenameToKey));
            Assert.Equal(ExpectedSize, index.Size());
        }
    }

    [Fact]
    public void Rename_ExistingKeyToExistingKeyInMultiKeyIndex_ReturnsInsertedVectorsCount()
    {
        // Arrange
        const uint Dimensions = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputKeys = new ulong[] { 1, 1, 1, 2, 2 };
        var inputVectors = GenerateManyFloatVectors(5, (int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var renamedCount = index.Rename(1, 2);

            // Assert
            Assert.Equal(3, renamedCount);
        }
    }

    [Fact]
    public void Rename_ExistingKeyToExistingKeyInMultiKeyIndex_UpdatesOptionsProperly()
    {
        // Arrange
        const uint Dimensions = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputKeys = new ulong[] { 1, 1, 1, 2, 2 };
        var inputVectors = GenerateManyFloatVectors(5, (int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var renamedCount = index.Rename(1, 2);

            // Assert
            Assert.False(index.Contains(1));
            Assert.True(index.Contains(2));
            Assert.Equal(5UL, index.Size());
        }
    }

    [Fact]
    public void Search_UsingL2sq_ReturnsCorrectDistances()
    {
        // Arrange
        const uint Dimensions = 2;
        const int StartKey = 1;
        const int VectorsCount = 5;
        const int MaxSearchCount = 2;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputKeys = Enumerable.Range(StartKey, VectorsCount).Select(i => (ulong)i).ToArray();
        var inputVectors = new float[][]
        {
            new float[] { -0.5f, -0.6f },
            new float[] { -0.1f, -0.1f },
            new float[] { 0.2f, 0.2f },
            new float[] { 0.7f, 0.8f },
            new float[] { 0.9f, 1.0f }
        };
        var queryVector = new float[] { 0.0f, 0.0f };

        using (var index = new USearchIndex(MetricKind.L2sq, ScalarKind.Float32, 2))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var matchCount = index.Search(queryVector, MaxSearchCount, out ulong[] returnedKeys, out float[] returnedDistances);
            _ = index.Get(returnedKeys[0], out float[] firstVector);
            _ = index.Get(returnedKeys[1], out float[] secondVector);

            // Assert
            Assert.Equal(MaxSearchCount, matchCount);
            Assert.Equal(new ulong[] { 2UL, 3UL }, returnedKeys);
            Assert.Equal(L2sqDistance.L2sq<float>(queryVector, firstVector), returnedDistances[0], EqualTolerance);
            Assert.Equal(L2sqDistance.L2sq<float>(queryVector, secondVector), returnedDistances[1], EqualTolerance);
        }
    }

    [Fact]
    public void Search_AfterRemovingKey_ReturnsCorrectMatches()
    {
        // Arrange
        const uint Dimensions = 2;
        const ulong StartKey = 1;
        const int VectorsCount = 10;
        const ulong RemoveKey = 3;
        const int MaxSearchCount = 9;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputKeys = Enumerable.Range((int)StartKey, VectorsCount).Select(x => (ulong)x).ToArray();
        var inputVectors = GenerateManyFloatVectors(VectorsCount, (int)Dimensions);
        var queryVector = GenerateFloatVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);
            index.Remove(RemoveKey);

            // Act
            var matchCount = index.Search(queryVector, MaxSearchCount, out ulong[] returnedKeys, out float[] returnedDistances);

            // Assert
            Assert.Equal(VectorsCount - 1, matchCount);
            Assert.Equivalent(inputKeys.Where(x => x != RemoveKey).ToArray(), returnedKeys);
        }
    }

    [Fact]
    public void Search_CountExceedsIndexSize_ReturnsAllMatches()
    {
        // Arrange
        const uint Dimensions = 2;
        const ulong StartKey = 1;
        const int VectorsCount = 2;
        const int MaxSearchCount = 10;

        var indexOptions = new IndexOptions(
            metricKind: MetricKind.Cos,
            quantization: ScalarKind.Float32,
            dimensions: Dimensions,
            multi: true
        );

        var inputKeys = Enumerable.Range((int)StartKey, VectorsCount).Select(x => (ulong)x).ToArray();
        var inputVectors = GenerateManyFloatVectors(VectorsCount, (int)Dimensions);
        var queryVector = GenerateFloatVector((int)Dimensions);

        using (var index = new USearchIndex(indexOptions))
        {
            index.Add(inputKeys, inputVectors);

            // Act
            var matchCount = index.Search(queryVector, MaxSearchCount, out ulong[] returnedKeys, out float[] returnedDistances);

            // Assert
            Assert.Equal(VectorsCount, matchCount);
            Assert.Equivalent(inputKeys, returnedKeys);
        }
    }

    #region private ================================================================================

    private static float[] GenerateFloatVector(int vectorLength)
    {
        return Enumerable.Range(0, vectorLength).Select(i => (float)i).ToArray();
    }

    private static float[][] GenerateManyFloatVectors(int n, int vectorLength)
    {
        var result = new float[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = Enumerable.Range(0, vectorLength).Select(i => (float)i).ToArray();
        }
        return result;
    }

    private static sbyte[] GenerateByteVector(int vectorLength)
    {
        return Enumerable.Range(0, vectorLength).Select(i => (sbyte)i).ToArray();
    }

    private static byte[] GenerateBitsVector(int vectorLength)
    {
        return Enumerable.Range(0, vectorLength).Select(i => (byte)i).ToArray();
    }

    private static sbyte[][] GenerateManyByteVectors(int n, int vectorLength)
    {
        var result = new sbyte[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = Enumerable.Range(0, vectorLength).Select(i => (sbyte)i).ToArray();
        }
        return result;
    }

    private static byte[][] GenerateManyBitsVectors(int n, int vectorLength)
    {
        var result = new byte[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = Enumerable.Range(0, vectorLength).Select(i => (byte)i).ToArray();
        }
        return result;
    }

    private static double[] GenerateDoubleVector(int n)
    {
        return Enumerable.Range(0, n).Select(i => (double)i).ToArray();
    }

    private static float[][] GenerateManyDoubleVectors(int n, int vectorLength)
    {
        var result = new float[n][];
        for (int i = 0; i < n; i++)
        {
            result[i] = Enumerable.Range(0, vectorLength).Select(i => (float)i).ToArray();
        }
        return result;
    }

    #endregion
}

internal sealed class RealEqualityComparer<T> : IEqualityComparer<T> where T : unmanaged
{
    private readonly T _threshold;

    public RealEqualityComparer(T threshold) => this._threshold = threshold;

    public bool Equals(T a, T b)
    {
        return Math.Abs((dynamic)a - (dynamic)b) < this._threshold;
    }

    public int GetHashCode(T obj)
    {
        return obj.GetHashCode();
    }
}

internal static class L2sqDistance
{
    public static double L2sq<TNumber>(ReadOnlySpan<TNumber> x, ReadOnlySpan<TNumber> y) where TNumber : unmanaged
    {
        if (typeof(TNumber) == typeof(float))
        {
            ReadOnlySpan<float> floatSpanX = MemoryMarshal.Cast<TNumber, float>(x);
            ReadOnlySpan<float> floatSpanY = MemoryMarshal.Cast<TNumber, float>(y);
            return L2sqImplementation(floatSpanX, floatSpanY);
        }
        else if (typeof(TNumber) == typeof(double))
        {
            ReadOnlySpan<double> doubleSpanX = MemoryMarshal.Cast<TNumber, double>(x);
            ReadOnlySpan<double> doubleSpanY = MemoryMarshal.Cast<TNumber, double>(y);
            return L2sqImplementation(doubleSpanX, doubleSpanY);
        }
        else
        {
            throw new NotSupportedException();
        }
    }

    public static double L2sq<TNumber>(TNumber[] x, TNumber[] y)
    where TNumber : unmanaged
    {
        return L2sq(new ReadOnlySpan<TNumber>(x), new ReadOnlySpan<TNumber>(y));
    }

    #region private ================================================================================

    private static double L2sqImplementation(ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        double distanceSum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            distanceSum += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return distanceSum;
    }

    private static double L2sqImplementation(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        double distanceSum = 0;
        for (int i = 0; i < x.Length; i++)
        {
            distanceSum += (x[i] - y[i]) * (x[i] - y[i]);
        }
        return distanceSum;
    }

    #endregion
}
