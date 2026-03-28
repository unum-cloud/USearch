using System;
using System.IO;
using System.Linq;
using Xunit;

namespace Cloud.Unum.USearch.Tests;

public class IndexTests
{
    [Fact]
    public void TestAddAndSearch()
    {
        var options = new IndexOptions(
            metricKind: MetricKind.L2sq,
            quantization: ScalarKind.Float32,
            dimensions: 3,
            connectivity: 16,
            expansionAdd: 128,
            expansionSearch: 64
        );
        using var index = new USearchIndex<float>(options);

        index.Add(1, new float[] { 1.0f, 0.0f, 0.0f });
        index.Add(2, new float[] { 0.0f, 1.0f, 0.0f });
        index.Add(3, new float[] { 0.0f, 0.0f, 1.0f });

        float[] query = { 1.0f, 0.0f, 0.0f };
        int count = index.Search(query, 3, out ulong[] keys, out float[] distances);

        Assert.Equal(3, count);
        Assert.Equal(1UL, keys[0]);
        Assert.Equal(0.0f, distances[0]);

        Assert.Contains(2UL, keys);
        Assert.Contains(3UL, keys);
    }

    [Fact]
    public void TestSaveLoad()
    {
        var options = new IndexOptions(
            metricKind: MetricKind.L2sq,
            quantization: ScalarKind.Float32,
            dimensions: 3,
            connectivity: 16
        );
        string path = "test_index.usearch";
        if(File.Exists(path)) File.Delete(path);

        using (var index = new USearchIndex<float>(options))
        {
            index.Add(1, new float[] { 1.0f, 2.0f, 3.0f });
            index.Add(2, new float[] { 4.0f, 5.0f, 6.0f });
            index.Save(path);
        }

        Assert.True(File.Exists(path));

        using (var loadedIndex = new USearchIndex<float>(path))
        {
            Assert.Equal(2, loadedIndex.Size());
            Assert.Equal(3UL, loadedIndex.Dimensions());

            float[] query = { 1.0f, 2.0f, 3.0f };
            loadedIndex.Search(query, 1, out ulong[] keys, out float[] distances);

            Assert.Single(keys);
            Assert.Equal(1UL, keys[0]);
            Assert.Equal(0.0f, distances[0]);
        }

        File.Delete(path);
    }

    [Fact]
    public void TestConcurrency()
    {
        var options = new IndexOptions(
            metricKind: MetricKind.L2sq,
            quantization: ScalarKind.Float32,
            dimensions: 128,
            connectivity: 16,
            expansionAdd: 128,
            expansionSearch: 64
        );
        using var index = new USearchIndex<float>(options);

        int n = 1000;
        var rnd = new Random(42);
        var vectors = new float[n][];
        for(int i=0; i<n; i++)
        {
            vectors[i] = new float[128];
            for(int j=0; j<128; j++) vectors[i][j] = (float)rnd.NextDouble();
        }

        // Parallel Add
        System.Threading.Tasks.Parallel.For(0, n, i => {
            index.Add((ulong)i, vectors[i]);
        });

        Assert.Equal(n, index.Size());

        // Parallel Search
        System.Threading.Tasks.Parallel.For(0, n, i => {
            int count = index.Search(vectors[i], 1, out ulong[] keys, out float[] dists);
            Assert.True(count >= 1);
            Assert.Equal((ulong)i, keys[0]);
            Assert.True(Math.Abs(dists[0]) < 1e-5);
        });
    }
}
