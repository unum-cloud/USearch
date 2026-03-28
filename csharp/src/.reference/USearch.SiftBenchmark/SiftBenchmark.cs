using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using Cloud.Unum.USearch;

namespace USearch.SiftBenchmark
{
    public class SiftBenchmark
    {
        private static float[][]? _baseVectors;
        private static float[][]? _queryVectors;
        private static int[][]? _groundTruth;

        private static USearchIndex<float>? _cachedGraph;
        private static (int M, int EfConstruction) _cachedGraphParams;

        private USearchIndex<float>? _graph;

        [Params(16)]
        public int M { get; set; }

        [Params(200)]
        public int EfConstruction { get; set; }

        [Params(50, 100, 200)]
        public int EfSearch { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            string workingDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);

            Dataset.DownloadAndExtractAsync(workingDir).GetAwaiter().GetResult();

            string siftDir = Path.Combine(workingDir, "sift");
            if (_baseVectors == null)
            {
                Console.WriteLine("Reading dataset...");
                _baseVectors = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_base.fvecs"));
                _queryVectors = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_query.fvecs"));
                _groundTruth = Dataset.ReadIvecs(Path.Combine(siftDir, "sift_groundtruth.ivecs"));
                Console.WriteLine($"Loaded {_baseVectors.Length} base vectors");
                Console.WriteLine($"Loaded {_queryVectors.Length} query vectors");
            }

            if (_cachedGraph == null || _cachedGraphParams.M != M || _cachedGraphParams.EfConstruction != EfConstruction)
            {
                var options = new IndexOptions(
                    metricKind: MetricKind.L2sq,
                    quantization: ScalarKind.Float32,
                    dimensions: (ulong)_baseVectors![0].Length,
                    connectivity: (ulong)M,
                    expansionAdd: (ulong)EfConstruction,
                    expansionSearch: (ulong)EfSearch
                );

                Console.WriteLine($"Building graph with M={M}, EfConstruction={EfConstruction}...");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var graph = new USearchIndex<float>(options);
                for (int i = 0; i < _baseVectors.Length; i++)
                {
                    graph.Add((ulong)i, _baseVectors[i]);
                }
                sw.Stop();
                Console.WriteLine($"Graph built in {sw.Elapsed.TotalSeconds:N2}s.");

                _cachedGraph = graph;
                _cachedGraphParams = (M, EfConstruction);
            }

            _graph = _cachedGraph;
            // Note: USearchIndex options currently don't expose dynamically updating EfSearch,
            // but we can pass it to the search method indirectly if needed, or it might rely on the
            // initialization.
        }

        [Benchmark]
        public void Search()
        {
            for (int i = 0; i < _queryVectors!.Length; i++)
            {
                _graph!.Search(_queryVectors[i], 1, out _, out _);
            }
        }

        [IterationCleanup]
        public void Cleanup()
        {
            int correct1 = 0;
            int correct10 = 0;
            int total = _queryVectors!.Length;

            for (int i = 0; i < total; i++)
            {
                int count1 = _graph!.Search(_queryVectors[i], 1, out ulong[] keys1, out float[] _);
                if (count1 > 0 && keys1[0] == (ulong)_groundTruth![i][0])
                {
                    correct1++;
                }

                int count10 = _graph.Search(_queryVectors[i], 10, out ulong[] keys10, out float[] _);
                if (keys10.Any(k => k == (ulong)_groundTruth![i][0]))
                {
                    correct10++;
                }
            }

            double recall1 = (double)correct1 / total;
            double recall10 = (double)correct10 / total;
            Console.WriteLine($"[Configuration M={M}, EfConstruction={EfConstruction}, EfSearch={EfSearch}] Recall@1: {recall1:P2}, Recall@10: {recall10:P2}");
        }
    }
}
