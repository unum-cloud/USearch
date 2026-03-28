using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using Cloud.Unum.USearch;

namespace USearch.HybridBenchmark
{
    public struct Item
    {
        public int Id;
        public float[] Vector;
        public int Attribute;
    }

    [MinIterationCount(1)]
    [MinWarmupCount(1)]
    [MaxWarmupCount(2)]
    [MaxIterationCount(2)]
    public class HybridBenchmark
    {
        private static float[][]? _baseVectors;
        private static float[][]? _queryVectors;

        private static Item[]? _baseItems;
        private static Item[]? _queryItems;

        private static int[][]? _groundTruth;

        private static USearchIndex<float>? _cachedGraph;
        private static (int M, int EfConstruction) _cachedGraphParams;
        private static Dictionary<ulong, int>? _cachedMetadata;

        private USearchIndex<float>? _graph;
        private Dictionary<ulong, int>? _metadata;

        [Params(16)]
        public int M { get; set; }

        [Params(200)]
        public int EfConstruction { get; set; }

        [Params(50, 100, 200)]
        public int EfSearch { get; set; }

        [Params(1000, 10_000)]
        public int ItemsToSearch { get; set; }

        public int Gamma { get; set; } = 12; // Typical value for SIFT1M per paper
        public int Mb { get; set; } = 16; // Small multiple of M, typically M, 2M, or 64. Using M.

        [GlobalSetup]
        public void Setup()
        {
            string workingDir = Path.Combine(Path.GetTempPath(), "hnsw-bench");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);

            Dataset.DownloadAndExtractAsync(workingDir).GetAwaiter().GetResult();

            string siftDir = Path.Combine(workingDir, "sift");
            if (_baseVectors == null)
            {
                Console.WriteLine("Reading dataset...");
                _baseVectors = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_base.fvecs"));
                _queryVectors = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_query.fvecs"));

                int keepBaseVectors = _baseVectors.Length;
                int keepQueryVectors = _queryVectors.Length;

                var baseAttributes = Dataset.GenerateRandomAttributes(_baseVectors.Length, seed: 42);
                var queryAttributes = Dataset.GenerateRandomAttributes(_queryVectors.Length, seed: 43);

                var baseItemsLen = Math.Min(keepBaseVectors, _baseVectors.Length);
                _baseItems = new Item[baseItemsLen];
                for (int i = 0; i < baseItemsLen; i++)
                {
                    _baseItems[i] = new Item { Id = i, Vector = _baseVectors[i], Attribute = baseAttributes[i] };
                }

                var queryItemsLen = Math.Min(keepQueryVectors, _queryVectors.Length);
                _queryItems = new Item[queryItemsLen];
                for (int i = 0; i < queryItemsLen; i++)
                {
                    _queryItems[i] = new Item { Id = i, Vector = _queryVectors[i], Attribute = queryAttributes[i] };
                }

                string groundTruthPath = Path.Combine(workingDir, "groundTruth.bin");
                if (File.Exists(groundTruthPath))
                {
                    Console.WriteLine("Loading hybrid ground truth...");
                    _groundTruth = Dataset.LoadGroundTruth(groundTruthPath);
                }
                else
                {
                    _groundTruth = Dataset.ComputeHybridGroundTruth(_baseVectors, baseAttributes, _queryVectors, queryAttributes, 10);
                    Dataset.SaveGroundTruth(_groundTruth, groundTruthPath);
                }

                Console.WriteLine($"Loaded {baseItemsLen}/{_baseVectors.Length} base vectors");
                Console.WriteLine($"Loaded {queryItemsLen}/{_queryVectors.Length} query vectors");
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
                var metadata = new Dictionary<ulong, int>();

                for (int i = 0; i < _baseItems!.Length; i++)
                {
                    ulong key = (ulong)_baseItems[i].Id;
                    graph.Add(key, _baseItems[i].Vector);
                    metadata[key] = _baseItems[i].Attribute;
                }
                sw.Stop();
                Console.WriteLine($"Graph built in {sw.Elapsed.TotalSeconds:N2}s.");

                _cachedGraph = graph;
                _cachedMetadata = metadata;
                _cachedGraphParams = (M, EfConstruction);
            }

            _graph = _cachedGraph;
            _metadata = _cachedMetadata;
            // USearch doesn't directly support setting EfSearch dynamically like HNSW.Net's _graph.Parameters.EfSearch = EfSearch
        }

        [Benchmark]
        public void Search()
        {
            var c = ItemsToSearch;
            while(c >= 0)
            {
                int i = c % _queryItems!.Length;
                var queryItem = _queryItems[i];

                // USearch doesn't have a predicate argument in Search yet.
                // We emulate post-filtering by over-fetching and filtering manually.
                // For a proper benchmark of "exact matches with filtering", we fetch more than K.
                int overfetch = 100; // fetch more and post-filter
                int fetched = _graph!.Search(queryItem.Vector, overfetch, out ulong[] keys, out float[] _);

                var results = new List<ulong>();
                for (int j = 0; j < fetched; j++)
                {
                    if (_metadata![keys[j]] == queryItem.Attribute)
                    {
                        results.Add(keys[j]);
                        if (results.Count >= 10) break;
                    }
                }
                c--;
            }
        }

        [IterationCleanup]
        public void Cleanup()
        {
            int correct1 = 0;
            int correct10 = 0;
            int total = _queryItems!.Length;

            for (int i = 0; i < total; i++)
            {
                var queryItem = _queryItems[i];
                int overfetch = 1000;
                int fetched = _graph!.Search(queryItem.Vector, overfetch, out ulong[] keys, out float[] _);

                var results = new List<ulong>();
                for (int j = 0; j < fetched; j++)
                {
                    if (_metadata![keys[j]] == queryItem.Attribute)
                    {
                        results.Add(keys[j]);
                        if (results.Count >= 10) break;
                    }
                }

                if (results.Count > 0 && results[0] == (ulong)_groundTruth![i][0])
                {
                    correct1++;
                }

                if (results.Any(r => r == (ulong)_groundTruth![i][0]))
                {
                    correct10++;
                }
            }

            double recall1 = (double)correct1 / total;
            double recall10 = (double)correct10 / total;
            Console.WriteLine($"[Configuration M={M}, EfConstruction={EfConstruction}, EfSearch={EfSearch}] Recall@1: {recall1:P2}, Recall@10: {recall10:P2}");

            string workingDir = Path.Combine(Path.GetTempPath(), "hnsw-bench");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);
            File.WriteAllText(Path.Combine(workingDir, $"Recall_{M}_{EfConstruction}_{EfSearch}_Recall@1.txt"), recall1.ToString());
            File.WriteAllText(Path.Combine(workingDir, $"Recall_{M}_{EfConstruction}_{EfSearch}_Recall@10.txt"), recall10.ToString());
        }
    }

    public class RecallColumn : BenchmarkDotNet.Columns.IColumn
    {
        private readonly string _columnName;
        private readonly string _recallKey;

        public RecallColumn(string columnName, string recallKey)
        {
            _columnName = columnName;
            _recallKey = recallKey;
        }

        public string Id => nameof(RecallColumn) + "." + _columnName;
        public string ColumnName => _columnName;
        public bool AlwaysShow => true;
        public BenchmarkDotNet.Columns.ColumnCategory Category => BenchmarkDotNet.Columns.ColumnCategory.Metric;
        public int PriorityInCategory => 0;
        public bool IsNumeric => true;
        public BenchmarkDotNet.Columns.UnitType UnitType => BenchmarkDotNet.Columns.UnitType.Dimensionless;
        public string Legend => _columnName;

        public string GetValue(BenchmarkDotNet.Reports.Summary summary, BenchmarkDotNet.Running.BenchmarkCase benchmarkCase)
        {
            var m = benchmarkCase.Parameters["M"]?.ToString();
            var efConstruction = benchmarkCase.Parameters["EfConstruction"]?.ToString();
            var efSearch = benchmarkCase.Parameters["EfSearch"]?.ToString();

            string fileName = $"Recall_{m}_{efConstruction}_{efSearch}_{_recallKey}.txt";
            string workingDir = Path.Combine(Path.GetTempPath(), "hnsw-bench");
            string path = Path.Combine(workingDir, fileName);

            if (File.Exists(path))
            {
                if (double.TryParse(File.ReadAllText(path), out double val))
                {
                    return val.ToString("P2");
                }
            }
            return "N/A";
        }

        public string GetValue(BenchmarkDotNet.Reports.Summary summary, BenchmarkDotNet.Running.BenchmarkCase benchmarkCase, BenchmarkDotNet.Reports.SummaryStyle style) => GetValue(summary, benchmarkCase);
        public bool IsDefault(BenchmarkDotNet.Reports.Summary summary, BenchmarkDotNet.Running.BenchmarkCase benchmarkCase) => false;
        public bool IsAvailable(BenchmarkDotNet.Reports.Summary summary) => true;
    }
}