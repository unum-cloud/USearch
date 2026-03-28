using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using HNSW.Net;

namespace HNSW.Net.HybridBenchmark
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
        private static float[][] _baseVectors;
        private static float[][] _queryVectors;

        private static Item[] _baseItems;
        private static Item[] _queryItems;

        private static int[][] _groundTruth;

        private static SmallWorld<Item, float> _cachedGraph;
        private static (int M, int EfConstruction, bool OptimizeForFiltering, int Gamma, int Mb) _cachedGraphParams;

        private SmallWorld<Item, float> _graph;

        [Params(16)]
        public int M { get; set; }

        [Params(200)]
        public int EfConstruction { get; set; }

        [Params(50, 100, 200)]
        public int EfSearch { get; set; }

        // By default false, which means it will do Post-Filtering
        [Params(false, true)]
        public bool OptimizeForFiltering { get; set; }

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

            if (_cachedGraph == null || _cachedGraphParams.M != M || _cachedGraphParams.EfConstruction != EfConstruction ||
                _cachedGraphParams.OptimizeForFiltering != OptimizeForFiltering ||
                _cachedGraphParams.Gamma != Gamma || _cachedGraphParams.Mb != Mb)
            {
                string graphCachePath = Path.Combine(workingDir, $"graph_{M}_{EfConstruction}_{OptimizeForFiltering}_{Gamma}_{Mb}.bin");
                float DistanceFunc(Item a, Item b) => L2Distance.SIMD(a.Vector, b.Vector);

                if (File.Exists(graphCachePath))
                {
                    Console.WriteLine($"Loading graph from {graphCachePath}...");
                    using var stream = File.OpenRead(graphCachePath);
                    var (graph, _) = SmallWorld<Item, float>.DeserializeGraph(_baseItems, DistanceFunc, DefaultRandomGenerator.Instance, stream);
                    _cachedGraph = graph;
                }
                else
                {
                    var parameters = new SmallWorldParameters
                    {
                        M = M,
                        LevelLambda = 1 / Math.Log(M),
                        ConstructionPruning = EfConstruction,
                        EfSearch = EfSearch,
                        EnableDistanceCacheForConstruction = true,
                        OptimizeForFiltering = OptimizeForFiltering,
                        Gamma = Gamma,
                        Mb = Mb
                    };

                    Console.WriteLine($"Building graph with M={M}, EfConstruction={EfConstruction}, OptimizeForFiltering={OptimizeForFiltering}, Gamma={Gamma}, Mb={Mb}...");
                    var sw = System.Diagnostics.Stopwatch.StartNew();

                    var graph = new SmallWorld<Item, float>(DistanceFunc, DefaultRandomGenerator.Instance, parameters);
                    graph.AddItems(_baseItems, new ConsoleProgress());
                    sw.Stop();
                    Console.WriteLine($"Graph built in {sw.Elapsed.TotalSeconds:N2}s.");

                    Console.WriteLine($"Saving graph to {graphCachePath}...");
                    using var stream = File.Create(graphCachePath);
                    graph.SerializeGraph(stream);

                    _cachedGraph = graph;
                }

                _cachedGraphParams = (M, EfConstruction, OptimizeForFiltering, Gamma, Mb);
            }

            _graph = _cachedGraph;
            _graph.Parameters.EfSearch = EfSearch;
        }

        [Benchmark]
        public void Search()
        {
            var c = ItemsToSearch;
            while(c >= 0)
            {
                int i = c % _queryItems.Length;
                var queryItem = _queryItems[i];
                // Post-filter matching the attribute. With OptimizeForFiltering=true it will do predicate subgraph traversal.
                _graph.KNNSearch(queryItem, 10, item => item.Attribute == queryItem.Attribute);
                c--;
            }
        }

        [IterationCleanup]
        public void Cleanup()
        {
            int correct1 = 0;
            int correct10 = 0;
            int total = _queryItems.Length;

            for (int i = 0; i < total; i++)
            {
                var queryItem = _queryItems[i];
                var results1 = _graph.KNNSearch(queryItem, 1, item => item.Attribute == queryItem.Attribute);
                if (results1.Count > 0 && results1[0].Item.Id == _groundTruth[i][0])
                {
                    correct1++;
                }

                var results10 = _graph.KNNSearch(queryItem, 10, item => item.Attribute == queryItem.Attribute);
                if (results10.Any(r => r.Item.Id == _groundTruth[i][0]))
                {
                    correct10++;
                }
            }

            double recall1 = (double)correct1 / total;
            double recall10 = (double)correct10 / total;
            Console.WriteLine($"[Configuration M={M}, EfConstruction={EfConstruction}, EfSearch={EfSearch}, OptimizeForFiltering={OptimizeForFiltering}] Recall@1: {recall1:P2}, Recall@10: {recall10:P2}");

            string workingDir = Path.Combine(Path.GetTempPath(), "hnsw-bench");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);
            File.WriteAllText(Path.Combine(workingDir, $"Recall_{M}_{EfConstruction}_{EfSearch}_{OptimizeForFiltering}_Recall@1.txt"), recall1.ToString());
            File.WriteAllText(Path.Combine(workingDir, $"Recall_{M}_{EfConstruction}_{EfSearch}_{OptimizeForFiltering}_Recall@10.txt"), recall10.ToString());
        }
    }

    internal class ConsoleProgress : IProgressReporter
    {
        private Stopwatch _sw;
        public void Progress(int current, int total)
        {
            if (_sw is null) _sw = Stopwatch.StartNew();
            if (_sw.Elapsed.TotalSeconds > 5)
            {
                Console.WriteLine($"At {current:n0} of {total:n0} or {100f * current / total}%");
                _sw.Restart();
            }
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
            var optimizeForFiltering = benchmarkCase.Parameters["OptimizeForFiltering"]?.ToString();

            string fileName = $"Recall_{m}_{efConstruction}_{efSearch}_{optimizeForFiltering}_{_recallKey}.txt";
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