using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using HNSW.Net;

namespace HNSW.Net.HybridPareto
{
    public struct Item
    {
        public int Id;
        public float[] Vector;
        public int Attribute;
    }

    public class Result
    {
        public string Method { get; set; }
        public int M { get; set; }
        public int EfConstruction { get; set; }
        public int EfSearch { get; set; }
        public int Gamma { get; set; }
        public int Mb { get; set; }
        public int Limit { get; set; }
        public double Qps { get; set; }
        public double Recall { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            double percentage = 0.25;
            if (args.Length > 0)
            {
                if (double.TryParse(args[0], out double parsedPercentage))
                {
                    percentage = parsedPercentage;
                }
                else
                {
                    Console.WriteLine($"Invalid percentage argument '{args[0]}', defaulting to {percentage}");
                }
            }

            Console.WriteLine($"Using {percentage:P} of the SIFT dataset.");

            string workingDir = Path.Combine(Path.GetTempPath(), "hnsw-bench");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);

            Dataset.DownloadAndExtractAsync(workingDir).GetAwaiter().GetResult();

            string siftDir = Path.Combine(workingDir, "sift");
            Console.WriteLine("Reading dataset...");
            var baseVectorsFull = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_base.fvecs"));
            var queryVectorsFull = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_query.fvecs"));

            int keepBaseVectors = (int)(baseVectorsFull.Length * percentage);
            int keepQueryVectors = (int)(queryVectorsFull.Length * percentage);
            // Ensure we keep at least 1 query vector to avoid zero queries
            if (keepQueryVectors == 0 && queryVectorsFull.Length > 0) keepQueryVectors = 1;

            var baseVectors = baseVectorsFull.Take(keepBaseVectors).ToArray();
            var queryVectors = queryVectorsFull.Take(keepQueryVectors).ToArray();

            var baseAttributes = Dataset.GenerateRandomAttributes(baseVectors.Length, seed: 42);
            var queryAttributes = Dataset.GenerateRandomAttributes(queryVectors.Length, seed: 43);

            var baseItems = new Item[baseVectors.Length];
            for (int i = 0; i < baseVectors.Length; i++)
            {
                baseItems[i] = new Item { Id = i, Vector = baseVectors[i], Attribute = baseAttributes[i] };
            }

            var queryItems = new Item[queryVectors.Length];
            for (int i = 0; i < queryVectors.Length; i++)
            {
                queryItems[i] = new Item { Id = i, Vector = queryVectors[i], Attribute = queryAttributes[i] };
            }

            Console.WriteLine($"Subset: {baseVectors.Length}/{baseVectorsFull.Length} base vectors");
            Console.WriteLine($"Subset: {queryVectors.Length}/{queryVectorsFull.Length} query vectors");

            int[] limits = { 10, 100 };
            int maxLimit = limits.Max();

            Console.WriteLine("Computing hybrid ground truth for subset...");
            var groundTruth = Dataset.ComputeHybridGroundTruth(baseVectors, baseAttributes, queryVectors, queryAttributes, maxLimit);

            var results = new List<Result>();

            int[] mValues = { 32 };
            int[] efConstructionValues = { 10, 20, 50, 100, 200, 400 };
            int[] efSearchValues = { 10, 20, 50, 100, 200, 400 };

            float DistanceFunc(Item a, Item b) => L2Distance.SIMD(a.Vector, b.Vector);

            foreach (var m in mValues)
            {
                foreach (var efConstruction in efConstructionValues)
                {
                    // Define configurations:
                    // (MethodName, OptimizeForFiltering, Gamma, Mb)
                    var configurations = new[]
                    {
                        ("HNSW", false, 0, 0),
                        ("ACORN-1", true, 1, m),
                        ("ACORN-Gamma12", true, 12, m),
                        ("ACORN-Gamma24", true, 24, m)
                    };

                    foreach (var (method, optimizeForFiltering, gamma, mb) in configurations)
                    {
                        var parameters = new SmallWorldParameters
                        {
                            M = m,
                            LevelLambda = 1 / Math.Log(m),
                            ConstructionPruning = efConstruction,
                            EnableDistanceCacheForConstruction = true,
                            OptimizeForFiltering = optimizeForFiltering,
                            Gamma = gamma,
                            Mb = mb
                        };

                        Console.WriteLine($"\nBuilding {method} graph with M={m}, EfConstruction={efConstruction}...");
                        var swBuild = Stopwatch.StartNew();
                        var graph = new SmallWorld<Item, float>(DistanceFunc, DefaultRandomGenerator.Instance, parameters);
                        graph.AddItems(baseItems, new ConsoleProgress());
                        swBuild.Stop();
                        Console.WriteLine($"Graph built in {swBuild.Elapsed.TotalSeconds:N2}s.");

                        foreach (var efSearch in efSearchValues)
                        {
                            graph.Parameters.EfSearch = efSearch;

                            Console.WriteLine($"Searching {method} with EfSearch={efSearch}...");

                            // Warmup
                            for (int i = 0; i < Math.Min(100, queryItems.Length); i++)
                            {
                                var q = queryItems[i];
                                graph.KNNSearch(q, 10, item => item.Attribute == q.Attribute);
                            }

                            foreach (var limit in limits)
                            {
                                int correct = 0;
                                var swSearch = Stopwatch.StartNew();
                                for (int i = 0; i < queryItems.Length; i++)
                                {
                                    var queryItem = queryItems[i];
                                    var searchResults = graph.KNNSearch(queryItem, limit, item => item.Attribute == queryItem.Attribute);

                                    var groundTruthIds = groundTruth[i].Take(limit).ToList();
                                    int matchCount = searchResults.Count(r => groundTruthIds.Contains(r.Item.Id));
                                    correct += matchCount;
                                }
                                swSearch.Stop();

                                double recall = (double)correct / (queryItems.Length * limit);
                                double qps = queryItems.Length / swSearch.Elapsed.TotalSeconds;

                                Console.WriteLine($"  -> Limit {limit} - Recall: {recall:P2}, QPS: {qps:N2}");

                                results.Add(new Result
                                {
                                    Method = method,
                                    M = m,
                                    EfConstruction = efConstruction,
                                    EfSearch = efSearch,
                                    Gamma = gamma,
                                    Mb = mb,
                                    Qps = qps,
                                    Recall = recall,
                                    Limit = limit
                                });
                            }
                        }
                    }
                }
            }

            Console.WriteLine("\n--- Pareto Sets ---");

            var groupedResults = results.GroupBy(r => new { r.Method, r.Limit });
            var allParetoFronts = new List<Result>();

            foreach (var group in groupedResults)
            {
                Console.WriteLine($"\nMethod: {group.Key.Method}, Limit: {group.Key.Limit}");
                Console.WriteLine($"{"M",-5} {"EfC",-5} {"EfS",-5} {"Gamma",-6} {"Mb",-4} {"QPS",-10} {"Recall",-10}");

                var methodResults = group.ToList();
                var paretoFront = new List<Result>();

                foreach (var r in methodResults)
                {
                    bool isDominated = false;
                    foreach (var other in methodResults)
                    {
                        if (other == r) continue;

                        bool otherIsBetterOrEqual = other.Qps >= r.Qps && other.Recall >= r.Recall;
                        bool otherIsStrictlyBetter = other.Qps > r.Qps || other.Recall > r.Recall;

                        if (otherIsBetterOrEqual && otherIsStrictlyBetter)
                        {
                            isDominated = true;
                            break;
                        }
                    }

                    if (!isDominated)
                    {
                        paretoFront.Add(r);
                    }
                }

                // Sort Pareto front by Recall
                paretoFront = paretoFront.OrderBy(r => r.Recall).ToList();
                allParetoFronts.AddRange(paretoFront);

                foreach (var p in paretoFront)
                {
                    Console.WriteLine($"{p.M,-5} {p.EfConstruction,-5} {p.EfSearch,-5} {p.Gamma,-6} {p.Mb,-4} {p.Qps,-10:F2} {p.Recall,-10:P2}");
                }
            }

            GenerateHtmlReport(allParetoFronts);
        }

        static void GenerateHtmlReport(List<Result> paretoFronts)
        {
            var html = new StringBuilder();
            html.AppendLine("<!DOCTYPE html>");
            html.AppendLine("<html>");
            html.AppendLine("<head>");
            html.AppendLine("    <title>HNSW.Net Hybrid Pareto Results</title>");
            html.AppendLine("    <script src=\"https://cdn.plot.ly/plotly-latest.min.js\"></script>");
            html.AppendLine("    <style>");
            html.AppendLine("        body { font-family: Arial, sans-serif; margin: 20px; }");
            html.AppendLine("        table { border-collapse: collapse; width: 100%; margin-top: 20px; }");
            html.AppendLine("        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }");
            html.AppendLine("        th { background-color: #f2f2f2; }");
            html.AppendLine("        .plot-container { width: 100%; height: 600px; margin-bottom: 40px; }");
            html.AppendLine("    </style>");
            html.AppendLine("</head>");
            html.AppendLine("<body>");
            html.AppendLine("    <h1>HNSW.Net Hybrid Pareto Results</h1>");

            // Generate a plot for each method comparing limits
            var methods = paretoFronts.Select(r => r.Method).Distinct().ToList();

            foreach (var method in methods)
            {
                string plotDivId = $"plot_{method.Replace("-", "_")}";
                html.AppendLine($"    <h2>Method: {method}</h2>");
                html.AppendLine($"    <div id=\"{plotDivId}\" class=\"plot-container\"></div>");
            }

            html.AppendLine("    <h2>Data Table</h2>");
            html.AppendLine("    <table>");
            html.AppendLine("        <tr><th>Method</th><th>Limit</th><th>M</th><th>EfC</th><th>EfS</th><th>Gamma</th><th>Mb</th><th>Recall</th><th>QPS</th></tr>");
            foreach (var p in paretoFronts.OrderBy(r => r.Method).ThenBy(r => r.Limit).ThenBy(r => r.Recall))
            {
                html.AppendLine($"        <tr><td>{p.Method}</td><td>{p.Limit}</td><td>{p.M}</td><td>{p.EfConstruction}</td><td>{p.EfSearch}</td><td>{p.Gamma}</td><td>{p.Mb}</td><td>{p.Recall:P2}</td><td>{p.Qps:F2}</td></tr>");
            }
            html.AppendLine("    </table>");

            html.AppendLine("    <script>");

            foreach (var method in methods)
            {
                string plotDivId = $"plot_{method.Replace("-", "_")}";
                var methodData = paretoFronts.Where(r => r.Method == method).ToList();
                var limits = methodData.Select(r => r.Limit).Distinct().OrderBy(l => l).ToList();

                var traces = new List<string>();
                var colors = new Dictionary<int, string> { { 10, "'#fc3988'" }, { 100, "'#61bd73'" } };

                foreach (var limit in limits)
                {
                    var limitData = methodData.Where(r => r.Limit == limit).OrderBy(r => r.Recall).ToList();
                    var xData = JsonSerializer.Serialize(limitData.Select(r => r.Recall));
                    var yData = JsonSerializer.Serialize(limitData.Select(r => r.Qps));
                    var textData = JsonSerializer.Serialize(limitData.Select(r => $"ef={r.EfSearch}"));
                    var color = colors.ContainsKey(limit) ? colors[limit] : "'#1f77b4'";

                    string trace = $@"
                    {{
                        x: {xData},
                        y: {yData},
                        text: {textData},
                        mode: 'lines+markers+text',
                        name: 'Limit: {limit}',
                        textposition: 'top center',
                        line: {{ color: {color}, width: 2 }},
                        marker: {{ size: 6 }}
                    }}";
                    traces.Add(trace);
                }

                html.AppendLine($@"
                    var data_{plotDivId} = [{string.Join(",", traces)}];
                    var layout_{plotDivId} = {{
                        title: '{method} - Query Performance',
                        hovermode: 'closest',
                        font: {{ family: 'Arial' }},
                        plot_bgcolor: '#ffffff',
                        paper_bgcolor: '#ffffff',
                        xaxis: {{ title: 'Recall', range: [0, 1], gridcolor: '#eee' }},
                        yaxis: {{ title: 'QPS', rangemode: 'tozero', gridcolor: '#eee' }}
                    }};
                    Plotly.newPlot('{plotDivId}', data_{plotDivId}, layout_{plotDivId});
                ");
            }

            html.AppendLine("    </script>");
            html.AppendLine("</body>");
            html.AppendLine("</html>");

            File.WriteAllText("results.html", html.ToString());
            Console.WriteLine("\nGenerated results.html");
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
}
