using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace HNSW.Net.HybridBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = DefaultConfig.Instance
                .AddColumn(new RecallColumn("Recall@1", "Recall@1"))
                .AddColumn(new RecallColumn("Recall@10", "Recall@10"));
            BenchmarkRunner.Run<HybridBenchmark>(config, args);
        }
    }
}
