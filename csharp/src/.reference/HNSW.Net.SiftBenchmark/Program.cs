using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace HNSW.Net.SiftBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = DefaultConfig.Instance;
            BenchmarkRunner.Run<SiftBenchmark>(config, args);
        }
    }
}
