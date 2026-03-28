using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Net;
using System.Threading.Tasks;

namespace HNSW.Net.HybridPareto
{
    public static class Dataset
    {
        public static int[] GenerateRandomAttributes(int count, int seed = 42)
        {
            var random = new Random(seed);
            var attributes = new int[count];
            for (int i = 0; i < count; i++)
            {
                attributes[i] = random.Next(1, 13); // Range 1-12 inclusive
            }
            return attributes;
        }

        public static int[][] ComputeHybridGroundTruth(float[][] baseVectors, int[] baseAttributes, float[][] queryVectors, int[] queryAttributes, int k)
        {
            Console.WriteLine("Computing hybrid ground truth...");
            var groundTruth = new int[queryVectors.Length][];

            // To make this much faster, let's use Parallel.For
            Parallel.For(0, queryVectors.Length, i =>
            {
                var queryVector = queryVectors[i];
                var queryAttribute = queryAttributes[i];
                var distances = new List<(int Id, float Distance)>();

                for (int j = 0; j < baseVectors.Length; j++)
                {
                    if (baseAttributes[j] == queryAttribute)
                    {
                        float dist = L2Distance.SIMD(queryVector, baseVectors[j]);
                        distances.Add((j, dist));
                    }
                }

                distances.Sort((a, b) => a.Distance.CompareTo(b.Distance));
                groundTruth[i] = new int[Math.Min(k, distances.Count)];
                for (int j = 0; j < groundTruth[i].Length; j++)
                {
                    groundTruth[i][j] = distances[j].Id;
                }

                if (i > 0 && i % 1000 == 0)
                {
                    Console.WriteLine($"Computed ground truth for {i} queries");
                }
            });

            Console.WriteLine("Finished computing hybrid ground truth.");
            return groundTruth;
        }
        public static float[][] ReadFvecs(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            var results = new List<float[]>();
            while (stream.Position < stream.Length)
            {
                int dim = reader.ReadInt32();
                var vector = new float[dim];
                for (int i = 0; i < dim; i++)
                {
                    vector[i] = reader.ReadSingle();
                }
                results.Add(vector);
            }
            return results.ToArray();
        }

        public static int[][] ReadIvecs(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            var results = new List<int[]>();
            while (stream.Position < stream.Length)
            {
                int dim = reader.ReadInt32();
                var vector = new int[dim];
                for (int i = 0; i < dim; i++)
                {
                    vector[i] = reader.ReadInt32();
                }
                results.Add(vector);
            }
            return results.ToArray();
        }

        public static void SaveGroundTruth(int[][] groundTruth, string path)
        {
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);
            writer.Write(groundTruth.Length);
            for (int i = 0; i < groundTruth.Length; i++)
            {
                var row = groundTruth[i];
                writer.Write(row.Length);
                for (int j = 0; j < row.Length; j++)
                {
                    writer.Write(row[j]);
                }
            }
        }

        public static int[][] LoadGroundTruth(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            int length = reader.ReadInt32();
            var groundTruth = new int[length][];
            for (int i = 0; i < length; i++)
            {
                int rowLength = reader.ReadInt32();
                var row = new int[rowLength];
                for (int j = 0; j < rowLength; j++)
                {
                    row[j] = reader.ReadInt32();
                }
                groundTruth[i] = row;
            }
            return groundTruth;
        }

        public static async Task DownloadAndExtractAsync(string workingDir)
        {
            string tarPath = Path.Combine(workingDir, "sift.tar.gz");
            if (!File.Exists(tarPath))
            {
                Console.WriteLine("Downloading Sift1M dataset...");
                using var client = new FluentFTP.AsyncFtpClient("ftp.irisa.fr");
                await client.Connect();
                await client.DownloadFile(tarPath, "/local/texmex/corpus/sift.tar.gz");
                await client.Disconnect();
            }

            // Check if extracted files already exist to avoid re-extracting
            string baseFile = Path.Combine(workingDir, "sift", "sift_base.fvecs");
            if (!File.Exists(baseFile))
            {
                Console.WriteLine("Extracting Sift1M dataset...");
                using var fsIn = File.OpenRead(tarPath);
                using var gzipStream = new GZipStream(fsIn, CompressionMode.Decompress);
                TarFile.ExtractToDirectory(gzipStream, workingDir, true);
            }
        }
    }
}
