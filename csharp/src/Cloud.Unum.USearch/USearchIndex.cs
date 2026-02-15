using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Cloud.Unum.USearch;

public class USearchIndex<T> : IDisposable where T : INumberBase<T>, IRootFunctions<T>, IComparable<T>
{
    private readonly HnswGraph<T> _graph;
    private bool _disposedValue;

    public USearchIndex(IndexOptions options)
    {
        var metric = Metrics.GetDistanceFunction<T>(options.MetricKind);
        _graph = new HnswGraph<T>(options, metric);
    }

    public USearchIndex(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new BinaryReader(stream);

        // 1. Vectors
        uint rows = reader.ReadUInt32();
        uint cols = reader.ReadUInt32();

        int sizeOfT = Unsafe.SizeOf<T>();
        int dim = (int)(cols / sizeOfT);
        var vectors = new List<T[]>((int)rows);

        for (int i = 0; i < rows; i++)
        {
            byte[] bytes = reader.ReadBytes((int)cols);
            T[] vec = new T[dim];
            Buffer.BlockCopy(bytes, 0, vec, 0, bytes.Length);
            vectors.Add(vec);
        }

        // 2. Metadata
        char[] magic = reader.ReadChars(7); // "usearch"
        if (new string(magic) != "usearch")
            throw new USearchException("Invalid file format");

        ushort vMajor = reader.ReadUInt16();
        ushort vMinor = reader.ReadUInt16();
        ushort vPatch = reader.ReadUInt16();

        byte metricKindByte = reader.ReadByte();
        MetricKind metricKind = MetricKind.Unknown;
        switch ((char)metricKindByte)
        {
            case 'i': metricKind = MetricKind.Ip; break;
            case 'c': metricKind = MetricKind.Cos; break;
            case 'e': metricKind = MetricKind.L2sq; break;
            case 'h': metricKind = MetricKind.Haversine; break;
        }

        byte scalarKindByte = reader.ReadByte();
        ScalarKind scalarKind = ScalarKind.Unknown;
        switch (scalarKindByte)
        {
            case 11: scalarKind = ScalarKind.Float32; break;
            case 10: scalarKind = ScalarKind.Float64; break;
            case 12: scalarKind = ScalarKind.Float16; break;
            case 23: scalarKind = ScalarKind.Int8; break;
            case 1: scalarKind = ScalarKind.Bits1; break;
        }

        reader.ReadByte(); // kind_key
        reader.ReadByte(); // kind_slot

        ulong countPresent = reader.ReadUInt64();
        ulong countDeleted = reader.ReadUInt64();
        ulong dimensions = reader.ReadUInt64();
        bool multi = reader.ReadBoolean();

        reader.ReadBytes(22); // Padding

        var options = new IndexOptions(
            metricKind: metricKind,
            quantization: scalarKind,
            dimensions: dimensions,
            connectivity: 0, // Will be read from graph header
            expansionAdd: 0,
            expansionSearch: 0,
            multi: multi
        );

        var metric = Metrics.GetDistanceFunction<T>(options.MetricKind);
        _graph = new HnswGraph<T>(options, metric);

        // 3. Graph
        _graph.Load(reader, vectors);
    }

    public void Add(ulong key, T[] vector)
    {
        _graph.Add(key, vector);
    }

    public int Search(T[] query, int count, out ulong[] keys, out T[] distances)
    {
        var (ids, dists) = _graph.Search(query, count);
        keys = new ulong[ids.Length];
        distances = dists;

        for (int i = 0; i < ids.Length; i++)
        {
            keys[i] = _graph.GetKey(ids[i]);
        }

        return ids.Length;
    }

    public void Save(string path)
    {
        using var stream = File.OpenWrite(path);
        using var writer = new BinaryWriter(stream);

        // 1. Vectors (32-bit dimensions for compatibility)
        // matrix_rows (size), matrix_cols (bytes per vector)
        uint rows = (uint)_graph.Count;
        int sizeOfT = Unsafe.SizeOf<T>();
        uint cols = (uint)(_graph.Options.Dimensions * (ulong)sizeOfT);

        writer.Write(rows);
        writer.Write(cols);

        for (int i = 0; i < _graph.Count; i++)
        {
            var vec = _graph.GetVector(i);
            byte[] byteBuffer = new byte[vec.Length * sizeOfT];
            Buffer.BlockCopy(vec, 0, byteBuffer, 0, byteBuffer.Length);
            writer.Write(byteBuffer);
        }

        // 2. Metadata (Dense Header)
        WriteDenseHeader(writer);

        // 3. Graph
        _graph.Save(writer);
    }

    private void WriteDenseHeader(BinaryWriter writer)
    {
        writer.Write("usearch".ToCharArray());

        // Version 2.10.0
        writer.Write((ushort)2);
        writer.Write((ushort)10);
        writer.Write((ushort)0);

        byte metricKind = 0;
        switch (_graph.Options.MetricKind)
        {
            case MetricKind.Ip: metricKind = (byte)'i'; break;
            case MetricKind.Cos: metricKind = (byte)'c'; break;
            case MetricKind.L2sq: metricKind = (byte)'e'; break;
            case MetricKind.Haversine: metricKind = (byte)'h'; break;
            default: metricKind = 0; break;
        }
        writer.Write(metricKind);

        byte scalarKind = 0;
        switch (_graph.Options.Quantization)
        {
            case ScalarKind.Float32: scalarKind = 11; break;
            case ScalarKind.Float64: scalarKind = 10; break;
            case ScalarKind.Float16: scalarKind = 12; break;
            case ScalarKind.Int8: scalarKind = 23; break;
            case ScalarKind.Bits1: scalarKind = 1; break;
            default: scalarKind = 0; break;
        }
        writer.Write(scalarKind);

        writer.Write((byte)14); // uint64 keys
        writer.Write((byte)15); // uint32 slots

        writer.Write((ulong)_graph.Count);
        writer.Write((ulong)0); // count_deleted
        writer.Write((ulong)_graph.Options.Dimensions);

        writer.Write(_graph.Options.Multi);

        writer.Write(new byte[22]); // Padding
    }

    public int Size() => _graph.Count;
    public ulong Dimensions() => _graph.Options.Dimensions;
    public ulong Connectivity() => _graph.Options.Connectivity;

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposedValue)
        {
            if (disposing)
            {
                // TODO: dispose managed state (graph doesn't need disposal unless we add native memory)
            }
            _disposedValue = true;
        }
    }

    public void Dispose()
    {
        Dispose(disposing: true);
        GC.SuppressFinalize(this);
    }
}
