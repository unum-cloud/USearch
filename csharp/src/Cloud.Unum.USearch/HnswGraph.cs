using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace Cloud.Unum.USearch;

internal class HnswGraph<T> where T : INumberBase<T>, IRootFunctions<T>, IComparable<T>
{
    private IndexOptions _options;
    private readonly Metrics.DistanceFunction<T> _metric;
    private readonly List<Node> _nodes = new();
    private int _entryPoint = -1;
    private int _maxLevel = -1;
    private readonly double _levelMultiplier;
    private readonly int _connectivityBase;
    private readonly object _globalLock = new object();

    public HnswGraph(IndexOptions options, Metrics.DistanceFunction<T> metric)
    {
        _options = options;
        if (_options.Connectivity == 0) _options = _options with { Connectivity = 16 };
        if (_options.ExpansionAdd == 0) _options = _options with { ExpansionAdd = 128 };
        if (_options.ExpansionSearch == 0) _options = _options with { ExpansionSearch = 64 };

        _metric = metric;
        _connectivityBase = (int)_options.Connectivity * 2;
        _levelMultiplier = 1.0 / Math.Log((double)_options.Connectivity);
    }

    public int Count => _nodes.Count;
    public IndexOptions Options => _options;

    public ulong GetKey(int id) => _nodes[id].Key;
    public T[] GetVector(int id) => _nodes[id].Vector;

    public void Save(BinaryWriter writer)
    {
        // 1. Header (64 bytes)
        writer.Write((ulong)_nodes.Count);
        writer.Write((ulong)_options.Connectivity);
        writer.Write((ulong)_connectivityBase);
        writer.Write((ulong)_maxLevel);
        writer.Write((ulong)(_entryPoint == -1 ? 0 : _entryPoint));

        // Padding (64 - 5*8 = 24 bytes)
        writer.Write(new byte[24]);

        // 2. Levels
        for (int i = 0; i < _nodes.Count; i++)
        {
            writer.Write((short)_nodes[i].Level);
        }

        // 3. Nodes
        for (int i = 0; i < _nodes.Count; i++)
        {
            var node = _nodes[i];

            // Node head: Key (8 bytes) + Level (2 bytes)
            writer.Write(node.Key);
            writer.Write((short)node.Level);

            // Neighbors
            for (int l = 0; l <= node.Level; l++)
            {
                var neighbors = node.Neighbors[l];
                // In C++, neighbor list starts with count (4 bytes).
                // Assuming compressed_slot_t is 4 bytes.
                // neighbors_bytes = count (4) + slots (count * 4).

                writer.Write((uint)neighbors.Count);
                foreach (var n in neighbors)
                {
                    writer.Write((uint)n);
                }
            }
        }
    }

    public void Load(BinaryReader reader, List<T[]> vectors)
    {
        ulong size = reader.ReadUInt64();
        ulong connectivity = reader.ReadUInt64();
        ulong connectivityBase = reader.ReadUInt64();
        ulong maxLevel = reader.ReadUInt64();
        ulong entrySlot = reader.ReadUInt64();

        reader.ReadBytes(24); // Padding

        short[] levels = new short[vectors.Count];
        for (int i = 0; i < vectors.Count; i++)
        {
            levels[i] = reader.ReadInt16();
        }

        _nodes.Clear();
        _nodes.Capacity = vectors.Count;

        for (int i = 0; i < vectors.Count; i++)
        {
            ulong key = reader.ReadUInt64();
            short level = reader.ReadInt16();

            // Construct node
            // Note: connectivity might differ from _options if file differs?
            // We should use file's connectivity?
            // HnswGraph was initialized with _options.
            // Ideally _options should match file.

            Node node = new Node(key, vectors[i], level, (int)connectivity, (int)connectivityBase);

            for (int l = 0; l <= level; l++)
            {
                uint count = reader.ReadUInt32();
                var neighbors = node.Neighbors[l];
                for (uint j = 0; j < count; j++)
                {
                    neighbors.Add((int)reader.ReadUInt32());
                }
            }
            _nodes.Add(node);
        }

        _maxLevel = (int)maxLevel;
        _entryPoint = (int)entrySlot;
    }

    public (int[] ids, T[] distances) Search(T[] query, int k)
    {
        if (_entryPoint == -1)
        {
            return (Array.Empty<int>(), Array.Empty<T>());
        }

        int currObj = _entryPoint;
        for (int l = _maxLevel; l > 0; l--)
        {
            currObj = SearchLayer(query, currObj, l);
        }

        int ef = Math.Max((int)_options.ExpansionSearch, k);
        var candidatesPQ = SearchLevel(query, currObj, 0, ef);

        var results = new List<(int id, T dist)>();
        while (candidatesPQ.TryDequeue(out int id, out T d))
        {
            results.Add((id, d));
        }

        results.Reverse(); // Now best to worst

        if (results.Count > k)
        {
            results = results.GetRange(0, k);
        }

        int[] ids = new int[results.Count];
        T[] dists = new T[results.Count];
        for (int i = 0; i < results.Count; i++)
        {
            ids[i] = results[i].id;
            dists[i] = results[i].dist;
        }
        return (ids, dists);
    }

    internal class Node
    {
        public ulong Key;
        public T[] Vector;
        public int Level;
        public List<int>[] Neighbors;
        public readonly object Lock = new object();

        public Node(ulong key, T[] vector, int level, int connectivity, int connectivityBase)
        {
            Key = key;
            Vector = vector;
            Level = level;
            Neighbors = new List<int>[level + 1];
            // Base layer 0 has connectivityBase (usually 2*M)
            Neighbors[0] = new List<int>(connectivityBase);
            for (int i = 1; i <= level; i++)
            {
                Neighbors[i] = new List<int>(connectivity);
            }
        }
    }

    public void Add(ulong key, T[] vector)
    {
        int level = GetRandomLevel();
        int connectivity = (int)_options.Connectivity;
        int connectivityBase = connectivity * 2;

        Node newNode = new Node(key, vector, level, connectivity, connectivityBase);
        int newNodeId;

        lock (_globalLock)
        {
            _nodes.Add(newNode);
            newNodeId = _nodes.Count - 1;

            if (_entryPoint == -1)
            {
                _entryPoint = newNodeId;
                _maxLevel = level;
                return;
            }
        }

        int currObj = _entryPoint;
        T dist = _metric(vector, _nodes[currObj].Vector);

        for (int l = _maxLevel; l > level; l--)
        {
            currObj = SearchLayer(vector, currObj, l);
        }

        for (int l = Math.Min(level, _maxLevel); l >= 0; l--)
        {
            int ef = (int)_options.ExpansionAdd;
            var candidatesPQ = SearchLevel(vector, currObj, l, ef);

            var candidates = new List<(int id, T dist)>();
            while (candidatesPQ.TryDequeue(out int id, out T d))
            {
                candidates.Add((id, d));
            }
            // PQ is MaxHeap, so dequeued worst first. Reverse to get best first.
            candidates.Reverse();

            // Best candidate is the starting point for next layer
            currObj = candidates[0].id;

            int m = l == 0 ? connectivityBase : connectivity;
            var selected = SelectNeighbors(vector, candidates, m);

            foreach (var neighborId in selected)
            {
                _nodes[newNodeId].Neighbors[l].Add(neighborId);
                Connect(newNodeId, neighborId, l);
            }
        }

        if (level > _maxLevel)
        {
            lock (_globalLock)
            {
                if (level > _maxLevel)
                {
                    _maxLevel = level;
                    _entryPoint = newNodeId;
                }
            }
        }
    }

    private void Connect(int newNodeId, int neighborId, int level)
    {
        var neighbor = _nodes[neighborId];
        lock (neighbor.Lock)
        {
            var neighborNeighbors = neighbor.Neighbors[level];
            neighborNeighbors.Add(newNodeId);

            int mMax = level == 0 ? (int)_options.Connectivity * 2 : (int)_options.Connectivity;

            if (neighborNeighbors.Count > mMax)
            {
                var candidates = new List<(int id, T dist)>();
                var neighborVector = neighbor.Vector;

                foreach (var nId in neighborNeighbors)
                {
                    T dist = _metric(neighborVector, _nodes[nId].Vector);
                    candidates.Add((nId, dist));
                }
                candidates.Sort((a, b) => a.dist.CompareTo(b.dist));

                var selected = SelectNeighbors(neighborVector, candidates, mMax);
                neighbor.Neighbors[level] = selected;
            }
        }
    }

    private List<int> SelectNeighbors(T[] query, List<(int id, T dist)> candidates, int m)
    {
        var selected = new List<int>();
        foreach (var (candId, candDist) in candidates)
        {
            if (selected.Count >= m) break;

            bool good = true;
            foreach (var selId in selected)
            {
                T distToSel = _metric(_nodes[candId].Vector, _nodes[selId].Vector);
                if (distToSel.CompareTo(candDist) < 0)
                {
                    good = false;
                    break;
                }
            }

            if (good)
            {
                selected.Add(candId);
            }
        }
        return selected;
    }

    private int GetRandomLevel()
    {
        double r = -Math.Log(Random.Shared.NextDouble()) * _levelMultiplier;
        return (int)r;
    }

    private class MaxHeapComparer : IComparer<T>
    {
        public int Compare(T? x, T? y)
        {
            if (x == null) return y == null ? 0 : -1;
            if (y == null) return 1;
            return y.CompareTo(x);
        }
    }

    private int SearchLayer(T[] query, int entryPoint, int level)
    {
        int bestNode = entryPoint;
        T bestDistance = _metric(query, _nodes[bestNode].Vector);

        bool changed = true;
        while (changed)
        {
            changed = false;
            int candidateNode = bestNode;
            T candidateDistance = bestDistance;

            var node = _nodes[bestNode];
            int[] neighborIds = ArrayPool<int>.Shared.Rent(node.Neighbors[level].Capacity);
            int count = 0;

            lock (node.Lock)
            {
                var list = node.Neighbors[level];
                count = list.Count;
                if (neighborIds.Length < count)
                {
                    ArrayPool<int>.Shared.Return(neighborIds);
                    neighborIds = ArrayPool<int>.Shared.Rent(count);
                }
                list.CopyTo(neighborIds);
            }

            try
            {
                for (int i = 0; i < count; i++)
                {
                    int neighborId = neighborIds[i];
                    T distance = _metric(query, _nodes[neighborId].Vector);
                    if (distance.CompareTo(candidateDistance) < 0)
                    {
                        candidateDistance = distance;
                        candidateNode = neighborId;
                    }
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(neighborIds);
            }

            if (candidateNode != bestNode)
            {
                bestDistance = candidateDistance;
                bestNode = candidateNode;
                changed = true;
            }
        }
        return bestNode;
    }

    private PriorityQueue<int, T> SearchLevel(T[] query, int entryPoint, int level, int ef)
    {
        var visited = new HashSet<int>();
        var candidates = new PriorityQueue<int, T>(); // min-heap
        var results = new PriorityQueue<int, T>(new MaxHeapComparer()); // max-heap

        T dist = _metric(query, _nodes[entryPoint].Vector);

        candidates.Enqueue(entryPoint, dist);
        results.Enqueue(entryPoint, dist);
        visited.Add(entryPoint);

        T lowerBound = dist; // Max distance in results

        while (candidates.Count > 0)
        {
            candidates.TryPeek(out int candidateNode, out T candidateDist);
            if (candidateDist.CompareTo(lowerBound) > 0)
            {
                if (results.Count == ef)
                    break;
            }

            candidates.Dequeue();

            var node = _nodes[candidateNode];
            int[] neighborIds = ArrayPool<int>.Shared.Rent(node.Neighbors[level].Capacity);
            int count = 0;

            lock (node.Lock)
            {
                var list = node.Neighbors[level];
                count = list.Count;
                if (neighborIds.Length < count)
                {
                    ArrayPool<int>.Shared.Return(neighborIds);
                    neighborIds = ArrayPool<int>.Shared.Rent(count);
                }
                list.CopyTo(neighborIds);
            }

            try
            {
                for (int i = 0; i < count; i++)
                {
                    int neighbor = neighborIds[i];
                    if (!visited.Contains(neighbor))
                    {
                        visited.Add(neighbor);
                        T d = _metric(query, _nodes[neighbor].Vector);

                        if (results.Count < ef || d.CompareTo(lowerBound) < 0)
                        {
                            candidates.Enqueue(neighbor, d);
                            results.Enqueue(neighbor, d);
                            if (results.Count > ef)
                            {
                                results.Dequeue();
                            }
                            results.TryPeek(out _, out lowerBound);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(neighborIds);
            }
        }
        return results;
    }
}
