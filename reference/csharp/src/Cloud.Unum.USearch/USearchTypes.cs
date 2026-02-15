using System;
using System.Runtime.InteropServices;

namespace Cloud.Unum.USearch;

/// <summary>
/// Specifies the kind of metric to be used for distance calculation between vectors.
/// </summary>
public enum MetricKind : uint
{
    /// <summary>
    /// Unknown metric kind.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// Cosine similarity.
    /// </summary>
    Cos = 1,

    /// <summary>
    /// Inner product.
    /// </summary>
    Ip = 2,

    /// <summary>
    /// Squared Euclidean distance.
    /// </summary>
    L2sq = 3,

    /// <summary>
    /// Haversine distance.
    /// </summary>
    Haversine = 4,

    /// <summary>
    /// Kullback-Leibler divergence.
    /// </summary>
    Divergence = 5,

    /// <summary>
    /// Pearson correlation.
    /// </summary>
    Pearson = 6,

    /// <summary>
    /// Jaccard index.
    /// </summary>
    Jaccard = 7,

    /// <summary>
    /// Hamming distance.
    /// </summary>
    Hamming = 8,

    /// <summary>
    /// Tanimoto coefficient.
    /// </summary>
    Tanimoto = 9,

    /// <summary>
    /// Sørensen-Dice coefficient.
    /// </summary>
    Sorensen = 10,
}

/// <summary>
/// Specifies the kind of scalar used for quantization of vector data during indexing.
/// </summary>
public enum ScalarKind : uint
{
    /// <summary>
    /// Unknown scalar kind.
    /// </summary>
    Unknown = 0,

    /// <summary>
    /// 32-bit floating point.
    /// </summary>
    Float32 = 1,

    /// <summary>
    /// 64-bit floating point.
    /// </summary>
    Float64 = 2,

    /// <summary>
    /// 16-bit floating point.
    /// </summary>
    Float16 = 3,

    /// <summary>
    /// 8-bit integer.
    /// </summary>
    Int8 = 4,

    /// <summary>
    /// 1-bit binary.
    /// </summary>
    Bits1 = 5,
}

/// <summary>
/// Represents the initialization options for creating a USearch index.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct IndexOptions
{
    /// <summary>
    /// The metric kind used for distance calculation between vectors.
    /// </summary>
    public MetricKind metric_kind;

    /// <summary>
    /// The optional custom metric function for distance calculation between vectors. Not supported yet.
    /// </summary>
    public IntPtr metric;

    /// <summary>
    /// The scalar kind used for quantization of vector data during indexing.
    /// </summary>
    public ScalarKind quantization;

    /// <summary>
    /// The number of dimensions in the vectors to be indexed.
    /// </summary>
    public ulong dimensions;

    /// <summary>
    /// The optional connectivity parameter that limits connections-per-node in the graph.
    /// </summary>
    public ulong connectivity;

    /// <summary>
    /// The optional expansion factor used for index construction when adding vectors.
    /// </summary>
    public ulong expansion_add;

    /// <summary>
    /// The optional expansion factor used for index construction during search operations.
    /// </summary>
    public ulong expansion_search;

    /// <summary>
    /// Indicates whether multiple vectors can map to the same key.
    /// </summary>
    [MarshalAs(UnmanagedType.Bool)]
    public bool multi;

    /// <summary>
    /// Initializes a new instance of the IndexOptions struct with specified parameters.
    /// </summary>
    /// <param name="metricKind">The metric kind used for distance calculation between vectors.</param>
    /// <param name="metric">The optional custom metric function for distance calculation between vectors. Not supported yet.</param>
    /// <param name="quantization">The scalar kind used for quantization of vector data during indexing.</param>
    /// <param name="dimensions">The number of dimensions in the vectors to be indexed.</param>
    /// <param name="connectivity">The optional connectivity parameter that limits connections-per-node in the graph.</param>
    /// <param name="expansionAdd">The optional expansion factor used for index construction when adding vectors.</param>
    /// <param name="expansionSearch">The optional expansion factor used for index construction during search operations.</param>
    /// <param name="multi">Indicates whether multiple vectors can map to the same key.</param>
    public IndexOptions(
        MetricKind metricKind = MetricKind.Unknown,
        IntPtr metric = default,
        ScalarKind quantization = ScalarKind.Unknown,
        ulong dimensions = 0,
        ulong connectivity = 0,
        ulong expansionAdd = 0,
        ulong expansionSearch = 0,
        bool multi = false
    )
    {
        this.metric_kind = metricKind;
        this.metric = metric;
        this.quantization = quantization;
        this.dimensions = dimensions;
        this.connectivity = connectivity;
        this.expansion_add = expansionAdd;
        this.expansion_search = expansionSearch;
        this.multi = multi;
    }
}
