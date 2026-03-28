using System;
using System.Numerics;
using System.Numerics.Tensors;

namespace Cloud.Unum.USearch;

public static class Metrics
{
    public static T Calculate<T>(MetricKind metric, ReadOnlySpan<T> x, ReadOnlySpan<T> y) where T : INumberBase<T>, IRootFunctions<T>
    {
        switch (metric)
        {
            case MetricKind.L2sq:
                return L2sq(x, y);
            case MetricKind.Cos:
                return Cosine(x, y);
            case MetricKind.Ip:
                return IP(x, y);
            default:
                throw new ArgumentException($"Unsupported metric: {metric}");
        }
    }

    public static T L2sq<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y) where T : INumberBase<T>
    {
        if (x.Length != y.Length)
            throw new ArgumentException("Lengths must match");

        T sum = T.Zero;
        for (int i = 0; i < x.Length; i++)
        {
            T diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }

    public static T Cosine<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y) where T : INumberBase<T>, IRootFunctions<T>
    {
        // Cosine distance = 1 - CosineSimilarity
        return T.One - TensorPrimitives.CosineSimilarity(x, y);
    }

    public static T IP<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y) where T : INumberBase<T>
    {
        // Inner Product distance = 1 - DotProduct
        return T.One - TensorPrimitives.Dot(x, y);
    }

    // Helper to get distance function delegate
    public delegate T DistanceFunction<T>(ReadOnlySpan<T> x, ReadOnlySpan<T> y);

    public static DistanceFunction<T> GetDistanceFunction<T>(MetricKind metric) where T : INumberBase<T>, IRootFunctions<T>
    {
        return metric switch
        {
            MetricKind.L2sq => L2sq,
            MetricKind.Cos => Cosine,
            MetricKind.Ip => IP,
            _ => throw new ArgumentException($"Unsupported metric: {metric}")
        };
    }
}
