using System;
using System.Numerics;
using Xunit;

namespace Cloud.Unum.USearch.Tests;

public class MetricsTests
{
    [Fact]
    public void TestL2sq()
    {
        float[] x = { 1.0f, 2.0f, 3.0f };
        float[] y = { 4.0f, 5.0f, 6.0f };
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        float expected = 27.0f;
        float actual = Metrics.L2sq<float>(x, y);
        Assert.Equal(expected, actual, 0.0001f);
    }

    [Fact]
    public void TestCosine()
    {
        // Orthogonal vectors, similarity 0, distance 1
        float[] x = { 1.0f, 0.0f };
        float[] y = { 0.0f, 1.0f };
        float expected = 1.0f;
        float actual = Metrics.Cosine<float>(x, y);
        Assert.Equal(expected, actual, 0.0001f);

        // Same vectors, similarity 1, distance 0
        x = new float[] { 1.0f, 0.0f };
        y = new float[] { 1.0f, 0.0f };
        expected = 0.0f;
        actual = Metrics.Cosine<float>(x, y);
        Assert.Equal(expected, actual, 0.0001f);

        // Opposite vectors, similarity -1, distance 2
        x = new float[] { 1.0f, 0.0f };
        y = new float[] { -1.0f, 0.0f };
        expected = 2.0f;
        actual = Metrics.Cosine<float>(x, y);
        Assert.Equal(expected, actual, 0.0001f);
    }

    [Fact]
    public void TestIP()
    {
        // Inner product distance = 1 - dot
        float[] x = { 1.0f, 2.0f };
        float[] y = { 3.0f, 4.0f };
        // dot = 1*3 + 2*4 = 3 + 8 = 11
        // distance = 1 - 11 = -10
        float expected = -10.0f;
        float actual = Metrics.IP<float>(x, y);
        Assert.Equal(expected, actual, 0.0001f);
    }

    [Fact]
    public void TestGenericDouble()
    {
        double[] x = { 1.0, 2.0, 3.0 };
        double[] y = { 4.0, 5.0, 6.0 };
        double expected = 27.0;
        double actual = Metrics.L2sq<double>(x, y);
        Assert.Equal(expected, actual, 0.0001);
    }
}
